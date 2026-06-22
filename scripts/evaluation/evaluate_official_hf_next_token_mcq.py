#!/usr/bin/env python3
"""Fast zero-shot MCQ evaluation for normalized official HF benchmark rows."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import sentencepiece as spm
import torch
from transformers import AutoModelForCausalLM


PROJECT = Path("/DATA/castle/projects/make_llm")
CASTLE_ROOT = Path("/DATA/castle").resolve()
DEFAULT_OUTPUT_ROOT = PROJECT / "results/03_evals/official_hf_scores_fast"
BENCHMARK_FILES = {
    "mmlu": "mmlu.jsonl",
    "mmlu_pro": "mmlu_pro.jsonl",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT))
    except ValueError:
        return str(path)


def require_under_castle(path: Path, label: str, *, must_exist: bool = False) -> Path:
    resolved = path.expanduser().resolve(strict=must_exist)
    if not resolved.is_relative_to(CASTLE_ROOT):
        raise SystemExit(f"{label} must stay under {CASTLE_ROOT}: {path} -> {resolved}")
    return resolved


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fp:
        for line in fp:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def benchmark_file(benchmark_root: Path, benchmark_id: str) -> Path:
    filename = BENCHMARK_FILES.get(benchmark_id, f"{benchmark_id}.jsonl")
    path = benchmark_root / filename
    if not path.exists():
        known = sorted(item.stem for item in benchmark_root.glob("*.jsonl"))
        raise SystemExit(
            f"Benchmark file not found for {benchmark_id}: {path}. "
            f"Available benchmark ids under {benchmark_root}: {known}"
        )
    return path


def format_prompt(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") or {}
    language = metadata.get("language")
    if language == "ko":
        parts = [
            "다음 문제의 정답을 고르세요. 답은 보기 기호 하나만 쓰세요.",
            "",
            "문제:",
            str(row.get("question") or "").strip(),
            "",
            "보기:",
        ]
        answer_header = "정답:"
    else:
        parts = [
            "Choose the correct answer. Write only the option letter.",
            "",
            "Question:",
            str(row.get("question") or "").strip(),
            "",
            "Choices:",
        ]
        answer_header = "Answer:"
    choices = row.get("choices") or []
    labels = row.get("choice_labels") or []
    for label, choice in zip(labels, choices):
        parts.append(f"{label}. {choice}")
    parts.extend(["", answer_header])
    return "\n".join(parts)


def encode(sp: spm.SentencePieceProcessor, text: str) -> list[int]:
    return list(sp.encode(text, out_type=int))


def summarize(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(predictions)
    correct = sum(1 for row in predictions if row["correct"])
    category_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in predictions:
        category = str(row.get("category") or "unknown")
        category_counts[category]["total"] += 1
        if row["correct"]:
            category_counts[category]["correct"] += 1
    return {
        "examples": total,
        "correct": correct,
        "accuracy": correct / total if total else None,
        "category_accuracy": {
            category: {
                "examples": counts["total"],
                "correct": counts["correct"],
                "accuracy": counts["correct"] / counts["total"] if counts["total"] else None,
            }
            for category, counts in sorted(category_counts.items())
        },
    }


@torch.inference_mode()
def evaluate_rows(
    *,
    rows: list[dict[str, Any]],
    benchmark_id: str,
    model: AutoModelForCausalLM,
    sp: spm.SentencePieceProcessor,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
    max_length: int,
    pad_token_id: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    label_cache: dict[str, list[int]] = {}
    for row in rows:
        for label in row.get("choice_labels") or []:
            label_cache.setdefault(str(label), encode(sp, f" {label}"))
    multi_token_labels = {label: ids for label, ids in label_cache.items() if len(ids) != 1}
    if multi_token_labels:
        raise SystemExit(f"Label tokens must be single-token for fast next-token eval: {multi_token_labels}")

    predictions: list[dict[str, Any]] = []
    batch_size = max(1, int(batch_size))
    started = time.time()
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        prompt_ids = [encode(sp, format_prompt(row))[-max_length:] for row in batch_rows]
        max_len = max(len(ids) for ids in prompt_ids)
        input_ids = []
        attention_mask = []
        for ids in prompt_ids:
            pad_len = max_len - len(ids)
            input_ids.append([pad_token_id] * pad_len + ids)
            attention_mask.append([0] * pad_len + [1] * len(ids))
        input_tensor = torch.tensor(input_ids, dtype=torch.long, device=device)
        mask_tensor = torch.tensor(attention_mask, dtype=torch.long, device=device)
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda"):
            logits = model(input_ids=input_tensor, attention_mask=mask_tensor).logits
        last_positions = mask_tensor.sum(dim=1) - 1
        for row_idx, row in enumerate(batch_rows):
            log_probs = torch.log_softmax(logits[row_idx, last_positions[row_idx]].float(), dim=-1)
            scores = {
                str(label): float(log_probs[label_cache[str(label)][0]].detach().cpu())
                for label in row.get("choice_labels") or []
            }
            prediction = max(scores, key=scores.get) if scores else None
            answer_label = row.get("answer_label")
            predictions.append(
                {
                    "benchmark_id": benchmark_id,
                    "example_id": row.get("example_id"),
                    "category": row.get("category"),
                    "subcategory": row.get("subcategory"),
                    "answer_label": answer_label,
                    "prediction": prediction,
                    "correct": prediction == answer_label,
                    "scores": scores,
                }
            )
        if (start + len(batch_rows)) % 1000 == 0 or start + len(batch_rows) == len(rows):
            elapsed = time.time() - started
            done = start + len(batch_rows)
            speed = done / elapsed if elapsed > 0 else math.nan
            print(
                json.dumps(
                    {
                        "event": "progress",
                        "benchmark": benchmark_id,
                        "done": done,
                        "total": len(rows),
                        "rows_per_second": speed,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    return predictions, {"label_token_ids": {label: ids[0] for label, ids in sorted(label_cache.items())}}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--benchmark", action="append", required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-alias", required=True)
    parser.add_argument("--checkpoint-alias", required=True)
    parser.add_argument("--eval-tier", default="full_official_hf_next_token")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=4096)
    args = parser.parse_args()

    args.checkpoint_dir = require_under_castle(args.checkpoint_dir, "--checkpoint-dir", must_exist=True)
    args.tokenizer = require_under_castle(args.tokenizer, "--tokenizer", must_exist=True)
    args.benchmark_root = require_under_castle(args.benchmark_root, "--benchmark-root", must_exist=True)
    args.output_root = require_under_castle(args.output_root, "--output-root")

    output_dir = args.output_root / args.run_alias / args.checkpoint_alias / f"{args.eval_tier}_{timestamp()}"
    output_dir.mkdir(parents=True, exist_ok=True)
    run_config = {
        "started_at_utc": utc_now(),
        "script": rel(Path(__file__)),
        "checkpoint_dir": rel(args.checkpoint_dir),
        "tokenizer": rel(args.tokenizer),
        "benchmark_root": rel(args.benchmark_root),
        "benchmarks": args.benchmark,
        "output_dir": rel(output_dir),
        "run_alias": args.run_alias,
        "checkpoint_alias": args.checkpoint_alias,
        "eval_tier": args.eval_tier,
        "device": args.device,
        "dtype": args.dtype,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "scoring": "zero_shot_next_token_option_letter_logprob",
    }
    (output_dir / "run_config.json").write_text(json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    sp = spm.SentencePieceProcessor(model_file=str(args.tokenizer))
    pad_token_id = sp.pad_id() if sp.pad_id() >= 0 else max(sp.eos_id(), 0)
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint_dir,
        torch_dtype=dtype if device.type == "cuda" else torch.float32,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()

    all_predictions: list[dict[str, Any]] = []
    benchmark_summaries: dict[str, Any] = {}
    token_metadata: dict[str, Any] = {}
    started = time.time()
    for benchmark in args.benchmark:
        rows = read_jsonl(benchmark_file(args.benchmark_root, benchmark))
        predictions, meta = evaluate_rows(
            rows=rows,
            benchmark_id=benchmark,
            model=model,
            sp=sp,
            device=device,
            dtype=dtype,
            batch_size=args.batch_size,
            max_length=args.max_length,
            pad_token_id=pad_token_id,
        )
        all_predictions.extend(predictions)
        benchmark_summaries[benchmark] = summarize(predictions)
        token_metadata[benchmark] = meta

    predictions_path = output_dir / "predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8") as fp:
        for row in all_predictions:
            fp.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    summary = {
        **run_config,
        "ended_at_utc": utc_now(),
        "elapsed_seconds": time.time() - started,
        "device_resolved": str(device),
        "model_parameters": sum(param.numel() for param in model.parameters()),
        "predictions_path": rel(predictions_path),
        "benchmarks_summary": benchmark_summaries,
        "token_metadata": token_metadata,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
