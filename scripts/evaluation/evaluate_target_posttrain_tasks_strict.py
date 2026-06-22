#!/usr/bin/env python3
"""Generate and score the full target post-training diagnostic set.

This script preserves the same target-eval contract as
``evaluate_target_posttrain_tasks.py`` but uses a stricter report-facing score:

- no free section credit when a row has no required sections;
- whitespace-normalized term matching;
- short-answer rows are not penalized only because the correct answer is short;
- numeric rows check the answer value and, when available, direction/value cues.
"""

from __future__ import annotations

import argparse
import json
import re
import time
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import sentencepiece as spm
import torch
from transformers import AutoModelForCausalLM


PROJECT = Path("/DATA/castle/projects/make_llm")
DEFAULT_TOKENIZER = (
    PROJECT / "data/04_tokenizer/v0_0_9_80k_15shard_official_2026-06-08_092211/tokenizer.model"
)
NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def now_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT))
    except ValueError:
        return str(path)


def iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as fp:
        for line in fp:
            if line.strip():
                yield json.loads(line)


def load_target_rows(target_eval_dir: Path, limit_per_task: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    per_task = Counter()
    for path in sorted(target_eval_dir.glob("*.jsonl")):
        for row in iter_jsonl(path):
            task = str(row.get("task_type") or path.stem)
            if limit_per_task is not None and per_task[task] >= limit_per_task:
                continue
            row = dict(row)
            row["_target_file"] = path.name
            rows.append(row)
            per_task[task] += 1
    if not rows:
        raise SystemExit(f"empty target eval dir: {target_eval_dir}")
    return rows


def load_generations(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for row in iter_jsonl(path):
        row_id = str(row.get("id") or row.get("example_id") or "")
        completion = row.get("completion")
        if row_id and isinstance(completion, str):
            out[row_id] = completion
    return out


def clean_completion(text: str) -> str:
    return text.replace("<|end|>", "").strip()


def norm(text: str) -> str:
    text = unicodedata.normalize("NFKC", clean_completion(str(text)))
    return re.sub(r"\s+", " ", text).strip().lower()


def compact(text: str) -> str:
    return re.sub(r"\s+", "", norm(text))


def contains_term(text_norm: str, text_compact: str, term: str) -> bool:
    term_norm = norm(term)
    if not term_norm:
        return False
    return term_norm in text_norm or compact(term_norm) in text_compact


def term_hits(text: str, terms: list[str]) -> list[str]:
    text_norm = norm(text)
    text_compact = compact(text)
    return [term for term in terms if contains_term(text_norm, text_compact, term)]


def ratio_score(found: int, total: int) -> float:
    if total <= 0:
        return 1.0
    return found / total


def numbers(text: str) -> list[float]:
    values = []
    for match in NUMBER_RE.finditer(clean_completion(text).replace(",", "")):
        try:
            values.append(float(match.group(0)))
        except ValueError:
            continue
    return values


def numeric_close(values: list[float], expected: float, tolerance: float) -> bool:
    return any(abs(value - expected) <= tolerance for value in values)


def close_value_present(text: str, value: float, tolerance: float) -> bool:
    return numeric_close(numbers(text), value, tolerance)


def short_answer_row(row: dict[str, Any]) -> bool:
    task = str(row.get("task_type") or "")
    prompt = str(row.get("prompt") or "")
    reference = clean_completion(str(row.get("reference_answer") or ""))
    return (
        "short_direct_answer" in task
        or "단답" in prompt
        or "한 문장" in prompt
        or (reference and len(reference) < 20 and not row.get("required_sections"))
    )


def length_score(row: dict[str, Any], text: str) -> float:
    size = len(clean_completion(text))
    if size <= 0:
        return 0.0
    if short_answer_row(row):
        return 1.0 if 2 <= size <= 800 else 0.0
    return 1.0 if 20 <= size <= 2500 else 0.0


def weighted_score(parts: list[tuple[str, float, float]]) -> tuple[float, dict[str, float]]:
    active = [(name, weight, value) for name, weight, value in parts if weight > 0]
    total = sum(weight for _, weight, _ in active)
    if total <= 0:
        return 1.0, {}
    contributions = {name: (weight / total) * value for name, weight, value in active}
    return sum(contributions.values()), contributions


def numeric_answer_score(row: dict[str, Any], text: str) -> tuple[float, dict[str, float]]:
    key = row.get("answer_key") if isinstance(row.get("answer_key"), dict) else {}
    expected = float(key.get("value", 0.0))
    tolerance = float(key.get("tolerance", 0.05))
    value_hit = 1.0 if close_value_present(text, expected, tolerance) else 0.0

    cue_parts: list[tuple[str, float, float]] = [("answer_value", 0.70, value_hit)]
    if key.get("higher"):
        higher_hit = 1.0 if contains_term(norm(text), compact(text), str(key["higher"])) else 0.0
        cue_parts.append(("direction", 0.15, higher_hit))
    if key.get("left_value") is not None and key.get("right_value") is not None:
        left_hit = close_value_present(text, float(key["left_value"]), tolerance)
        right_hit = close_value_present(text, float(key["right_value"]), tolerance)
        cue_parts.append(("input_values", 0.15, 1.0 if left_hit and right_hit else 0.0))
    elif len(cue_parts) == 1:
        cue_parts[0] = ("answer_value", 1.0, value_hit)
    return weighted_score(cue_parts)


def score_completion(row: dict[str, Any], completion: str) -> dict[str, Any]:
    text = clean_completion(completion)
    required_sections = [str(item) for item in row.get("required_sections") or []]
    required_terms = [str(item) for item in row.get("required_terms") or []]
    forbidden_terms = [str(item) for item in row.get("forbidden_terms") or []]

    section_hits = term_hits(text, required_sections)
    required_hits = term_hits(text, required_terms)
    forbidden_hits = term_hits(text, forbidden_terms)

    section_score = ratio_score(len(section_hits), len(required_sections))
    required_score = ratio_score(len(required_hits), len(required_terms))
    forbidden_score = 1.0 if not forbidden_hits else 0.0
    len_score = length_score(row, text)

    key = row.get("answer_key") if isinstance(row.get("answer_key"), dict) else None
    if key:
        num_score, num_parts = numeric_answer_score(row, text)
        unit = str(key.get("unit", ""))
        unit_score = 1.0 if unit and contains_term(norm(text), compact(text), unit) else 0.0
        overall, contributions = weighted_score(
            [
                ("numeric_answer", 0.45, num_score),
                ("unit", 0.15 if unit else 0.0, unit_score),
                ("required_terms", 0.15 if required_terms else 0.0, required_score),
                ("forbidden_absence", 0.15 if forbidden_terms else 0.0, forbidden_score),
                ("length", 0.10, len_score),
            ]
        )
    else:
        num_score = None
        unit_score = None
        num_parts = {}
        overall, contributions = weighted_score(
            [
                ("sections", 0.35 if required_sections else 0.0, section_score),
                ("required_terms", 0.25 if required_terms else 0.0, required_score),
                ("forbidden_absence", 0.25 if forbidden_terms else 0.0, forbidden_score),
                ("length", 0.15, len_score),
            ]
        )

    return {
        "example_id": row.get("example_id"),
        "task_type": row.get("task_type"),
        "source_group": row.get("source_group"),
        "target_file": row.get("_target_file"),
        "overall": round(float(overall), 6),
        "section_score": round(section_score, 6),
        "required_term_score": round(required_score, 6),
        "forbidden_absence_score": round(forbidden_score, 6),
        "length_score": round(len_score, 6),
        "numeric_answer_score": None if num_score is None else round(float(num_score), 6),
        "unit_score": unit_score,
        "score_contributions": {k: round(float(v), 6) for k, v in contributions.items()},
        "numeric_answer_parts": {k: round(float(v), 6) for k, v in num_parts.items()},
        "section_hits": section_hits,
        "required_term_hits": required_hits,
        "forbidden_hits": forbidden_hits,
        "completion_chars": len(text),
    }


def decode_new_text(sp: spm.SentencePieceProcessor, output_ids: list[int], start_idx: int, stop_ids: set[int]) -> str:
    new_ids = []
    for token_id in output_ids[start_idx:]:
        if token_id in stop_ids:
            break
        new_ids.append(token_id)
    return sp.decode(new_ids).strip()


def encode_prompt(sp: spm.SentencePieceProcessor, prompt: str, max_input_tokens: int) -> list[int]:
    prompt_ids = [sp.bos_id()] + list(sp.encode(prompt, out_type=int))
    if len(prompt_ids) > max_input_tokens:
        prompt_ids = prompt_ids[-max_input_tokens:]
    return prompt_ids


def batches(items: list[dict[str, Any]], batch_size: int):
    for idx in range(0, len(items), batch_size):
        yield items[idx : idx + batch_size]


def generate_completions(
    *,
    checkpoint_dir: Path,
    tokenizer_path: Path,
    rows: list[dict[str, Any]],
    output_path: Path,
    device_name: str,
    dtype_name: str,
    max_input_tokens: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    seed: int,
    batch_size: int,
) -> dict[str, str]:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype_name]
    sp = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_dir,
        torch_dtype=dtype if device.type == "cuda" else torch.float32,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()

    pad_id = sp.pad_id()
    if pad_id < 0:
        pad_id = sp.eos_id()
    end_id = sp.piece_to_id("<|end|>")
    stop_ids = {sp.eos_id()}
    if end_id >= 0:
        stop_ids.add(end_id)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    completions: dict[str, str] = {}
    processed = 0
    started = time.time()
    with output_path.open("w", encoding="utf-8") as fp:
        for chunk in batches(rows, batch_size):
            encoded = [encode_prompt(sp, str(row["prompt"]), max_input_tokens) for row in chunk]
            max_len = max(len(ids) for ids in encoded)
            input_rows = []
            mask_rows = []
            for ids in encoded:
                pad_len = max_len - len(ids)
                input_rows.append([pad_id] * pad_len + ids)
                mask_rows.append([0] * pad_len + [1] * len(ids))
            input_tensor = torch.tensor(input_rows, dtype=torch.long, device=device)
            attention_mask = torch.tensor(mask_rows, dtype=torch.long, device=device)
            kwargs: dict[str, Any] = {
                "max_new_tokens": max_new_tokens,
                "eos_token_id": sorted(stop_ids),
                "pad_token_id": pad_id,
                "use_cache": True,
                "repetition_penalty": repetition_penalty,
                "attention_mask": attention_mask,
            }
            if temperature > 0:
                kwargs.update({"do_sample": True, "temperature": temperature, "top_p": top_p, "top_k": top_k})
            else:
                kwargs.update({"do_sample": False})
            with torch.inference_mode():
                output = model.generate(input_tensor, **kwargs)
            output_list = output.detach().cpu().tolist()
            for row, output_ids, prompt_ids in zip(chunk, output_list, encoded):
                completion = decode_new_text(sp, output_ids, max_len, stop_ids)
                row_id = str(row["example_id"])
                completions[row_id] = completion
                fp.write(
                    json.dumps(
                        {
                            "id": row_id,
                            "task_type": row.get("task_type"),
                            "source_group": row.get("source_group"),
                            "target_file": row.get("_target_file"),
                            "prompt": row.get("prompt"),
                            "completion": completion,
                            "input_tokens": len(prompt_ids),
                            "output_tokens": max(0, len(output_ids) - max_len),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            processed += len(chunk)
            elapsed = max(time.time() - started, 1e-6)
            print(
                json.dumps(
                    {
                        "event": "progress",
                        "processed": processed,
                        "total": len(rows),
                        "examples_per_sec": round(processed / elapsed, 4),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    return completions


def summarize(scored: list[dict[str, Any]]) -> dict[str, Any]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_file: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scored:
        by_task[str(row.get("task_type") or "unknown")].append(row)
        by_file[str(row.get("target_file") or "unknown")].append(row)

    def avg(rows: list[dict[str, Any]], key: str) -> float | None:
        vals = [row[key] for row in rows if row.get(key) is not None]
        return round(sum(float(v) for v in vals) / len(vals), 6) if vals else None

    def build(rows_by_name: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
        return {
            name: {
                "count": len(rows),
                "overall": avg(rows, "overall"),
                "section_score": avg(rows, "section_score"),
                "required_term_score": avg(rows, "required_term_score"),
                "forbidden_absence_score": avg(rows, "forbidden_absence_score"),
                "length_score": avg(rows, "length_score"),
                "numeric_answer_score": avg(rows, "numeric_answer_score"),
                "unit_score": avg(rows, "unit_score"),
            }
            for name, rows in sorted(rows_by_name.items())
        }

    return {
        "count": len(scored),
        "overall": avg(scored, "overall"),
        "tasks": build(by_task),
        "files": build(by_file),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-eval-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--generations-jsonl", type=Path, default=None)
    parser.add_argument("--score-references", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--limit-per-task", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=220)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.score_references and args.generations_jsonl is None and args.checkpoint_dir is None:
        raise SystemExit("provide --checkpoint-dir, --generations-jsonl, or --score-references")

    started = time.time()
    target_eval_dir = args.target_eval_dir.resolve()
    rows = load_target_rows(target_eval_dir, args.limit_per_task)
    run_id = now_id()
    output_dir = args.output_dir or PROJECT / "results/03_evals/target_posttrain_tasks" / f"target_eval_strict_{run_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.score_references:
        completions = {str(row["example_id"]): str(row.get("reference_answer") or "") for row in rows}
        generations_path = output_dir / "reference_answers.jsonl"
        with generations_path.open("w", encoding="utf-8") as fp:
            for row in rows:
                fp.write(
                    json.dumps(
                        {
                            "id": row["example_id"],
                            "task_type": row.get("task_type"),
                            "source_group": row.get("source_group"),
                            "target_file": row.get("_target_file"),
                            "completion": row.get("reference_answer") or "",
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    elif args.generations_jsonl is not None:
        generations_path = args.generations_jsonl.resolve()
        completions = load_generations(generations_path)
    else:
        generations_path = output_dir / "generations.jsonl"
        completions = generate_completions(
            checkpoint_dir=args.checkpoint_dir.resolve(),
            tokenizer_path=args.tokenizer.resolve(),
            rows=rows,
            output_path=generations_path,
            device_name=args.device,
            dtype_name=args.dtype,
            max_input_tokens=args.max_input_tokens,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
            seed=args.seed,
            batch_size=args.batch_size,
        )

    scored = []
    missing = []
    for row in rows:
        row_id = str(row["example_id"])
        completion = completions.get(row_id)
        if completion is None:
            missing.append(row_id)
            continue
        scored.append(score_completion(row, completion))

    summary = {
        "created_at": utc_now(),
        "status": "pass" if not missing else "warn_missing_generations",
        "target_eval_dir": rel(target_eval_dir),
        "target_files": sorted(path.name for path in target_eval_dir.glob("*.jsonl")),
        "checkpoint_dir": rel(args.checkpoint_dir) if args.checkpoint_dir else None,
        "tokenizer": rel(args.tokenizer.resolve()),
        "generations_jsonl": rel(generations_path),
        "score_references": args.score_references,
        "limit_per_task": args.limit_per_task,
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "missing_generations": missing[:100],
        "missing_generation_count": len(missing),
        "elapsed_seconds": time.time() - started,
        "score_summary": summarize(scored),
        "metric_contract": {
            "name": "target_eval_strict_v2",
            "purpose": "project-internal target-task generation diagnostic, not a public benchmark or human score",
            "text_tasks": "active weights are normalized; sections 0.35 only when required_sections exist, required terms 0.25, forbidden absence 0.25, length 0.15",
            "numeric_tasks": "active weights are normalized; numeric answer 0.45, unit 0.15, required terms 0.15, forbidden absence 0.15, length 0.10",
            "matching": "Unicode NFKC and whitespace-normalized substring matching",
            "short_answer_length": "rows with short reference answers or explicit short-answer prompts use a 2-800 character length window",
        },
    }
    with (output_dir / "per_example_scores.jsonl").open("w", encoding="utf-8") as fp:
        for row in scored:
            fp.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if not missing else 1


if __name__ == "__main__":
    raise SystemExit(main())
