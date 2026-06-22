#!/usr/bin/env python3
"""Summarize one official evaluation suite directory."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


PROJECT = Path("/DATA/castle/projects/make_llm")
CASTLE_ROOT = Path("/DATA/castle").resolve()


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def require_under_castle(path: Path, label: str, *, must_exist: bool = False) -> Path:
    resolved = path.expanduser().resolve(strict=must_exist)
    if not resolved.is_relative_to(CASTLE_ROOT):
        raise SystemExit(f"{label} must stay under {CASTLE_ROOT}: {path} -> {resolved}")
    return resolved


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT))
    except ValueError:
        return str(path)


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fp:
        return json.load(fp)


def pct(value: Any) -> str:
    if value is None:
        return ""
    return f"{float(value) * 100:.2f}"


def scalar(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def infer_component(summary: dict[str, Any], path: Path) -> str:
    script = str(summary.get("script") or "")
    scoring = str(summary.get("scoring") or "")
    path_text = str(path)
    if "choice_text" in script or "choice_text" in scoring or "official_hf_choice_text" in path_text:
        return "choice_text_likelihood"
    if "evaluate_public_benchmarks" in script:
        return "public_light_mcq"
    if "generate_checkpoint_samples" in script or "qualitative" in path_text:
        return "qualitative_samples"
    if "next_token" in script or "next_token" in scoring or "official_hf_scores_fast" in path_text:
        return "official_next_token_mcq"
    return "unknown"


def rows_from_summary(path: Path, summary: dict[str, Any]) -> Iterable[dict[str, Any]]:
    component = infer_component(summary, path)
    common = {
        "component": component,
        "run_alias": scalar(summary.get("run_alias")),
        "checkpoint_alias": scalar(summary.get("checkpoint_alias")),
        "eval_tier": scalar(summary.get("eval_tier")),
        "summary_path": rel(path),
        "output_dir": scalar(summary.get("output_dir")),
        "started_at_utc": scalar(summary.get("started_at_utc")),
        "ended_at_utc": scalar(summary.get("ended_at_utc")),
        "elapsed_seconds": scalar(summary.get("elapsed_seconds")),
        "device": scalar(summary.get("device")),
        "dtype": scalar(summary.get("dtype")),
    }

    benchmark_summaries = summary.get("benchmarks_summary")
    if isinstance(benchmark_summaries, dict):
        for benchmark, metrics in sorted(benchmark_summaries.items()):
            if not isinstance(metrics, dict):
                continue
            if "accuracy" in metrics:
                yield {
                    **common,
                    "benchmark": benchmark,
                    "metric": "accuracy",
                    "examples": scalar(metrics.get("examples")),
                    "correct": scalar(metrics.get("correct")),
                    "value_pct": pct(metrics.get("accuracy")),
                }
                continue
            for key, metric_name in (("avg_logprob", "choice_text_avg_logprob"), ("sum_logprob", "choice_text_sum_logprob")):
                nested = metrics.get(key)
                if isinstance(nested, dict):
                    yield {
                        **common,
                        "benchmark": benchmark,
                        "metric": metric_name,
                        "examples": scalar(nested.get("examples")),
                        "correct": scalar(nested.get("correct")),
                        "value_pct": pct(nested.get("accuracy")),
                    }
        return

    examples = summary.get("examples")
    if examples is not None:
        yield {
            **common,
            "benchmark": "",
            "metric": "examples",
            "examples": scalar(examples),
            "correct": "",
            "value_pct": "",
        }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "component",
        "run_alias",
        "checkpoint_alias",
        "eval_tier",
        "benchmark",
        "metric",
        "examples",
        "correct",
        "value_pct",
        "summary_path",
        "output_dir",
        "started_at_utc",
        "ended_at_utc",
        "elapsed_seconds",
        "device",
        "dtype",
    ]
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_markdown(path: Path, rows: list[dict[str, Any]], manifest: dict[str, Any]) -> None:
    lines = [
        "# Official Evaluation Suite Summary",
        "",
        f"- generated_at_utc: `{manifest['generated_at_utc']}`",
        f"- target_name: `{manifest['target_name']}`",
        f"- suite_dir: `{manifest['suite_dir']}`",
        f"- checkpoint_dir: `{manifest['checkpoint_dir']}`",
        f"- tokenizer: `{manifest['tokenizer']}`",
        "",
    ]
    if not rows:
        lines.extend(["No summary rows found yet.", ""])
        path.write_text("\n".join(lines), encoding="utf-8")
        return

    lines.extend(
        [
            "| component | benchmark | metric | examples | correct | value_pct |",
            "| --- | --- | --- | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        lines.append(
            "| {component} | {benchmark} | {metric} | {examples} | {correct} | {value_pct} |".format(
                component=row.get("component", ""),
                benchmark=row.get("benchmark", ""),
                metric=row.get("metric", ""),
                examples=row.get("examples", ""),
                correct=row.get("correct", ""),
                value_pct=row.get("value_pct", ""),
            )
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite-dir", type=Path, required=True)
    parser.add_argument("--target-name", default="kfri-dmpmaker")
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--run-alias", default="")
    parser.add_argument("--checkpoint-alias", default="")
    parser.add_argument("--launch-decision", default="")
    parser.add_argument("--note", default="")
    args = parser.parse_args()

    suite_dir = require_under_castle(args.suite_dir, "--suite-dir", must_exist=True)
    checkpoint_dir = require_under_castle(args.checkpoint_dir, "--checkpoint-dir", must_exist=True)
    tokenizer = require_under_castle(args.tokenizer, "--tokenizer", must_exist=True)

    summary_files = sorted(
        path
        for path in suite_dir.rglob("summary.json")
        if path.name == "summary.json" and path.parent != suite_dir
    )
    rows: list[dict[str, Any]] = []
    parsed_summaries: list[dict[str, Any]] = []
    for summary_path in summary_files:
        summary = read_json(summary_path)
        parsed_summaries.append({"summary_path": rel(summary_path), "component": infer_component(summary, summary_path)})
        rows.extend(rows_from_summary(summary_path, summary))

    manifest = {
        "generated_at_utc": utc_now(),
        "target_name": args.target_name,
        "suite_dir": rel(suite_dir),
        "checkpoint_dir": rel(checkpoint_dir),
        "tokenizer": rel(tokenizer),
        "run_alias": args.run_alias,
        "checkpoint_alias": args.checkpoint_alias,
        "launch_decision": args.launch_decision,
        "note": args.note,
        "summary_files": parsed_summaries,
        "row_count": len(rows),
        "outputs": {
            "csv": rel(suite_dir / "suite_summary.csv"),
            "markdown": rel(suite_dir / "suite_summary.md"),
            "manifest": rel(suite_dir / "suite_manifest.json"),
        },
    }

    write_csv(suite_dir / "suite_summary.csv", rows)
    write_markdown(suite_dir / "suite_summary.md", rows, manifest)
    (suite_dir / "suite_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
