#!/usr/bin/env python3
"""Export a checkpoint with tokenizer and manifests as a HF-style bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


PROJECT = Path("/DATA/castle/projects/make_llm")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT))
    except ValueError:
        return str(path)


def copy_file(src: Path, dst: Path) -> dict:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return {"source": rel(src), "target": rel(dst), "sha256": sha256_file(dst), "bytes": dst.stat().st_size}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, default=None)
    parser.add_argument("--tokenizer-audit", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for name in ("config.json", "generation_config.json", "model.safetensors"):
        src = args.checkpoint_dir / name
        if not src.exists():
            raise SystemExit(f"missing checkpoint file: {src}")
        copied.append(copy_file(src, args.output_dir / name))
    for pattern in ("*.model", "*.vocab", "*.manifest.json"):
        for src in sorted(args.tokenizer_dir.glob(pattern)):
            copied.append(copy_file(src, args.output_dir / src.name))
    if args.dataset_manifest:
        copied.append(copy_file(args.dataset_manifest, args.output_dir / "dataset_manifest.json"))
    if args.tokenizer_audit:
        copied.append(copy_file(args.tokenizer_audit, args.output_dir / "tokenizer_audit.json"))

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "checkpoint_dir": rel(args.checkpoint_dir),
        "tokenizer_dir": rel(args.tokenizer_dir),
        "output_dir": rel(args.output_dir),
        "format": "hf_transformers_with_sentencepiece",
        "files": copied,
    }
    (args.output_dir / "export_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
