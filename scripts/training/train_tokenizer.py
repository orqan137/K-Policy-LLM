#!/usr/bin/env python3
"""Train a SentencePiece tokenizer for make_llm."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT = Path("/DATA/castle/projects/make_llm")
DEFAULT_NUM_THREADS = 24
DEFAULT_INPUT = PROJECT / "data/02_clean/tokenizer/train.txt"
DEFAULT_OUTPUT_DIR = PROJECT / "data/04_tokenizer"
DEFAULT_USER_DEFINED_SYMBOLS = [
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|end|>",
    "<|context|>",
    "<|question|>",
    "<|answer|>",
    "<|evidence|>",
    "<|think|>",
    "<|final|>",
    "<|verify|>",
    "<|claim|>",
    "<|stance|>",
    "<|url|>",
    "<|email|>",
    "<|phone|>",
    "<|rrn|>",
    "<|address|>",
    "<|person|>",
    "<|org|>",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-prefix", default="make_llm_spm_bpe_64k")
    parser.add_argument("--vocab-size", type=int, default=64000)
    parser.add_argument("--input-sentence-size", type=int, default=2_000_000)
    parser.add_argument("--max-sentence-length", type=int, default=131072)
    parser.add_argument("--character-coverage", type=float, default=0.9995)
    parser.add_argument("--normalization-rule-name", default="identity")
    parser.add_argument("--add-dummy-prefix", action="store_true")
    parser.add_argument("--hard-vocab-limit", action="store_true")
    parser.add_argument("--user-defined-symbols", nargs="*", default=DEFAULT_USER_DEFINED_SYMBOLS)
    parser.add_argument("--num-threads", type=int, default=DEFAULT_NUM_THREADS)
    args = parser.parse_args()

    try:
        import sentencepiece as spm
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "sentencepiece is not installed in the canonical env. "
            "Run: bash /DATA/castle/projects/make_llm/scripts/sync_env.sh"
        ) from exc

    if not args.input.exists():
        raise SystemExit(f"missing tokenizer input: {args.input}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_prefix = args.output_dir / args.model_prefix
    spm.SentencePieceTrainer.Train(
        input=str(args.input),
        model_prefix=str(model_prefix),
        model_type="bpe",
        vocab_size=args.vocab_size,
        character_coverage=args.character_coverage,
        byte_fallback=True,
        normalization_rule_name=args.normalization_rule_name,
        add_dummy_prefix=args.add_dummy_prefix,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        pad_piece="<pad>",
        unk_piece="<unk>",
        bos_piece="<bos>",
        eos_piece="<eos>",
        user_defined_symbols=args.user_defined_symbols,
        input_sentence_size=args.input_sentence_size,
        max_sentence_length=args.max_sentence_length,
        shuffle_input_sentence=True,
        train_extremely_large_corpus=True,
        hard_vocab_limit=args.hard_vocab_limit,
        num_threads=args.num_threads,
    )
    model_path = model_prefix.with_suffix(".model")
    vocab_path = model_prefix.with_suffix(".vocab")
    manifest = {
        "input": str(args.input),
        "model": str(model_path),
        "vocab": str(vocab_path),
        "model_type": "bpe",
        "vocab_size": args.vocab_size,
        "character_coverage": args.character_coverage,
        "byte_fallback": True,
        "normalization_rule_name": args.normalization_rule_name,
        "add_dummy_prefix": args.add_dummy_prefix,
        "hard_vocab_limit": args.hard_vocab_limit,
        "user_defined_symbols": args.user_defined_symbols,
        "input_sentence_size": args.input_sentence_size,
        "max_sentence_length": args.max_sentence_length,
        "num_threads": args.num_threads,
    }
    model_prefix.with_suffix(".manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(model_path)
    print(vocab_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
