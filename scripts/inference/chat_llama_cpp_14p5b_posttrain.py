#!/usr/bin/env python3
"""Small terminal chat client for the local llama.cpp posttrain server."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_SYSTEM_PROMPT = (
    "너는 한국어로 자연스럽고 간결하게 대화하는 로컬 사후학습 언어모델이다. "
    "모르면 모른다고 말하고, 계산은 천천히 검산한다."
)


def build_prompt(system_prompt: str, turns: list[dict[str, str]], user_text: str) -> str:
    lines = [
        "<|system|>",
        system_prompt.strip(),
        "<|end|>",
        "",
    ]
    for turn in turns:
        role = turn["role"]
        if role == "user":
            lines.extend(["<|user|>", turn["content"].strip(), "<|end|>", ""])
        elif role == "assistant":
            content = turn["content"].strip()
            if not content.endswith("<|end|>"):
                content = f"{content}\n<|end|>"
            lines.extend(["<|assistant|>", content, ""])
    lines.extend(["<|user|>", user_text.strip(), "<|end|>", "", "<|assistant|>"])
    return "\n".join(lines)


def request_completion(
    *,
    base_url: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    repeat_penalty: float,
    timeout: float,
) -> str:
    payload = {
        "prompt": prompt,
        "n_predict": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "repeat_penalty": repeat_penalty,
        "stop": ["<|end|>", "<|user|>", "<|system|>", "</s>"],
        "stream": False,
    }
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/completion",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(f"server request failed: {exc}") from exc

    content = body.get("content")
    if not isinstance(content, str):
        raise RuntimeError(f"unexpected llama.cpp response: {body}")
    return content.strip()


def save_transcript(path: Path, system_prompt: str, turns: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "system_prompt": system_prompt,
        "turns": turns,
    }
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default="http://127.0.0.1:8622")
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--max-tokens", type=int, default=320)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repeat-penalty", type=float, default=1.08)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--save", type=Path, default=None, help="Optional transcript JSON path.")
    args = parser.parse_args()

    turns: list[dict[str, str]] = []
    print("make_llm 14.5B posttrain llama.cpp chat")
    print("Commands: /exit, /reset, /save [path]")
    print()

    while True:
        try:
            user_text = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_text:
            continue
        if user_text in {"/exit", "/quit"}:
            break
        if user_text == "/reset":
            turns.clear()
            print("history reset")
            continue
        if user_text.startswith("/save"):
            parts = user_text.split(maxsplit=1)
            save_path = Path(parts[1]) if len(parts) == 2 else args.save
            if save_path is None:
                save_path = Path("/DATA/castle/projects/make_llm/results/03_evals/chat_transcripts/llama_cpp_14p5b_posttrain_chat.json")
            save_transcript(save_path, args.system_prompt, turns)
            print(f"saved: {save_path}")
            continue

        prompt = build_prompt(args.system_prompt, turns, user_text)
        try:
            answer = request_completion(
                base_url=args.server,
                prompt=prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                repeat_penalty=args.repeat_penalty,
                timeout=args.timeout,
            )
        except RuntimeError as exc:
            print(f"error: {exc}", file=sys.stderr)
            print("server가 떠 있는지 확인하세요: scripts/launch_llama_cpp_14p5b_posttrain_dpo_8622.sh", file=sys.stderr)
            continue

        print(f"bot> {answer}")
        turns.append({"role": "user", "content": user_text})
        turns.append({"role": "assistant", "content": answer})

    if args.save is not None:
        save_transcript(args.save, args.system_prompt, turns)
        print(f"saved: {args.save}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
