#!/usr/bin/env python3
"""Serve or query HarnesS-1 with a local vLLM OpenAI-compatible endpoint."""

from __future__ import annotations

# Allow direct execution from subdirectories while keeping imports package-relative.
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


import argparse
import json
import subprocess
import sys
import urllib.request


DEFAULT_MODEL = "harness-1"


def serve(args: argparse.Namespace) -> None:
    cmd = [
        "vllm",
        "serve",
        args.model,
        "--served-model-name",
        args.served_model_name,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--max-model-len",
        str(args.max_model_len),
    ]
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def query(args: argparse.Namespace) -> None:
    payload = {
        "model": args.served_model_name,
        "messages": [{"role": "user", "content": args.prompt}],
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "stream": False,
    }
    req = urllib.request.Request(
        f"{args.url.rstrip('/')}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=args.timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    print(data["choices"][0]["message"]["content"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    s = sub.add_parser("serve", help="Start a vLLM server")
    s.add_argument("--model", default=DEFAULT_MODEL)
    s.add_argument("--served-model-name", default="harness-1")
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--tensor-parallel-size", type=int, default=1)
    s.add_argument("--max-model-len", type=int, default=32768)

    q = sub.add_parser("query", help="Query an already-running server")
    q.add_argument("--url", default="http://localhost:8000")
    q.add_argument("--served-model-name", default="harness-1")
    q.add_argument("--prompt", default="What is HarnesS-1?")
    q.add_argument("--temperature", type=float, default=0.0)
    q.add_argument("--max-tokens", type=int, default=128)
    q.add_argument("--timeout", type=int, default=120)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "serve":
        serve(args)
    elif args.command == "query":
        query(args)
    else:
        sys.exit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
