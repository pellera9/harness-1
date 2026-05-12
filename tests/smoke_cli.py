#!/usr/bin/env python3
"""Cheap CLI smoke checks that do not call external model APIs."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


QUERY_IDS = Path("tmp/smoke_query_ids.json")

COMMANDS = [
    [sys.executable, "training/generate_sft_data.py", "--help"],
    [sys.executable, "training/train_sft.py", "--help"],
    [sys.executable, "inference/evaluate_harness1.py", "--help"],
    [
        sys.executable,
        "inference/queue_browsecomp_ablation.py",
        "--dry-run",
        "--query-id-file",
        str(QUERY_IDS),
        "--limit",
        "1",
        "--conditions",
        "full",
    ],
    [sys.executable, "inference/hf_inference.py", "--help"],
    [sys.executable, "inference/vllm_local_inference.py", "--help"],
]


def main() -> None:
    QUERY_IDS.parent.mkdir(parents=True, exist_ok=True)
    QUERY_IDS.write_text('["smoke_query"]\n', encoding="utf-8")
    for cmd in COMMANDS:
        print("+", " ".join(cmd))
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print("ok")


if __name__ == "__main__":
    main()
