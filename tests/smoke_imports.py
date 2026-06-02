#!/usr/bin/env python3
"""Import smoke test for the Harness-1 repository."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


MODULES = [
    "harness.config",
    "harness.tools",
    "harness.ultra_core",
    "datagen.search_dataset",
    "training.generate_sft_data",
    "training.train_sft",
    "training.train_rl",
    "inference.evaluate_harness1",
    "inference.queue_browsecomp_ablation",
    "inference.hf_inference",
    "inference.vllm_local_inference",
]


def main() -> None:
    for name in MODULES:
        importlib.import_module(name)
        print(f"ok {name}")


if __name__ == "__main__":
    main()
