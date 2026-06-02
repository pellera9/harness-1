#!/usr/bin/env python3
"""Minimal Hugging Face inference for the released Harness-1 checkpoint.

This script is intentionally small: it verifies that the merged model at
`$HARNESS1_HF_MODEL` can be loaded with the standard Transformers API and used for
plain generation. Full search-agent evaluation uses `evaluate_harness1.py`.
"""

from __future__ import annotations

# Allow direct execution from subdirectories while keeping imports package-relative.
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL = "pat-jj/harness-1"
DEFAULT_BASE_MODEL = "openai/gpt-oss-20b"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--base-model",
        default=DEFAULT_BASE_MODEL,
        help="Base model to use only when --adapter is set.",
    )
    parser.add_argument(
        "--adapter",
        action="store_true",
        help="Treat --model as a PEFT adapter and load it on top of --base-model.",
    )
    parser.add_argument("--prompt", default="Briefly describe Harness-1 in one sentence.")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", default="bfloat16", choices=["auto", "float16", "bfloat16", "float32"])
    return parser.parse_args()


def resolve_dtype(name: str):
    if name == "auto":
        return "auto"
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def main() -> None:
    args = parse_args()
    model_id = args.model
    if args.adapter:
        from peft import PeftModel

        tokenizer = AutoTokenizer.from_pretrained(args.base_model)
        base = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            device_map=args.device_map,
            torch_dtype=resolve_dtype(args.dtype),
        )
        model = PeftModel.from_pretrained(base, model_id)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map=args.device_map,
            torch_dtype=resolve_dtype(args.dtype),
        )
    inputs = tokenizer(args.prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.temperature > 0,
            temperature=args.temperature if args.temperature > 0 else None,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = output_ids[0, inputs["input_ids"].shape[-1] :]
    print(tokenizer.decode(generated, skip_special_tokens=True).strip())


if __name__ == "__main__":
    main()
