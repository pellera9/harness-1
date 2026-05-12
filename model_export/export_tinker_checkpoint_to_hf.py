#!/usr/bin/env python3
"""Merge a locally downloaded Tinker adapter into a Hugging Face model.

Download the adapter checkpoint outside this script using your private Tinker
checkpoint URI, then pass the downloaded directory with --adapter-path. This
repository intentionally does not embed private Tinker checkpoint paths.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


BASE_MODEL = "openai/gpt-oss-20b"
HF_REPO_ID = "pat-jj/harness-1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-path", required=True, help="Local downloaded Tinker adapter directory")
    parser.add_argument("--output-path", default="model_export/merged_model")
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--repo-id", default=HF_REPO_ID)
    parser.add_argument("--push", action="store_true", help="Upload merged model to Hugging Face Hub")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    adapter_path = Path(args.adapter_path)
    required = [adapter_path / "adapter_model.safetensors", adapter_path / "adapter_config.json"]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit(
            "Missing downloaded adapter files:\n"
            + "\n".join(f"  - {path}" for path in missing)
            + "\nDownload the Tinker sampler checkpoint first and pass its local directory."
        )

    output_path = Path(args.output_path)
    if output_path.exists():
        raise SystemExit(f"Output path already exists: {output_path}")

    merge_script = Path(__file__).with_name("merge_tinker_adapter_to_hf_model.py")
    subprocess.run(
        [
            "python",
            str(merge_script),
            "--hf-model",
            args.base_model,
            "--tinker-adapter-path",
            str(adapter_path),
            "--output-path",
            str(output_path),
        ],
        check=True,
    )

    if args.push:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(args.repo_id, repo_type="model", exist_ok=True)
        api.upload_folder(
            repo_id=args.repo_id,
            repo_type="model",
            folder_path=str(output_path),
            commit_message="Upload HarnesS-1 merged model",
        )

    print("Merged model written to", output_path)


if __name__ == "__main__":
    main()
