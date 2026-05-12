#!/usr/bin/env python3
"""Queue Harness-1 evaluation reruns with V8D flags enabled.

Runs evaluate_harness1.py (in-domain) and evaluate_transfer.py
(transfer) with the full training-matching V8D flag set, using the same query
IDs that produced the current Table 3 numbers.
"""
from __future__ import annotations

# Allow direct execution from subdirectories while keeping imports package-relative.
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


import argparse
import json
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

V8D_ENV: Dict[str, str] = {
    "V8D_SUBTRACTIVE_CURATION": "1",
    "V8D_IMPORTANCE_TAGGING": "1",
    "V8D_AUTO_POPULATE_FIRST_SEARCH": "1",
    "V8D_EVIDENCE_GRAPH": "1",
    "V8D_SENTENCE_COMPRESS": "1",
    "V8D_CHUNK_NEIGHBORS": "0",
    "V8D_CONTENT_DEDUP": "1",
    "V8D_VERIFY_TOOL": "1",
    "V8D_TOKEN_BUDGET_MARKER": "1",
    "V8D_ADAPTIVE_RERANK_INSTRUCTION": "0",
    "SENTENCE_COMPRESS_K": "4",
    "AUTO_POPULATE_TOP_K": "8",
    "SAVE_FULL_TRAJECTORIES": "1",
}

CHECKPOINT = os.environ.get("HARNESS1_TINKER_CHECKPOINT")

IN_DOMAIN_DATASETS = ["browsecompplus", "web", "patents", "sec"]
TRANSFER_DATASETS = ["longsealqa", "seal0qa", "frames", "hotpotqa_subset"]


@dataclass
class Job:
    name: str
    command: List[str]
    output_dir: Path
    done_marker: Path
    env_extra: Dict[str, str] = field(default_factory=dict)


def _load_query_ids(path: Path) -> List[str]:
    return json.loads(path.read_text(encoding="utf-8"))


def _build_indomain_job(
    *,
    python_bin: str,
    dataset: str,
    query_ids: List[str],
    output_root: Path,
    checkpoint: str,
    parallel: int,
    temperature: float,
    max_turns: int,
) -> Job:
    out_dir = output_root / dataset
    results_path = out_dir / "eval_sft_results.json"
    traj_dir = out_dir / "trajectories"

    cmd = [
        python_bin, "inference/evaluate_harness1.py",
        "--dataset", dataset,
        "--split", "test",
        "--collection-split", "test",
        "--seed", "42",
        "--max-turns", str(max_turns),
        "--temperature", str(temperature),
        "--max-tokens", "2048",
        "--parallel", str(parallel),
        "--checkpoints", f"h1_tagged={checkpoint}",
        "--output", str(results_path),
        "--query-ids",
    ]
    cmd.extend(query_ids)

    env_extra = {
        "SAVE_TRAJECTORIES": "1",
        "TRAJECTORY_SAVE_PATH": str(traj_dir),
        "PYTHONPATH": ".",
    }
    env_extra.update(V8D_ENV)

    return Job(
        name=f"indomain_{dataset}",
        command=cmd,
        output_dir=out_dir,
        done_marker=results_path,
        env_extra=env_extra,
    )


def _build_transfer_job(
    *,
    python_bin: str,
    dataset: str,
    query_ids: List[str],
    output_root: Path,
    checkpoint: str,
    parallel: int,
    temperature: float,
    max_turns: int,
) -> Job:
    out_dir = output_root / dataset
    results_path = out_dir / "eval_sft_transfer_results.json"
    traj_dir = out_dir / "trajectories"

    cmd = [
        python_bin, "inference/evaluate_transfer.py",
        "--dataset", dataset,
        "--split", "test",
        "--seed", "42",
        "--max-turns", str(max_turns),
        "--temperature", str(temperature),
        "--max-tokens", "2048",
        "--parallel", str(parallel),
        "--checkpoints", f"h1_tagged={checkpoint}",
        "--output", str(results_path),
        "--query-ids",
    ]
    cmd.extend(query_ids)

    env_extra = {
        "SAVE_TRAJECTORIES": "1",
        "TRAJECTORY_SAVE_PATH": str(traj_dir),
        "PYTHONPATH": ".",
    }
    env_extra.update(V8D_ENV)

    return Job(
        name=f"transfer_{dataset}",
        command=cmd,
        output_dir=out_dir,
        done_marker=results_path,
        env_extra=env_extra,
    )


def _run_queue(
    jobs: Sequence[Job],
    workspace: Path,
    max_parallel: int,
) -> int:
    pending = list(jobs)
    running: List[tuple[Job, subprocess.Popen]] = []
    failures = 0

    while pending or running:
        while pending and len(running) < max_parallel:
            job = pending.pop(0)
            if job.done_marker.exists():
                print(f"[h1-tagged] skip existing: {job.name}", flush=True)
                continue
            job.output_dir.mkdir(parents=True, exist_ok=True)
            traj_dir = job.env_extra.get("TRAJECTORY_SAVE_PATH")
            if traj_dir:
                Path(traj_dir).mkdir(parents=True, exist_ok=True)

            env = os.environ.copy()
            env.update(job.env_extra)

            print(f"[h1-tagged] START {job.name}  ({len(job.command)} args)", flush=True)
            proc = subprocess.Popen(job.command, cwd=workspace, env=env)
            running.append((job, proc))

        if not running:
            continue

        still: List[tuple[Job, subprocess.Popen]] = []
        for job, proc in running:
            rc = proc.poll()
            if rc is None:
                still.append((job, proc))
                continue
            tag = "OK" if rc == 0 else f"FAIL(rc={rc})"
            print(f"[h1-tagged] DONE {job.name}  {tag}", flush=True)
            if rc != 0:
                failures += 1
        running = still
        if pending or running:
            time.sleep(15)

    return failures


def main() -> None:
    if not CHECKPOINT:
        raise SystemExit("Set HARNESS1_TINKER_CHECKPOINT or pass a checkpoint explicitly before running this queue.")

    parser = argparse.ArgumentParser(
        description="Rerun Harness-1 Table 3 evals with V8D flags enabled."
    )
    parser.add_argument("--python-bin", default="./.venv/bin/python")
    parser.add_argument("--checkpoint", default=CHECKPOINT)
    parser.add_argument(
        "--output-root", default="tmp/table3_h1_tagged_rerun",
    )
    parser.add_argument(
        "--query-id-dir", default="tmp/table3_h1_tagged_rerun/query_ids",
        help="Directory containing per-dataset query ID JSON files.",
    )
    parser.add_argument("--max-parallel", type=int, default=2)
    parser.add_argument("--parallel-episodes", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-turns", type=int, default=40)
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Subset of datasets to run. Default: all 8.",
    )
    args = parser.parse_args()

    workspace = Path.cwd()
    output_root = Path(args.output_root)
    qid_dir = Path(args.query_id_dir)

    requested = set(args.datasets) if args.datasets else None

    jobs: List[Job] = []

    for ds in IN_DOMAIN_DATASETS:
        if requested and ds not in requested:
            continue
        qid_file = qid_dir / f"{ds}.json"
        if not qid_file.exists():
            print(f"[h1-tagged] WARN: no query IDs for {ds}, skipping", flush=True)
            continue
        qids = _load_query_ids(qid_file)
        jobs.append(_build_indomain_job(
            python_bin=args.python_bin,
            dataset=ds,
            query_ids=qids,
            output_root=output_root,
            checkpoint=args.checkpoint,
            parallel=args.parallel_episodes,
            temperature=args.temperature,
            max_turns=args.max_turns,
        ))

    for ds in TRANSFER_DATASETS:
        if requested and ds not in requested:
            continue
        qid_file = qid_dir / f"{ds}.json"
        if not qid_file.exists():
            print(f"[h1-tagged] WARN: no query IDs for {ds}, skipping", flush=True)
            continue
        qids = _load_query_ids(qid_file)
        jobs.append(_build_transfer_job(
            python_bin=args.python_bin,
            dataset=ds,
            query_ids=qids,
            output_root=output_root,
            checkpoint=args.checkpoint,
            parallel=args.parallel_episodes,
            temperature=args.temperature,
            max_turns=args.max_turns,
        ))

    print(f"[h1-tagged] Queued {len(jobs)} jobs (max parallel: {args.max_parallel})")
    print(f"[h1-tagged] V8D flags: {V8D_ENV}")
    print(f"[h1-tagged] Output root: {output_root}", flush=True)

    failures = _run_queue(jobs, workspace, args.max_parallel)
    if failures:
        print(f"[h1-tagged] {failures} job(s) failed", flush=True)
        raise SystemExit(1)
    print("[h1-tagged] All jobs completed successfully", flush=True)


if __name__ == "__main__":
    main()
