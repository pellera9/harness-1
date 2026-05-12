#!/usr/bin/env python3
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
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence


@dataclass
class Job:
    name: str
    dataset: str
    run_tag: str
    source_query_ids: Path
    command: List[str]
    done_path: Path
    trajectory_dir: Path


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _wait_for_pids(pids: Sequence[int], poll_secs: int = 20) -> None:
    watched = [p for p in pids if p > 0]
    if not watched:
        return
    print(f"[queue-h1-evalsft-rerun] waiting for existing jobs: {watched}", flush=True)
    while True:
        alive = [p for p in watched if _pid_alive(p)]
        if not alive:
            print("[queue-h1-evalsft-rerun] watched jobs finished", flush=True)
            return
        print(f"[queue-h1-evalsft-rerun] still running: {alive}", flush=True)
        time.sleep(poll_secs)


def _load_query_ids(path: Path) -> List[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected list-form JSON at {path}")
    query_ids = [str(qid) for qid in payload]
    if not query_ids:
        raise ValueError(f"No query IDs found at {path}")
    return query_ids


def _dataset_order(dataset: str) -> int:
    order = {
        "sec": 0,
        "browsecompplus": 1,
        "web": 2,
        "patents": 3,
    }
    return order.get(dataset, 100)


def _build_job(
    *,
    python_bin: str,
    checkpoint: str,
    query_ids_path: Path,
    output_root: Path,
    parallel_episodes: int,
) -> Job:
    parts = list(query_ids_path.parts)
    try:
        harness_idx = parts.index("harness1")
    except ValueError as exc:
        raise ValueError(f"Unexpected query_ids path (missing harness1): {query_ids_path}") from exc
    if harness_idx == 0 or harness_idx + 4 >= len(parts):
        raise ValueError(f"Unexpected query_ids path layout: {query_ids_path}")

    source_label = parts[harness_idx - 1]
    dataset = parts[harness_idx + 2]
    run_tag = parts[harness_idx + 3]
    query_ids = _load_query_ids(query_ids_path)

    run_root = output_root / source_label / "harness1" / "in_domain_3x" / dataset / run_tag
    done_path = run_root / "eval_sft_harness1_results.json"
    trajectory_dir = run_root / "trajectories"
    command: List[str] = [
        python_bin,
        "inference/evaluate_harness1.py",
        "--dataset",
        dataset,
        "--split",
        "all",
        "--seed",
        "0",
        "--max-turns",
        "40",
        "--temperature",
        "0.6",
        "--max-tokens",
        "2048",
        "--parallel",
        str(parallel_episodes),
        "--checkpoints",
        f"harness1={checkpoint}",
        "--output",
        str(done_path),
        "--query-ids",
    ]
    command.extend(query_ids)

    return Job(
        name=f"{source_label}_{dataset}_{run_tag}",
        dataset=dataset,
        run_tag=run_tag,
        source_query_ids=query_ids_path,
        command=command,
        done_path=done_path,
        trajectory_dir=trajectory_dir,
    )


def _discover_query_id_files(source_roots: Sequence[Path]) -> List[Path]:
    paths: List[Path] = []
    for root in source_roots:
        paths.extend(root.glob("harness1/in_domain_3x/*/*/*/query_ids.json"))
    unique_paths = sorted({path.resolve() for path in paths})
    return unique_paths


def _run_queue(
    *,
    jobs: Sequence[Job],
    workspace_root: Path,
    max_parallel_jobs: int,
) -> int:
    pending = list(jobs)
    running: List[tuple[Job, subprocess.Popen]] = []
    failures = 0

    while pending or running:
        while pending and len(running) < max_parallel_jobs:
            job = pending.pop(0)
            if job.done_path.exists():
                print(
                    f"[queue-h1-evalsft-rerun] skip existing {job.name}: {job.done_path}",
                    flush=True,
                )
                continue

            job.done_path.parent.mkdir(parents=True, exist_ok=True)
            job.trajectory_dir.mkdir(parents=True, exist_ok=True)
            env = os.environ.copy()
            env["PYTHONPATH"] = "."
            env["SAVE_TRAJECTORIES"] = "1"
            env["TRAJECTORY_SAVE_PATH"] = str(job.trajectory_dir)
            print(
                f"[queue-h1-evalsft-rerun] start {job.name} "
                f"qids={job.source_query_ids}",
                flush=True,
            )
            proc = subprocess.Popen(job.command, cwd=workspace_root, env=env)
            running.append((job, proc))

        if not running:
            continue

        still_running: List[tuple[Job, subprocess.Popen]] = []
        for job, proc in running:
            rc = proc.poll()
            if rc is None:
                still_running.append((job, proc))
                continue
            if rc != 0:
                failures += 1
            print(
                f"[queue-h1-evalsft-rerun] done {job.name} exit_code={rc}",
                flush=True,
            )
        running = still_running
        if pending or running:
            time.sleep(10)

    return failures


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Rerun Harness-1 in-domain jobs with evaluate_harness1.py using "
            "the exact query_ids from prior compare_harnesses outputs."
        )
    )
    parser.add_argument("--python-bin", default="./.venv/bin/python")
    parser.add_argument(
        "--source-roots",
        default=(
            "tmp/table345_3x_20260430_additional_resume_from402,"
            "tmp/table345_3x_20260430_sec_balanced"
        ),
        help=(
            "Comma-separated roots containing old compare_harnesses outputs "
            "(query_ids.json files are discovered recursively)."
        ),
    )
    parser.add_argument(
        "--output-root",
        default="tmp/table345_3x_20260430_harness1_evalsft_reruns",
        help="Output root for rerun results and trajectories.",
    )
    parser.add_argument(
        "--checkpoint",
        default=os.environ.get("HARNESS1_TINKER_CHECKPOINT"),
        help="Harness-1 checkpoint path. Defaults to HARNESS1_TINKER_CHECKPOINT.",
    )
    parser.add_argument("--max-parallel-jobs", type=int, default=2)
    parser.add_argument(
        "--parallel-episodes",
        type=int,
        default=4,
        help="Episode-level parallelism inside each eval process.",
    )
    parser.add_argument("--wait-pids", default="")
    args = parser.parse_args()

    wait_pids = [int(p.strip()) for p in args.wait_pids.split(",") if p.strip()]
    _wait_for_pids(wait_pids)

    workspace_root = Path.cwd()
    output_root = Path(args.output_root)
    source_roots = [Path(s.strip()) for s in args.source_roots.split(",") if s.strip()]
    query_id_files = _discover_query_id_files(source_roots)
    if not query_id_files:
        raise FileNotFoundError(
            f"No harness1 query_ids.json files found under: {source_roots}"
        )

    jobs = [
        _build_job(
            python_bin=args.python_bin,
            checkpoint=args.checkpoint,
            query_ids_path=path,
            output_root=output_root,
            parallel_episodes=args.parallel_episodes,
        )
        for path in query_id_files
    ]
    jobs.sort(key=lambda job: (_dataset_order(job.dataset), job.name))
    print(f"[queue-h1-evalsft-rerun] total jobs queued: {len(jobs)}", flush=True)

    failures = _run_queue(
        jobs=jobs,
        workspace_root=workspace_root,
        max_parallel_jobs=args.max_parallel_jobs,
    )
    if failures:
        raise SystemExit(failures)
    print("[queue-h1-evalsft-rerun] all queued jobs finished", flush=True)


if __name__ == "__main__":
    main()
