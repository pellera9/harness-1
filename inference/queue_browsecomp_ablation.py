#!/usr/bin/env python3
"""Run real BrowseComp+ component ablations for Harness-1.

Each condition evaluates the same fixed query IDs, so the resulting
BrowseComp+ ablation table is paired across queries instead of guessed.
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
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))


CHECKPOINT = os.environ.get("HARNESS1_TINKER_CHECKPOINT")

FULL_ENV: Dict[str, str] = {
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
    "SAVE_FULL_TRAJECTORIES": "0",
    "SAVE_TRAJECTORIES": "1",
    "PYTHONPATH": ".",
}

ABLATIONS: Dict[str, Dict[str, str]] = {
    "full": {},
    "minus_auto_seed": {"V8D_AUTO_POPULATE_FIRST_SEARCH": "0"},
    "minus_importance_tags": {
        "V8D_IMPORTANCE_TAGGING": "0",
        "V8D_SUBTRACTIVE_CURATION": "0",
    },
    "minus_evidence_graph": {"V8D_EVIDENCE_GRAPH": "0"},
    "minus_verify": {"ABLATE_VERIFY_UNAVAILABLE": "1"},
    "minus_sentence_compress": {"V8D_SENTENCE_COMPRESS": "0"},
    "minus_content_dedup": {"V8D_CONTENT_DEDUP": "0"},
    "minus_review_docs": {"ABLATE_REVIEW_DOCS_UNAVAILABLE": "1"},
    "all_harness_mechanisms_disabled": {
        "V8D_SUBTRACTIVE_CURATION": "0",
        "V8D_IMPORTANCE_TAGGING": "0",
        "V8D_AUTO_POPULATE_FIRST_SEARCH": "0",
        "V8D_EVIDENCE_GRAPH": "0",
        "V8D_SENTENCE_COMPRESS": "0",
        "V8D_CONTENT_DEDUP": "0",
        "V8D_VERIFY_TOOL": "0",
        "V8D_TOKEN_BUDGET_MARKER": "0",
        "ABLATE_REVIEW_DOCS_UNAVAILABLE": "1",
    },
}


@dataclass
class Job:
    name: str
    command: List[str]
    output_dir: Path
    result_path: Path
    log_path: Path
    env_extra: Dict[str, str] = field(default_factory=dict)


def load_query_ids_from_file(path: Path, limit: int) -> List[str]:
    query_ids = json.loads(path.read_text(encoding="utf-8"))
    if limit > 0:
        query_ids = query_ids[:limit]
    if not query_ids:
        raise ValueError(f"No query IDs loaded from {path}")
    return [str(qid) for qid in query_ids]


def sample_query_ids(seed: int, limit: int) -> List[str]:
    import random

    from datagen.search_dataset import get_dataset

    if limit < 1:
        raise ValueError("--limit must be positive when sampling query IDs")
    dataset = get_dataset("browsecompplus")
    query_ids = dataset.get_test_query_ids()
    if len(query_ids) < limit:
        raise ValueError(
            f"Requested {limit} BrowseComp+ test queries but only {len(query_ids)} are available."
        )
    rng = random.Random(seed)
    return [str(qid) for qid in rng.sample(query_ids, limit)]


def resolve_query_ids(args: argparse.Namespace, output_root: Path) -> List[str]:
    if args.query_id_file:
        query_ids = load_query_ids_from_file(Path(args.query_id_file), args.limit)
    else:
        query_ids = sample_query_ids(args.seed, args.limit)
    qid_path = output_root / "query_ids.json"
    qid_path.write_text(json.dumps(query_ids, indent=2), encoding="utf-8")
    print(f"[browsecomp-ablation] query IDs written to {qid_path}", flush=True)
    return query_ids


def build_jobs(args: argparse.Namespace, query_ids: Sequence[str]) -> List[Job]:
    jobs: List[Job] = []
    requested = set(args.conditions) if args.conditions else None
    for name, env_delta in ABLATIONS.items():
        if requested and name not in requested:
            continue
        out_dir = Path(args.output_root) / name
        result_path = out_dir / "eval_sft_results.json"
        log_path = out_dir / "run.log"
        command = [
            args.python_bin,
            "inference/evaluate_harness1.py",
            "--dataset",
            "browsecompplus",
            "--split",
            "test",
            "--collection-split",
            "test",
            "--seed",
            str(args.seed),
            "--max-turns",
            str(args.max_turns),
            "--temperature",
            str(args.temperature),
            "--max-tokens",
            str(args.max_tokens),
            "--parallel",
            str(args.parallel_episodes),
            "--checkpoints",
            f"{name}={args.checkpoint}",
            "--output",
            str(result_path),
            "--query-ids",
            *query_ids,
        ]
        env = dict(FULL_ENV)
        env.update(env_delta)
        env["TRAJECTORY_SAVE_PATH"] = str(out_dir / "trajectories")
        jobs.append(
            Job(
                name=name,
                command=command,
                output_dir=out_dir,
                result_path=result_path,
                log_path=log_path,
                env_extra=env,
            )
        )
    return jobs


def run_queue(
    jobs: Sequence[Job],
    workspace: Path,
    max_parallel: int,
    poll_seconds: int,
) -> int:
    pending = list(jobs)
    running: List[tuple[Job, subprocess.Popen, object]] = []
    failures = 0
    while pending or running:
        while pending and len(running) < max_parallel:
            job = pending.pop(0)
            if job.result_path.exists():
                print(f"[browsecomp-ablation] skip existing: {job.name}", flush=True)
                continue
            job.output_dir.mkdir(parents=True, exist_ok=True)
            (job.output_dir / "trajectories").mkdir(parents=True, exist_ok=True)
            env = os.environ.copy()
            env.update(job.env_extra)
            log_f = job.log_path.open("w", encoding="utf-8")
            print(f"[browsecomp-ablation] START {job.name}", flush=True)
            proc = subprocess.Popen(
                job.command,
                cwd=workspace,
                env=env,
                stdout=log_f,
                stderr=subprocess.STDOUT,
            )
            running.append((job, proc, log_f))

        still: List[tuple[Job, subprocess.Popen, object]] = []
        for job, proc, log_f in running:
            rc = proc.poll()
            if rc is None:
                still.append((job, proc, log_f))
                continue
            log_f.close()
            tag = "OK" if rc == 0 else f"FAIL(rc={rc})"
            print(f"[browsecomp-ablation] DONE {job.name} {tag}", flush=True)
            if rc != 0:
                failures += 1
        running = still
        if pending or running:
            time.sleep(poll_seconds)
    return failures


def summarize(output_root: Path) -> None:
    rows = []
    for name in ABLATIONS:
        path = output_root / name / "eval_sft_results.json"
        if not path.exists():
            continue
        obj = json.loads(path.read_text(encoding="utf-8"))
        results = next(iter(obj.values()))
        n = len(results)
        def mean(key: str) -> float:
            return sum(float(r.get(key, 0.0)) for r in results) / max(n, 1)
        rows.append(
            {
                "condition": name,
                "n": n,
                "recall": mean("recall"),
                "trajectory_recall": mean("trajectory_recall"),
                "final_answer_recall": mean("final_answer_recall"),
                "tool_diversity": mean("tool_diversity"),
                "turns": mean("turns"),
                "errors": sum(1 for r in results if r.get("error")),
            }
        )
    if not rows:
        return
    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print("\ncondition,n,recall,trajectory_recall,final_answer_recall,tool_diversity,turns,errors")
    for row in rows:
        print(
            f"{row['condition']},{row['n']},{row['recall']:.4f},"
            f"{row['trajectory_recall']:.4f},{row['final_answer_recall']:.4f},"
            f"{row['tool_diversity']:.4f},{row['turns']:.2f},{row['errors']}"
        )
    print(f"\n[browsecomp-ablation] summary written to {summary_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python-bin", default="./.venv/bin/python")
    parser.add_argument("--checkpoint", default=CHECKPOINT)
    parser.add_argument(
        "--query-id-file",
        default=None,
        help="Optional fixed query-id JSON. Default: sample 100 valid BrowseComp+ test queries.",
    )
    parser.add_argument("--output-root", default="tmp/browsecomp100_component_ablation")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-turns", type=int, default=40)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--parallel-episodes", type=int, default=1)
    parser.add_argument("--max-parallel", type=int, default=1)
    parser.add_argument("--poll-seconds", type=int, default=15)
    parser.add_argument("--conditions", nargs="*", default=None, choices=sorted(ABLATIONS))
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    if not args.summarize_only:
        qids = resolve_query_ids(args, output_root)
        jobs = build_jobs(args, qids)
        print(f"[browsecomp-ablation] queued {len(jobs)} condition(s), {len(qids)} queries each")
        if args.dry_run:
            for job in jobs:
                print(f"[browsecomp-ablation] DRY {job.name}: {' '.join(job.command[:16])} ...")
        else:
            failures = run_queue(jobs, Path.cwd(), args.max_parallel, args.poll_seconds)
            if failures:
                raise SystemExit(failures)
    summarize(output_root)
