"""Evaluate Harness-1 checkpoints with the current RL environment.

Runs full multi-turn search episodes against held-out queries using the
current `train_rl.SlidingWindowSearchEnv`, so results are directly
comparable to rollouts produced by the RL training stack.
"""

from __future__ import annotations

# Allow direct execution from subdirectories while keeping imports package-relative.
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


import argparse
import asyncio
import json
import os
import random
import time
from pathlib import Path
from typing import Dict, List

import structlog
import tiktoken
import tinker

from harness.config import get_config
from datagen.search_dataset import SearchDataset, get_dataset
from harness.tools import (
    GrepCorpusTool,
    PruneChunksTool,
    ReadDocumentTool,
    SearchCorpusTool,
    ToolSet,
    UserTextTool,
)
from training.train_rl import SlidingWindowSearchEnv, SEARCH_DISPLAY_LIMIT, MAX_TURNS
from tinker_cookbook.completers import TinkerTokenCompleter

logger = structlog.get_logger("evaluate_harness1")

SAVE_FULL_TRAJECTORIES = os.environ.get("SAVE_FULL_TRAJECTORIES", "0") == "1"


def save_full_trajectory(env: "SlidingWindowSearchEnv") -> None:
    """Persist complete conversation (reasoning, tool calls, tool returns) per query."""
    traj_root = os.environ.get("TRAJECTORY_SAVE_PATH") or os.environ.get("LOG_PATH", "./tmp/rl_ultra_v3")
    full_dir = os.path.join(traj_root, "full")
    os.makedirs(full_dir, exist_ok=True)

    turns = []
    for i, (action, obs) in enumerate(zip(env._all_actions, env._all_observations)):
        turn_record = {"turn": i}
        if action.reasoning:
            turn_record["reasoning"] = action.reasoning

        tool_calls = []
        for tool, params in zip(action.tools, action.params):
            name = "user_text" if isinstance(tool, UserTextTool) else tool.tool_schema.name
            tool_calls.append({"tool": name, "params": params})
        turn_record["tool_calls"] = tool_calls

        tool_returns = []
        for j, obs_text in enumerate(obs.observations):
            tr = {"text": obs_text}
            if j < len(obs.tool_metadata) and obs.tool_metadata[j] is not None:
                try:
                    tr["metadata"] = obs.tool_metadata[j].model_dump()
                except Exception:
                    tr["metadata"] = str(obs.tool_metadata[j])
            tool_returns.append(tr)
        turn_record["tool_returns"] = tool_returns
        turns.append(turn_record)

    evidence_graph = None
    if env.wm.evidence_graph is not None:
        eg = env.wm.evidence_graph
        evidence_graph = {
            "entity_to_docs": {e: sorted(docs) for e, docs in eg.entity_to_docs.items()},
            "doc_to_entities": {d: sorted(ents) for d, ents in eg.doc_to_entities.items()},
        }

    record = {
        "query_id": env.query_id,
        "query_text": env.wm.query,
        "dataset": env.dataset.name,
        "system_prompt": env.system_prompt,
        "turns": turns,
        "curated_ids": env.wm.curated_ids,
        "curated_importance": dict(env.wm.curated_importance),
        "evidence_graph": evidence_graph,
        "reward": env._terminal_reward,
        "metrics": {
            k: v for k, v in env._terminal_metrics.items()
            if isinstance(v, (int, float, str, bool))
        },
    }

    qid_safe = str(env.query_id).replace("/", "_")
    with open(os.path.join(full_dir, f"{qid_safe}.json"), "w") as f:
        json.dump(record, f, indent=2, default=str)


async def run_single_episode(
    env: SlidingWindowSearchEnv,
    sampling_client: tinker.SamplingClient,
    temperature: float,
    max_tokens: int,
) -> Dict:
    """Run one full episode and return metrics."""
    policy = TinkerTokenCompleter(
        sampling_client=sampling_client,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    ob, stop_condition = await env.initial_observation()
    turns = 0
    start = time.time()

    while True:
        ac_with_logprobs = await policy(ob, stop_condition)
        step_result = await env.step(ac_with_logprobs.tokens)
        turns += 1
        if step_result.episode_done:
            break
        ob = step_result.next_observation
        stop_condition = step_result.next_stop_condition

    elapsed = time.time() - start

    result = {
        "reward": env._terminal_reward,
        "turns": turns,
        "n_curated": len(env.wm.curated_ids),
        "n_pool": len(env.wm.pool_ids),
        "elapsed_s": round(elapsed, 1),
        "error": env._terminal_metrics.get("no_error", 1.0) == 0.0,
        "tool_types_used": list(env._tool_types_used),
        "total_curate_calls": env._total_curate_calls,
    }
    result.update(env._terminal_metrics)
    return result


async def _eval_single_query(
    qid: str,
    dataset: SearchDataset,
    toolset: ToolSet,
    search_tool: SearchCorpusTool,
    text_token_counter,
    sampling_client,
    temperature: float,
    max_tokens: int,
    max_turns: int,
) -> Dict:
    _, query_text = dataset.get_query_by_id(qid)
    env = SlidingWindowSearchEnv(
        toolset=toolset,
        search_tool=search_tool,
        query_id=qid,
        query_text=query_text,
        dataset=dataset,
        text_token_counter=text_token_counter,
        max_turns=max_turns,
    )
    try:
        result = await run_single_episode(
            env=env,
            sampling_client=sampling_client,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        result["query_id"] = qid
        result["query"] = query_text[:80]
        if SAVE_FULL_TRAJECTORIES:
            try:
                save_full_trajectory(env)
            except Exception as exc:
                logger.warning("save_full_trajectory_error", qid=qid, error=str(exc)[:200])
        logger.info(
            "episode_result",
            qid=qid,
            recall=round(result.get("recall", 0), 3),
            pool_recall=round(result.get("trajectory_recall", 0), 3),
            precision=round(result.get("precision", 0), 3),
            reward=round(result.get("reward", 0), 3),
            curated=result["n_curated"],
            pool=result["n_pool"],
            turns=result["turns"],
            error=result["error"],
            time=result["elapsed_s"],
        )
        return result
    except Exception as e:
        logger.error("episode_failed", qid=qid, error=str(e)[:300])
        return {
            "query_id": qid,
            "query": query_text[:80],
            "error": True,
            "reward": 0,
            "recall": 0,
            "trajectory_recall": 0,
            "precision": 0,
            "n_curated": 0,
            "n_pool": 0,
            "turns": 0,
        }


async def eval_checkpoint(
    sampler_path: str,
    query_ids: List[str],
    dataset: SearchDataset,
    toolset: ToolSet,
    search_tool: SearchCorpusTool,
    text_token_counter,
    temperature: float,
    max_tokens: int,
    max_turns: int,
    parallel: int = 10,
) -> List[Dict]:
    """Evaluate a checkpoint on a fixed query list with parallel episodes."""
    sc = tinker.ServiceClient()
    sampling_client = sc.create_sampling_client(model_path=sampler_path)
    logger.info("loaded_checkpoint", path=sampler_path, parallel=parallel)

    sem = asyncio.Semaphore(parallel)

    async def bounded(qid: str) -> Dict:
        async with sem:
            return await _eval_single_query(
                qid, dataset, toolset, search_tool, text_token_counter,
                sampling_client, temperature, max_tokens, max_turns,
            )

    tasks = [bounded(qid) for qid in query_ids]
    results = await asyncio.gather(*tasks)
    return list(results)


def print_results_table(name: str, results: List[Dict]) -> None:
    """Pretty-print evaluation results."""
    print(f"\n{'=' * 80}")
    print(f"  {name}")
    print(f"{'=' * 80}")

    recalls = [r.get("recall", 0) for r in results]
    pool_recalls = [r.get("trajectory_recall", 0) for r in results]
    precisions = [r.get("precision", 0) for r in results]
    rewards = [r.get("reward", 0) for r in results]
    curated = [r.get("n_curated", 0) for r in results]
    pools = [r.get("n_pool", 0) for r in results]
    turns_list = [r.get("turns", 0) for r in results]
    errors = [r.get("error", False) for r in results]

    n = len(results)
    n_errors = sum(1 for e in errors if e)
    n_curated_gt0 = sum(1 for c in curated if c > 0)
    n_recall_gt0 = sum(1 for r in recalls if r > 0)

    print(f"\n  {'Metric':<25} {'Mean':>8} {'Min':>8} {'Max':>8} {'> 0':>8}")
    print(f"  {'-' * 25} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 8}")

    def row(label: str, vals: List[float]) -> None:
        gt0 = sum(1 for v in vals if v > 0)
        mn = min(vals) if vals else 0
        mx = max(vals) if vals else 0
        avg = sum(vals) / len(vals) if vals else 0
        print(f"  {label:<25} {avg:>8.4f} {mn:>8.4f} {mx:>8.4f} {gt0:>5}/{n}")

    row("Curated Recall", recalls)
    row("Pool Recall", pool_recalls)
    row("Precision", precisions)
    row("Reward", rewards)
    row("Curated Docs", [float(c) for c in curated])
    row("Pool Docs", [float(p) for p in pools])
    row("Turns", [float(t) for t in turns_list])

    print(f"\n  Errors: {n_errors}/{n}  |  Used curate: {n_curated_gt0}/{n}  |  Recall>0: {n_recall_gt0}/{n}")
    print(f"{'=' * 80}\n")

    print(f"  {'QID':<12} {'Recall':>7} {'PoolRec':>8} {'Prec':>6} {'Reward':>7} {'Cur':>4} {'Pool':>5} {'T':>3} {'Err':>4}")
    print(f"  {'-' * 12} {'-' * 7} {'-' * 8} {'-' * 6} {'-' * 7} {'-' * 4} {'-' * 5} {'-' * 3} {'-' * 4}")
    for r in sorted(results, key=lambda x: -x.get("recall", 0)):
        err = "ERR" if r.get("error") else ""
        print(
            f"  {str(r.get('query_id', '')):<12} "
            f"{r.get('recall', 0):>7.3f} "
            f"{r.get('trajectory_recall', 0):>8.3f} "
            f"{r.get('precision', 0):>6.3f} "
            f"{r.get('reward', 0):>7.3f} "
            f"{r.get('n_curated', 0):>4} "
            f"{r.get('n_pool', 0):>5} "
            f"{r.get('turns', 0):>3} "
            f"{err:>4}"
        )


async def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Harness-1 checkpoints with current RL env")
    parser.add_argument("--dataset", default="browsecompplus", help="Dataset name")
    parser.add_argument(
        "--split",
        default="test",
        choices=["all", "test", "train", "rl"],
        help="Which query split to eval on",
    )
    parser.add_argument(
        "--collection-split",
        default="train",
        choices=["test", "train", "rl"],
        help="Which retrieval corpus split the tools should use",
    )
    parser.add_argument("--n-queries", type=int, default=20, help="Number of queries to evaluate")
    parser.add_argument("--seed", type=int, default=42, help="Seed for query sampling")
    parser.add_argument(
        "--query-ids",
        nargs="*",
        default=None,
        help="Explicit query IDs to evaluate. If provided, bypasses --n-queries/--seed sampling.",
    )
    parser.add_argument("--max-turns", type=int, default=MAX_TURNS, help="Max turns per episode")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=2048, help="Max generation tokens")
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        required=True,
        help="Checkpoint specs as name=path pairs",
    )
    parser.add_argument("--output", type=str, default=None, help="Save results JSON to this path")
    parser.add_argument("--parallel", type=int, default=10, help="Max concurrent episodes")
    args = parser.parse_args()

    config = get_config()
    tiktoken_enc = tiktoken.get_encoding("o200k_harmony")
    text_token_counter = lambda text: len(tiktoken_enc.encode(text))

    dataset = get_dataset(args.dataset)
    collection_names = dataset.get_chroma_collections(split=args.collection_split)
    chroma_client = config.get_chroma_client()
    openai_client = config.get_openai_client()

    try:
        from harness.rerank import BasetenReranker
        reranker = BasetenReranker(token_counter=text_token_counter, max_tokens=4096)
    except Exception:
        reranker = None

    search_tool = SearchCorpusTool(
        chroma_client=chroma_client,
        openai_client=openai_client,
        chroma_collection_name=collection_names,
        reranker=reranker,
        snippet_max_chars=2048,
        display_limit=SEARCH_DISPLAY_LIMIT,
    )
    toolset = ToolSet(name=f"{args.dataset}_toolset")
    toolset.add_tool(search_tool)
    toolset.add_tool(GrepCorpusTool(
        chroma_client=chroma_client,
        chroma_collection_name=collection_names,
        token_counter=text_token_counter,
    ))
    toolset.add_tool(ReadDocumentTool(
        chroma_client=chroma_client,
        chroma_collection_name=collection_names,
        reranker=reranker,
        token_counter=text_token_counter,
        max_tokens=4096,
    ))
    toolset.add_tool(PruneChunksTool())

    if args.split == "all":
        all_qids = dataset.get_all_query_ids()
    elif args.split == "test":
        all_qids = dataset.get_test_query_ids()
    elif args.split == "rl":
        all_qids = dataset.get_rl_query_ids()
    else:
        all_qids = dataset.get_all_query_ids(split="train")

    if args.query_ids:
        known_qids = set(all_qids)
        query_ids = [qid for qid in args.query_ids if qid in known_qids]
        missing_qids = [qid for qid in args.query_ids if qid not in known_qids]
        if missing_qids:
            logger.warning("query_ids_not_found", missing=missing_qids[:10], n_missing=len(missing_qids))
        if not query_ids:
            raise ValueError("No valid query IDs remained after filtering against dataset split")
        logger.info(
            "using_explicit_query_ids",
            n=len(query_ids),
            split=args.split,
            collection_split=args.collection_split,
            dataset=args.dataset,
            query_ids=query_ids,
        )
    else:
        rng = random.Random(args.seed)
        query_ids = rng.sample(all_qids, min(args.n_queries, len(all_qids)))
        logger.info(
            "sampled_queries",
            n=len(query_ids),
            split=args.split,
            collection_split=args.collection_split,
            dataset=args.dataset,
        )

    checkpoints: Dict[str, str] = {}
    for spec in args.checkpoints:
        name, path = spec.split("=", 1)
        checkpoints[name] = path

    all_results: Dict[str, List[Dict]] = {}
    for ckpt_name, ckpt_path in checkpoints.items():
        logger.info("evaluating_checkpoint", name=ckpt_name, path=ckpt_path)
        results = await eval_checkpoint(
            sampler_path=ckpt_path,
            query_ids=query_ids,
            dataset=dataset,
            toolset=toolset,
            search_tool=search_tool,
            text_token_counter=text_token_counter,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            max_turns=args.max_turns,
            parallel=args.parallel,
        )
        all_results[ckpt_name] = results
        print_results_table(ckpt_name, results)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        serializable = {}
        for name, results in all_results.items():
            serializable[name] = []
            for r in results:
                sr = {k: v for k, v in r.items() if isinstance(v, (int, float, str, bool, list))}
                serializable[name].append(sr)
        with open(output_path, "w") as f:
            json.dump(serializable, f, indent=2)
        logger.info("results_saved", path=str(output_path))


if __name__ == "__main__":
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    asyncio.run(main())
