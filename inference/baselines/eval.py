from __future__ import annotations

# Allow direct execution from subdirectories while keeping imports package-relative.
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


import argparse
import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, List

import structlog
import tiktoken

from harness.agent import Agent
from harness.config import get_config
from datagen.search_dataset import SearchDataset, get_dataset, DATASET_REGISTRY
from harness.generate_search_sft import build_agent_factory, create_inference_model_factory
from harness.prompts import get_retrieval_subagent_prompt
from harness.rerank import BasetenReranker, ContextualReranker, Reranker
from harness.tasks import SearchTaskEvaluationOutput, SearchTaskOutput
from harness.tools import ToolSet
from harness.trajectory import Observation

logger = structlog.get_logger("search_agent.eval")
_RETRY_ENABLED_TRANSFER_DATASETS = {
    "seal0qa",
    "longsealqa",
    "frames",
    "hotpotqa_subset",
}


class EvaluationRunner:
    """Sample a subset of queries and evaluate the retrieval agent."""

    def __init__(
        self,
        *,
        dataset: SearchDataset,
        agent_factory: Callable[[], Agent],
        sample_size: int,
        seed: int,
        num_workers: int,
        output_dir: Path,
        max_query_retries: int = 0,
    ) -> None:
        if sample_size < 1:
            raise ValueError("sample_size must be >= 1")
        if num_workers < 1:
            raise ValueError("num_workers must be >= 1")
        if max_query_retries < 0:
            raise ValueError("max_query_retries must be >= 0")

        self.dataset = dataset
        self.agent_factory = agent_factory
        self.sample_size = sample_size
        self.seed = seed
        self.num_workers = num_workers
        self.output_dir = output_dir
        self.search_outputs_dir = output_dir / "search_task_outputs"
        self.max_query_retries = max_query_retries
        self.logger = logger.bind(component="EvaluationRunner")

    def run(self) -> dict:
        """Execute the evaluation and persist aggregated metrics."""

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.search_outputs_dir.mkdir(parents=True, exist_ok=True)
        query_ids = self._select_query_ids()
        self.logger.info("evaluation_started", total_queries=len(query_ids))

        results: List[SearchTaskEvaluationOutput] = []
        if self.num_workers == 1:
            for query_id in query_ids:
                results.append(self._evaluate_single(query_id))
        else:
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                future_map = {
                    executor.submit(self._evaluate_single, query_id): query_id
                    for query_id in query_ids
                }
                for future in as_completed(future_map):
                    query_id = future_map[future]
                    try:
                        result = future.result()
                    except Exception as exc:  # pragma: no cover - defensive
                        self.logger.exception(
                            "evaluation_thread_exception",
                            query_id=query_id,
                            error=str(exc),
                        )
                        result = SearchTaskEvaluationOutput(
                            query_id=query_id,
                            recall=None,
                            precision=None,
                            f1=None,
                            trajectory_recall=None,
                            retrieved_document_ids=[],
                            error=str(exc),
                        )
                    results.append(result)

        summary = self._summarize(results)
        self._write_outputs(results, summary)
        self.logger.info(
            "evaluation_finished",
            succeeded=summary["num_succeeded"],
            failed=summary["num_failed"],
            mean_recall=summary["mean_recall"],
            mean_trajectory_recall=summary["mean_trajectory_recall"],
            mean_final_answer_recall=summary["mean_final_answer_recall"],
            mean_prune_accuracy=summary["mean_prune_accuracy"],
            mean_precision=summary["mean_precision"],
            mean_f1=summary["mean_f1"],
            mean_rerank_recall=summary["mean_rerank_recall"],
            mean_rerank_dropped_relevant_count=summary[
                "mean_rerank_dropped_relevant_count"
            ],
        )
        return summary

    def _select_query_ids(self) -> List[str]:
        available_query_ids = self.dataset.get_all_query_ids(split="test")
        if len(available_query_ids) < self.sample_size:
            raise ValueError(
                f"Requested {self.sample_size} queries but dataset only has "
                f"{len(available_query_ids)} entries."
            )
        rng = random.Random(self.seed)
        rng.shuffle(available_query_ids)
        selected = available_query_ids[: self.sample_size]
        self.logger.debug("selected_queries", count=len(selected))
        if self.max_query_retries <= 0:
            return selected

        to_run: List[str] = []
        skipped = 0
        for query_id in selected:
            output_path = self.search_outputs_dir / f"{query_id}.json"
            if output_path.exists():
                try:
                    with output_path.open("r", encoding="utf-8") as fp:
                        data = json.load(fp)
                    output_chunk_ids = data.get("output_chunk_ids", [])
                    if output_chunk_ids:
                        skipped += 1
                        continue
                except (json.JSONDecodeError, OSError):
                    pass
            to_run.append(query_id)

        self.logger.info(
            "query_selection_complete",
            total_selected=len(selected),
            skipped_succeeded=skipped,
            to_run=len(to_run),
        )
        return to_run

    def _evaluate_single(self, query_id: str) -> SearchTaskEvaluationOutput:
        max_attempts = self.max_query_retries + 1
        result: SearchTaskEvaluationOutput | None = None
        for attempt in range(1, max_attempts + 1):
            result = self._evaluate_single_once(query_id)
            if result.succeeded():
                if attempt > 1:
                    self.logger.info(
                        "query_retry_succeeded",
                        query_id=query_id,
                        attempt=attempt,
                        max_attempts=max_attempts,
                    )
                return result
            if attempt < max_attempts:
                self.logger.warning(
                    "query_retrying",
                    query_id=query_id,
                    attempt=attempt,
                    next_attempt=attempt + 1,
                    max_attempts=max_attempts,
                    error=result.error,
                )
        if result is None:
            return SearchTaskEvaluationOutput(
                query_id=query_id,
                error="query evaluation failed before first attempt",
            )
        return result

    def _evaluate_single_once(self, query_id: str) -> SearchTaskEvaluationOutput:
        agent = None
        try:
            agent = self.agent_factory()
            _, query_text = self.dataset.get_query_by_id(query_id)
            initial_observation = Observation(
                observations=[get_retrieval_subagent_prompt(query_text)],
                sources=["user"],
                tool_metadata=[None],
            )
            trajectory = agent(initial_observation=initial_observation)
            output = SearchTaskOutput(
                trajectory=trajectory,
                query_id=query_id,
                dataset_name=self.dataset.name,
            )
            self._write_search_task_output(output)

            eval_output = SearchTaskEvaluationOutput.from_search_task_output(
                output, self.dataset
            )

            self.logger.info(
                "query_evaluated",
                query_id=query_id,
                recall=eval_output.recall,
                precision=eval_output.precision,
                f1=eval_output.f1,
                trajectory_recall=eval_output.trajectory_recall,
                final_answer_recall=eval_output.final_answer_recall,
                num_turns=eval_output.num_turns,
                prune_accuracy=eval_output.prune_accuracy,
            )
            return eval_output
        except Exception as exc:  # pragma: no cover - defensive logging
            self.logger.exception(
                "query_evaluation_failed", query_id=query_id, error=str(exc)
            )
            # Try to save partial trajectory even on failure
            if agent is not None:
                try:
                    # Agent stores partial trajectory even if __call__ throws
                    partial_trajectory = agent.trajectory
                    partial_output = SearchTaskOutput(
                        trajectory=partial_trajectory,
                        query_id=query_id,
                        dataset_name=self.dataset.name,
                    )
                    self._write_search_task_output(partial_output)
                    self.logger.info(
                        "partial_trajectory_saved",
                        query_id=query_id,
                    )
                except Exception as save_exc:
                    self.logger.warning(
                        "failed_to_save_partial_trajectory",
                        query_id=query_id,
                        error=str(save_exc),
                    )
            return SearchTaskEvaluationOutput(
                query_id=query_id,
                error=str(exc),
            )

    def _summarize(self, results: List[SearchTaskEvaluationOutput]) -> dict:
        successful = [result for result in results if result.succeeded()]

        def _mean(field: str) -> float:
            if not successful:
                return 0.0
            total = 0.0
            count = 0
            for result in successful:
                value = getattr(result, field)
                if value is None:
                    continue
                total += value
                count += 1
            return total / count if count else 0.0

        return {
            "num_queries": len(results),
            "num_succeeded": len(successful),
            "num_failed": len(results) - len(successful),
            "mean_recall": _mean("recall"),
            "mean_precision": _mean("precision"),
            "mean_f1": _mean("f1"),
            "mean_trajectory_recall": _mean("trajectory_recall"),
            "mean_num_turns": _mean("num_turns"),
            "mean_prune_accuracy": _mean("prune_accuracy"),
            "mean_final_answer_recall": _mean("final_answer_recall"),
            "mean_rerank_recall": _mean("rerank_recall"),
            "mean_rerank_dropped_relevant_count": _mean(
                "rerank_dropped_relevant_count"
            ),
        }

    def _write_outputs(
        self,
        results: List[SearchTaskEvaluationOutput],
        summary: dict,
    ) -> None:
        per_query_path = self.output_dir / "per_query_metrics.json"
        summary_path = self.output_dir / "summary.json"
        with per_query_path.open("w") as fp:
            json.dump([result.model_dump() for result in results], fp, indent=2)
        with summary_path.open("w") as fp:
            json.dump(summary, fp, indent=2)

    def _write_search_task_output(self, output: SearchTaskOutput) -> None:
        file_name = f"{output.query_id}.json"
        file_path = self.search_outputs_dir / file_name
        with file_path.open("w", encoding="utf-8") as fp:
            json.dump(output.model_dump(mode="json"), fp, indent=2)
        self.logger.debug(
            "search_task_output_saved",
            query_id=output.query_id,
            path=str(file_path),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run retrieval agent evaluation on a sampled subset of queries."
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="browsecompplus",
        choices=list(DATASET_REGISTRY.keys()),
        help="Name of the search dataset to use.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write per-query metrics and summary JSON files.",
    )
    parser.add_argument(
        "--num-queries",
        type=int,
        required=True,
        help="Number of queries to sample for evaluation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used to select the evaluation subset.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of worker threads to use for evaluation.",
    )
    parser.add_argument(
        "--chroma-collection",
        type=str,
        required=True,
        help="Chroma collection name backing the retrieval tools.",
    )
    parser.add_argument(
        "--inference-provider",
        type=str,
        default="moonshot",
        choices=["moonshot", "anthropic", "tinker", "openai"],
        help="Inference provider used to run the agent.",
    )
    parser.add_argument(
        "--moonshot-model",
        type=str,
        default="kimi-k2-thinking",
        help="Moonshot model name when provider is moonshot.",
    )
    parser.add_argument(
        "--anthropic-model",
        type=str,
        default="claude-opus-4-5-20251101",
        help="Anthropic model name when provider is anthropic.",
    )
    parser.add_argument(
        "--openai-model",
        type=str,
        default="gpt-5",
        help="OpenAI model name (e.g., chatgpt-5) when provider is openai.",
    )
    parser.add_argument(
        "--tinker-model",
        type=str,
        default="openai/gpt-oss-20b",
        help="Tinker base model when provider is tinker.",
    )
    parser.add_argument(
        "--tinker-model-path",
        type=str,
        default=None,
        help="Optional Tinker sampler weights path overriding --tinker-model.",
    )
    parser.add_argument(
        "--max-completion-tokens",
        type=int,
        default=4096,
        help="Maximum completion tokens for the inference provider.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature for the inference provider.",
    )
    parser.add_argument(
        "--threshold-budget",
        type=int,
        default=16384,
        help="Token threshold that triggers pruning in the retrieval subagent.",
    )
    parser.add_argument(
        "--token-budget",
        type=int,
        default=32768,
        help="Maximum token budget allowed for the retrieval subagent.",
    )
    parser.add_argument(
        "--max-trajectory-length",
        type=int,
        default=64,
        help="Maximum number of steps per trajectory.",
    )
    parser.add_argument(
        "--rerank-max-tokens",
        type=int,
        default=4096,
        help="Maximum tokens for reranker and ReadDocumentTool output.",
    )
    parser.add_argument(
        "--reranker",
        type=str,
        default="baseten",
        choices=["baseten", "contextual"],
        help="Reranker to use (default: baseten).",
    )
    parser.add_argument(
        "--search-display-limit",
        type=int,
        default=10,
        help="Number of search results shown to the agent per search call (default: 10).",
    )
    parser.add_argument(
        "--max-query-retries",
        type=int,
        default=None,
        help=(
            "Maximum retries for failed queries (retries are in addition to the "
            "initial attempt). If unset, auto-enables 3 retries for "
            "tinker openai/gpt-oss-120b on transfer datasets."
        ),
    )
    return parser.parse_args()


def resolve_max_query_retries(args: argparse.Namespace) -> int:
    if args.max_query_retries is not None:
        return args.max_query_retries
    model_name = (args.tinker_model or "").lower()
    if (
        args.inference_provider == "tinker"
        and "gpt-oss-120b" in model_name
        and args.dataset_name in _RETRY_ENABLED_TRANSFER_DATASETS
    ):
        return 3
    return 0


def main() -> None:
    args = parse_args()
    max_query_retries = resolve_max_query_retries(args)

    dataset = get_dataset(args.dataset_name)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Create reranker with tiktoken-based token counter
    config = get_config()
    tiktoken_encoding = tiktoken.get_encoding("o200k_harmony")
    rerank_token_counter = lambda text: len(tiktoken_encoding.encode(text))

    reranker: Reranker
    if args.reranker == "contextual":
        reranker = ContextualReranker(
            token_counter=rerank_token_counter,
            max_tokens=args.rerank_max_tokens,
        )
    else:
        reranker = BasetenReranker(
            token_counter=rerank_token_counter,
            max_tokens=args.rerank_max_tokens,
        )

    # Create shared toolset
    toolset = ToolSet.from_config(
        config,
        chroma_collection_name=args.chroma_collection,
        reranker=reranker,
        token_counter=rerank_token_counter,
        max_tokens=args.rerank_max_tokens,
        search_display_limit=args.search_display_limit,
    )

    inference_model_factory = create_inference_model_factory(args, strict_mode=False)
    agent_factory = build_agent_factory(
        args=args,
        inference_model_factory=inference_model_factory,
        toolset=toolset,
    )

    runner = EvaluationRunner(
        dataset=dataset,
        agent_factory=agent_factory,
        sample_size=args.num_queries,
        seed=args.seed,
        num_workers=args.num_workers,
        output_dir=args.output_dir,
        max_query_retries=max_query_retries,
    )
    logger.info(
        "query_retry_policy",
        dataset=args.dataset_name,
        inference_provider=args.inference_provider,
        tinker_model=args.tinker_model,
        max_query_retries=max_query_retries,
    )
    summary = runner.run()
    summary["max_query_retries"] = max_query_retries
    logger.info("evaluation_completed", summary=summary)


if __name__ == "__main__":
    main()
