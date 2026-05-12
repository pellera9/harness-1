"""Wikipedia-based evaluation runner for search agent.

This module provides evaluation infrastructure for Wikipedia-based search tools,
using Serper API (with site:wikipedia.org filter) for search and the `wikipediaapi`
library for fetching page content.
"""

from __future__ import annotations

import argparse
import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import structlog
import tiktoken

from harness.agent import (
    Agent,
    AgentInferenceModel,
    AnthropicAgentInferenceModel,
    MoonshotAgentInferenceModel,
    OpenAIAgentInferenceModel,
    TinkerAgentInferenceModel,
    TokenBudgetRetrievalSubagent,
    prune_chunks_from_trajectory,
)
from harness.config import get_config
from datagen.search_dataset import SearchDataset, get_dataset, DATASET_REGISTRY
from harness.generate_search_sft import create_inference_model_factory
from openai_harmony import (
    HarmonyEncodingName,
    RenderConversationConfig,
    load_harmony_encoding,
)
from harness.prompts import get_retrieval_subagent_prompt
from harness.rerank import BasetenReranker
from harness.tasks import SearchTaskOutput, SearchTaskEvaluationOutput, chunk_ids_to_doc_ids
from harness.tools import (
    PruneChunksTool,
    SearchCorpusToolCallMetadata,
    Tool,
    ToolCallMetadata,
    ToolSet,
)
from harness.trajectory import Observation, Trajectory

try:
    from eval_scripts.resolved_harness_rrf import (
        build_harness1_evaluation_output,
        dedupe_preserve_order,
        reciprocal_rank_fusion,
    )
except ImportError:
    from resolved_harness_rrf import (
        build_harness1_evaluation_output,
        dedupe_preserve_order,
        reciprocal_rank_fusion,
    )

try:
    from eval_scripts.web_tools import URLMapper, require_serper_health
    from eval_scripts.wiki_tools import (
        WikiGrepCorpusTool,
        WikiReadDocumentTool,
        WikiSearchCorpusTool,
        WikiToolSet,
    )
except ImportError:
    from web_tools import URLMapper, require_serper_health
    from wiki_tools import (
        WikiGrepCorpusTool,
        WikiReadDocumentTool,
        WikiSearchCorpusTool,
        WikiToolSet,
    )
from pydantic import Field

logger = structlog.get_logger("search_agent.wiki_eval")
_RETRY_ENABLED_TRANSFER_DATASETS = {
    "seal0qa",
    "longsealqa",
    "frames",
    "hotpotqa_subset",
}

try:
    from harness.agent import UnlimitedContextAgent
except ImportError:
    # Compatibility shim for branches where UnlimitedContextAgent was removed.
    # The evaluation path used in this work is budgeted mode (no --no-budget),
    # so this class is only a fallback to keep module import stable.
    class UnlimitedContextAgent(TokenBudgetRetrievalSubagent):
        def __init__(
            self,
            toolset: ToolSet,
            inference_model: AgentInferenceModel,
            token_counter: Callable[[Trajectory], int],
            force_output_threshold: int = 175000,
            text_token_counter: Optional[Callable[[str], int]] = None,
            max_trajectory_length: int = 128,
            show_token_budget: bool = False,
        ) -> None:
            super().__init__(
                toolset=toolset,
                inference_model=inference_model,
                token_counter=token_counter,
                text_token_counter=text_token_counter,
                max_trajectory_length=max_trajectory_length,
            )


# ============================================================================
# Wiki Agent with URL Mapping
# ============================================================================


class WikiTokenBudgetRetrievalSubagent(TokenBudgetRetrievalSubagent):
    """Agent with URL mapping AND token budget tracking for Wikipedia-based tools.

    Extends TokenBudgetRetrievalSubagent (which extends DeduplicatingPruningSearchAgent)
    to add URL-to-ID mapping for wiki evaluation.
    """

    _url_mapper: URLMapper

    def __init__(
        self,
        toolset: ToolSet,
        inference_model: AgentInferenceModel,
        token_counter: Callable[[Trajectory], int],
        text_token_counter: Optional[Callable[[str], int]] = None,
        max_trajectory_length: int = 128,
        threshold_budget: int = 16384,
        token_budget: int = 32268,
        tool_output_budget: int = 4096,
        spillage_fraction: float = 0.5,
    ) -> None:
        super().__init__(
            toolset=toolset,
            inference_model=inference_model,
            token_counter=token_counter,
            text_token_counter=text_token_counter,
            max_trajectory_length=max_trajectory_length,
            threshold_budget=threshold_budget,
            token_budget=token_budget,
            tool_output_budget=tool_output_budget,
            spillage_fraction=spillage_fraction,
        )
        self._url_mapper = URLMapper()

    def reset(self) -> None:
        """Reset the agent state including URL mapper."""
        super().reset()
        self._url_mapper = URLMapper()

    def _call_tool(
        self,
        tool: Tool,
        params: Dict[Any, Any],
        overrides: Optional[Dict[Any, Any]] = None,
    ) -> Tuple[str, Optional[ToolCallMetadata]]:
        """Call tool with url_mapper injected into overrides."""
        overrides = overrides or {}
        overrides["url_mapper"] = self._url_mapper

        # For wiki tools, track the query for read_document semantic search
        if isinstance(tool, WikiSearchCorpusTool):
            query = params.get("query", "")
            tool_output, tool_metadata = super()._call_tool(tool, params, overrides)

            if tool_metadata is not None and isinstance(
                tool_metadata, SearchCorpusToolCallMetadata
            ):
                for chunk_id in tool_metadata.returned_chunk_ids:
                    if chunk_id not in self._doc_id_to_query:
                        self._doc_id_to_query[chunk_id] = query

            return tool_output, tool_metadata

        if isinstance(tool, WikiReadDocumentTool):
            doc_id = params.get("doc_id") or params.get("id", "")
            if doc_id in self._doc_id_to_query:
                overrides["query"] = self._doc_id_to_query[doc_id]
            return super()._call_tool(tool, params, overrides)

        return super()._call_tool(tool, params, overrides)

    def get_url_mapping(self) -> Dict[str, str]:
        """Get ID-to-URL mapping for serialization."""
        return self._url_mapper.get_mapping()


class WikiUnlimitedContextAgent(UnlimitedContextAgent):
    """Agent with URL mapping for Wikipedia-based tools, without token budgeting.

    Extends UnlimitedContextAgent (no pruning/budgeting) to add URL-to-ID
    mapping for wiki evaluation.
    """

    _url_mapper: URLMapper

    def __init__(
        self,
        toolset: ToolSet,
        inference_model: AgentInferenceModel,
        token_counter: Callable[[Trajectory], int],
        force_output_threshold: int = 175000,
        text_token_counter: Optional[Callable[[str], int]] = None,
        max_trajectory_length: int = 128,
        show_token_budget: bool = False,
    ) -> None:
        super().__init__(
            toolset=toolset,
            inference_model=inference_model,
            token_counter=token_counter,
            force_output_threshold=force_output_threshold,
            text_token_counter=text_token_counter,
            max_trajectory_length=max_trajectory_length,
            show_token_budget=show_token_budget,
        )
        self._url_mapper = URLMapper()

    def reset(self) -> None:
        """Reset the agent state including URL mapper."""
        super().reset()
        self._url_mapper = URLMapper()

    def _call_tool(
        self,
        tool: Tool,
        params: Dict[Any, Any],
        overrides: Optional[Dict[Any, Any]] = None,
    ) -> Tuple[str, Optional[ToolCallMetadata]]:
        """Call tool with url_mapper injected into overrides."""
        overrides = overrides or {}
        overrides["url_mapper"] = self._url_mapper

        # For wiki tools, track the query for read_document semantic search
        if isinstance(tool, WikiSearchCorpusTool):
            query = params.get("query", "")
            tool_output, tool_metadata = super()._call_tool(tool, params, overrides)

            if tool_metadata is not None and isinstance(
                tool_metadata, SearchCorpusToolCallMetadata
            ):
                for chunk_id in tool_metadata.returned_chunk_ids:
                    if chunk_id not in self._doc_id_to_query:
                        self._doc_id_to_query[chunk_id] = query

            return tool_output, tool_metadata

        if isinstance(tool, WikiReadDocumentTool):
            doc_id = params.get("doc_id") or params.get("id", "")
            if doc_id in self._doc_id_to_query:
                overrides["query"] = self._doc_id_to_query[doc_id]
            return super()._call_tool(tool, params, overrides)

        return super()._call_tool(tool, params, overrides)

    def get_url_mapping(self) -> Dict[str, str]:
        """Get ID-to-URL mapping for serialization."""
        return self._url_mapper.get_mapping()


# ============================================================================
# Wiki Search Task Output
# ============================================================================


class WikiSearchTaskOutput(SearchTaskOutput):
    """SearchTaskOutput with URL mapping for wiki evaluation.

    Extends SearchTaskOutput to include the URL mapping dictionary,
    which is needed to resolve output IDs back to URLs for evaluation.
    """

    url_mapping: Dict[str, str] = Field(default_factory=dict)  # ID -> URL

    def get_resolved_output_chunk_ids(self) -> List[str]:
        """Resolve output IDs back to URLs for evaluation."""
        resolved = []
        for chunk_id in self.output_chunk_ids:
            url = self.url_mapping.get(chunk_id)
            if url:
                resolved.append(url)
            else:
                resolved.append(chunk_id)
        return resolved

    def get_resolved_traversed_chunk_ids(self) -> List[str]:
        """Resolve traversed IDs back to URLs."""
        resolved = []
        for chunk_id in self.nondeduplicated_traversed_chunk_ids:
            url = self.url_mapping.get(chunk_id)
            if url:
                resolved.append(url)
            else:
                resolved.append(chunk_id)
        return resolved

    def get_unique_resolved_output_document_ids(self) -> Set[str]:
        """Get unique document IDs (URLs) from output."""
        resolved_ids = self.get_resolved_output_chunk_ids()
        return chunk_ids_to_doc_ids(set(resolved_ids))


# ============================================================================
# Wiki Evaluation Runner
# ============================================================================


class WikiEvaluationRunner:
    """Run evaluation on Wikipedia-based search tasks.

    Similar to WebEvaluationRunner but uses wiki tools instead of generic web tools.
    Handles URL mapping resolution for computing metrics.
    """

    def __init__(
        self,
        *,
        dataset: SearchDataset,
        agent_factory: Callable[[], Agent],
        sample_size: int,
        seed: int,
        num_workers: int,
        output_dir: Path,
        split: str = "test",
        num_output_docs: int | None = None,
        retrieval_harness: str = "context1",
        num_rollouts: int = 4,
        rollout_workers: int = 1,
        rrf_k: int = 60,
        max_fused_results: int | None = 30,
        max_query_retries: int = 0,
    ) -> None:
        if sample_size < 1:
            raise ValueError("sample_size must be >= 1")
        if num_workers < 1:
            raise ValueError("num_workers must be >= 1")
        if max_query_retries < 0:
            raise ValueError("max_query_retries must be >= 0")
        if retrieval_harness not in ("context1", "harness1"):
            raise ValueError(
                f"retrieval_harness must be 'context1' or 'harness1', got {retrieval_harness!r}"
            )
        if retrieval_harness == "harness1" and num_rollouts < 1:
            raise ValueError("num_rollouts must be >= 1 for harness1")

        self.dataset = dataset
        self.agent_factory = agent_factory
        self.sample_size = sample_size
        self.seed = seed
        self.num_workers = num_workers
        self.output_dir = output_dir
        self.search_outputs_dir = output_dir / "search_task_outputs"
        self.split = None if split == "all" else split
        self.num_output_docs = num_output_docs
        self.retrieval_harness = retrieval_harness
        self.num_rollouts = num_rollouts
        self.rollout_workers = rollout_workers
        self.rrf_k = rrf_k
        self.max_fused_results = max_fused_results
        self.max_query_retries = max_query_retries
        self.logger = logger.bind(component="WikiEvaluationRunner")

    def run(self) -> dict:
        """Execute the evaluation and persist aggregated metrics."""
        require_serper_health()
        self.logger.info("serper_preflight_passed")
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
                    except Exception as exc:
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
                            final_answer_recall=None,
                            retrieved_document_ids=[],
                            error=str(exc),
                        )
                    results.append(result)

        summary = self._summarize(results)
        if self.retrieval_harness == "harness1":
            summary["retrieval_harness"] = "harness1"
            summary["num_rollouts"] = self.num_rollouts
            summary["rollout_workers"] = self.rollout_workers
            summary["rrf_k"] = self.rrf_k
            summary["max_fused_results"] = self.max_fused_results
            summary["fusion_method"] = "rrf"
        else:
            summary["retrieval_harness"] = "context1"
        self._write_outputs(results, summary)
        self.logger.info(
            "evaluation_finished",
            succeeded=summary["num_succeeded"],
            failed=summary["num_failed"],
            mean_recall=summary["mean_recall"],
            mean_precision=summary["mean_precision"],
            mean_f1=summary["mean_f1"],
            mean_trajectory_recall=summary["mean_trajectory_recall"],
            mean_final_answer_recall=summary["mean_final_answer_recall"],
            mean_prune_accuracy=summary["mean_prune_accuracy"],
        )
        return summary

    def _select_query_ids(self) -> List[str]:
        available_query_ids = self.dataset.get_all_query_ids(split=self.split)
        if len(available_query_ids) < self.sample_size:
            raise ValueError(
                f"Requested {self.sample_size} queries but dataset only has "
                f"{len(available_query_ids)} entries."
            )
        rng = random.Random(self.seed)
        rng.shuffle(available_query_ids)
        selected = available_query_ids[: self.sample_size]
        self.logger.debug("selected_queries", count=len(selected))

        # Filter out queries that already succeeded
        to_run = []
        skipped = 0
        for query_id in selected:
            output_path = self.search_outputs_dir / f"{query_id}.json"
            if output_path.exists():
                try:
                    with output_path.open("r", encoding="utf-8") as fp:
                        data = json.load(fp)
                    if self.retrieval_harness == "harness1":
                        he = data.get("harness_evaluation")
                        if (
                            isinstance(he, dict)
                            and he.get("kind") == "harness1"
                            and he.get("completed") is True
                            and he.get("num_rollouts_requested") == self.num_rollouts
                            and he.get("rrf_k") == self.rrf_k
                            and he.get("max_fused_results") == self.max_fused_results
                        ):
                            skipped += 1
                            continue
                    else:
                        output_chunk_ids = data.get("output_chunk_ids", [])
                        if self.num_output_docs is not None:
                            if len(output_chunk_ids) == self.num_output_docs:
                                skipped += 1
                                continue
                        elif output_chunk_ids:
                            skipped += 1
                            continue
                except (json.JSONDecodeError, IOError):
                    pass
            to_run.append(query_id)

        self.logger.info(
            "query_selection_complete",
            total_selected=len(selected),
            skipped_succeeded=skipped,
            to_run=len(to_run),
        )
        return to_run

    def _run_one_rollout(self, query_id: str) -> WikiSearchTaskOutput:
        agent = self.agent_factory()
        _, query_text = self.dataset.get_query_by_id(query_id)
        try:
            prompt = get_retrieval_subagent_prompt(
                query_text, num_output_docs=self.num_output_docs
            )
        except TypeError:
            prompt = get_retrieval_subagent_prompt(query_text)
        initial_observation = Observation(
            observations=[prompt],
            sources=["user"],
            tool_metadata=[None],
        )
        trajectory = agent(initial_observation=initial_observation)
        return WikiSearchTaskOutput(
            trajectory=trajectory,
            query_id=query_id,
            dataset_name=self.dataset.name,
            url_mapping=agent.get_url_mapping(),
        )

    def _evaluate_single(self, query_id: str) -> SearchTaskEvaluationOutput:
        max_attempts = self.max_query_retries + 1
        result: SearchTaskEvaluationOutput | None = None
        for attempt in range(1, max_attempts + 1):
            if self.retrieval_harness == "harness1":
                result = self._evaluate_single_harness1(query_id)
            else:
                result = self._evaluate_single_context1(query_id)
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

    def _evaluate_single_context1(self, query_id: str) -> SearchTaskEvaluationOutput:
        try:
            output = self._run_one_rollout(query_id)
            self._write_search_task_output(output)
            eval_output = self._create_evaluation_output(output)

            self.logger.info(
                "query_evaluated",
                query_id=query_id,
                recall=eval_output.recall,
                precision=eval_output.precision,
                f1=eval_output.f1,
                trajectory_recall=eval_output.trajectory_recall,
                num_turns=eval_output.num_turns,
                prune_accuracy=eval_output.prune_accuracy,
            )
            return eval_output
        except Exception as exc:
            self.logger.exception(
                "query_evaluation_failed", query_id=query_id, error=str(exc)
            )
            return SearchTaskEvaluationOutput(
                query_id=query_id,
                error=str(exc),
            )

    def _evaluate_single_harness1(self, query_id: str) -> SearchTaskEvaluationOutput:
        outputs: List[WikiSearchTaskOutput] = []
        rollout_errors: List[Dict[str, Any]] = []

        def _run_rollout(_idx: int) -> WikiSearchTaskOutput:
            return self._run_one_rollout(query_id)

        if self.rollout_workers == 1 or self.num_rollouts == 1:
            for rollout_idx in range(self.num_rollouts):
                try:
                    outputs.append(_run_rollout(rollout_idx))
                except Exception as exc:
                    rollout_errors.append(
                        {"rollout_idx": rollout_idx, "error": str(exc)}
                    )
        else:
            with ThreadPoolExecutor(
                max_workers=min(self.rollout_workers, self.num_rollouts)
            ) as executor:
                futures = {
                    executor.submit(_run_rollout, rollout_idx): rollout_idx
                    for rollout_idx in range(self.num_rollouts)
                }
                for future in as_completed(futures):
                    rollout_idx = futures[future]
                    try:
                        outputs.append(future.result())
                    except Exception as exc:
                        rollout_errors.append(
                            {"rollout_idx": rollout_idx, "error": str(exc)}
                        )

        try:
            if not outputs:
                first_error = rollout_errors[0]["error"] if rollout_errors else None
                if first_error:
                    raise RuntimeError(
                        f"all {self.num_rollouts} rollouts failed; first error: {first_error}"
                    )
                raise RuntimeError(f"all {self.num_rollouts} rollouts failed")

            per_rollout_evals = [
                self._create_evaluation_output(o) for o in outputs
            ]
            ranked_resolved = [
                dedupe_preserve_order(o.get_resolved_output_chunk_ids()) for o in outputs
            ]
            fused_resolved = reciprocal_rank_fusion(
                ranked_resolved,
                rrf_k=self.rrf_k,
                max_results=self.max_fused_results,
            )
            trajectory_union: Set[str] = set()
            for o in outputs:
                trajectory_union.update(o.get_resolved_traversed_chunk_ids())
            trajectory_sorted = sorted(trajectory_union)

            eval_output = build_harness1_evaluation_output(
                query_id=query_id,
                dataset=self.dataset,
                per_rollout_evals=per_rollout_evals,
                ranked_resolved_lists=ranked_resolved,
                trajectory_resolved_sorted=trajectory_sorted,
                rrf_k=self.rrf_k,
                max_fused_results=self.max_fused_results,
            )

            self._write_harness1_search_task_output(
                outputs[0],
                fused_resolved=fused_resolved,
                num_successful_rollouts=len(outputs),
                num_failed_rollouts=len(rollout_errors),
                completed=True,
            )

            self.logger.info(
                "query_evaluated_harness1",
                query_id=query_id,
                recall=eval_output.recall,
                num_rollouts=len(outputs),
                failed_rollouts=len(rollout_errors),
            )
            return eval_output
        except Exception as exc:
            self.logger.exception(
                "query_evaluation_failed", query_id=query_id, error=str(exc)
            )
            return SearchTaskEvaluationOutput(
                query_id=query_id,
                error=str(exc),
            )

    def _write_harness1_search_task_output(
        self,
        primary: WikiSearchTaskOutput,
        *,
        fused_resolved: List[str],
        num_successful_rollouts: int,
        num_failed_rollouts: int,
        completed: bool,
    ) -> None:
        file_name = f"{primary.query_id}.json"
        file_path = self.search_outputs_dir / file_name
        payload = primary.model_dump(mode="json")
        payload["harness_evaluation"] = {
            "kind": "harness1",
            "completed": completed,
            "num_rollouts_requested": self.num_rollouts,
            "num_rollouts_successful": num_successful_rollouts,
            "num_rollouts_failed": num_failed_rollouts,
            "rrf_k": self.rrf_k,
            "max_fused_results": self.max_fused_results,
        }
        payload["fused_resolved_chunk_ids"] = fused_resolved
        with file_path.open("w", encoding="utf-8") as fp:
            json.dump(payload, fp, indent=2)
        self.logger.debug(
            "search_task_output_saved_harness1",
            query_id=primary.query_id,
            path=str(file_path),
        )

    def _create_evaluation_output(
        self, output: WikiSearchTaskOutput
    ) -> SearchTaskEvaluationOutput:
        """Create evaluation output with URL-resolved metrics."""
        query_id = output.query_id

        # Resolve IDs to URLs for evaluation
        resolved_output_ids = output.get_resolved_output_chunk_ids()
        resolved_traversed_ids = output.get_resolved_traversed_chunk_ids()

        # Calculate metrics with resolved URLs
        recall = self.dataset.evaluate_results_recall(query_id, resolved_output_ids)
        precision = self.dataset.evaluate_results_precision(
            query_id, resolved_output_ids
        )
        f1 = self.dataset.evaluate_results_f1_score(query_id, resolved_output_ids)
        trajectory_recall = self.dataset.evaluate_results_recall(
            query_id, resolved_traversed_ids
        )
        final_answer_recall = self.dataset.evaluate_results_final_answer_recall(
            query_id, resolved_output_ids
        )
        num_turns = output.trajectory.num_turns

        # Calculate prune accuracy
        prune_accuracy = self._calculate_prune_accuracy(output)

        return SearchTaskEvaluationOutput(
            query_id=query_id,
            recall=recall,
            precision=precision,
            f1=f1,
            trajectory_recall=trajectory_recall,
            final_answer_recall=final_answer_recall,
            retrieved_document_ids=sorted(
                output.get_unique_resolved_output_document_ids()
            ),
            num_turns=num_turns,
            prune_accuracy=prune_accuracy,
        )

    def _calculate_prune_accuracy(
        self, output: WikiSearchTaskOutput
    ) -> Optional[float]:
        """Calculate prune accuracy with URL resolution."""
        pruned_chunk_ids = output.get_all_pruned_chunk_ids()
        if not pruned_chunk_ids:
            return None

        # Resolve pruned IDs to URLs
        resolved_pruned = []
        for chunk_id in pruned_chunk_ids:
            url = output.url_mapping.get(chunk_id)
            resolved_pruned.append(url if url else chunk_id)

        # Get expected document IDs from dataset
        expected_chunk_ids = set(self.dataset.get_expected_document_ids(output.query_id))
        expected_doc_ids = chunk_ids_to_doc_ids(expected_chunk_ids)

        # Count bad prunes
        bad_prunes = 0
        for chunk_id in resolved_pruned:
            doc_id_set = chunk_ids_to_doc_ids({chunk_id})
            doc_id = next(iter(doc_id_set))
            if doc_id in expected_doc_ids:
                bad_prunes += 1

        total_prunes = len(resolved_pruned)
        correct_prunes = total_prunes - bad_prunes
        return correct_prunes / total_prunes

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
            "mean_final_answer_recall": _mean("final_answer_recall"),
            "mean_num_turns": _mean("num_turns"),
            "mean_prune_accuracy": _mean("prune_accuracy"),
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

    def _write_search_task_output(self, output: WikiSearchTaskOutput) -> None:
        file_name = f"{output.query_id}.json"
        file_path = self.search_outputs_dir / file_name
        with file_path.open("w", encoding="utf-8") as fp:
            json.dump(output.model_dump(mode="json"), fp, indent=2)
        self.logger.debug(
            "search_task_output_saved",
            query_id=output.query_id,
            path=str(file_path),
        )


# ============================================================================
# CLI
# ============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Wikipedia-based retrieval agent evaluation on a sampled subset of queries."
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="web_test",
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
        "--split",
        type=str,
        default="test",
        choices=["train", "test", "all"],
        help="Dataset split to evaluate on. Use 'all' for full dataset.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of worker threads to use for evaluation.",
    )
    parser.add_argument(
        "--inference-provider",
        type=str,
        default="moonshot",
        choices=["moonshot", "anthropic", "tinker", "openai", "gemini", "together"],
        help="Inference provider used to run the agent.",
    )
    parser.add_argument(
        "--together-model",
        type=str,
        default="Qwen/Qwen3.5-397B-A17B",
        help="Together AI model name when provider is together.",
    )
    parser.add_argument(
        "--gemini-model",
        type=str,
        default="gemini-3.1-pro-preview",
        help="Gemini model name when provider is gemini.",
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
        default="claude-sonnet-4-5@20250929",
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
        "--summary-only",
        action="store_true",
        default=False,
        help="When set, WikiReadDocumentTool returns only page title + summary (no full text).",
    )
    parser.add_argument(
        "--no-budget",
        action="store_true",
        default=False,
        help="Disable token budgeting/pruning. Agent runs freely until --force-output-tokens, then forces output with no tools.",
    )
    parser.add_argument(
        "--force-output-tokens",
        type=int,
        default=175000,
        help="Token threshold at which to force output when --no-budget is set.",
    )
    parser.add_argument(
        "--show-token-budget",
        action="store_true",
        default=False,
        help="When used with --no-budget, show token usage counter in observations (visible to the model) without enforcing pruning.",
    )
    parser.add_argument(
        "--num-output-docs",
        type=int,
        default=None,
        help="If set, require the agent to output exactly this many ranked documents. Outputs with a different count are treated as failures and rerun.",
    )
    parser.add_argument(
        "--retrieval-harness",
        type=str,
        default="context1",
        choices=["context1", "harness1"],
        help="Retrieval harness to run: single-rollout context1 or multi-rollout harness1.",
    )
    parser.add_argument(
        "--num-rollouts",
        type=int,
        default=4,
        help="Rollouts per query for --retrieval-harness harness1.",
    )
    parser.add_argument(
        "--rollout-workers",
        type=int,
        default=1,
        help="Parallel workers across rollouts within each query for harness1.",
    )
    parser.add_argument(
        "--rrf-k",
        type=int,
        default=60,
        help="RRF k constant used when fusing harness1 rollouts.",
    )
    parser.add_argument(
        "--max-fused-results",
        type=int,
        default=30,
        help="Maximum fused results kept for harness1; <=0 keeps all.",
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


def build_wiki_agent_factory(
    *,
    args: argparse.Namespace,
    inference_model_factory: Callable[[], AgentInferenceModel],
    toolset: WikiToolSet,
) -> Callable[[], Agent]:
    """Build factory for wiki agent (budgeted or unlimited)."""
    harmony_enc = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

    def token_counter(trajectory: Trajectory) -> int:
        return len(
            harmony_enc.render_conversation(
                trajectory.to_openai_harmony_format(),
                config=RenderConversationConfig(auto_drop_analysis=False),
            )
        )

    no_budget = getattr(args, "no_budget", False)

    if no_budget:
        force_output_tokens = getattr(args, "force_output_tokens", 175000)
        show_token_budget = getattr(args, "show_token_budget", False)

        def factory() -> WikiUnlimitedContextAgent:
            inference_model = inference_model_factory()
            return WikiUnlimitedContextAgent(
                toolset=toolset,
                inference_model=inference_model,
                token_counter=token_counter,
                force_output_threshold=force_output_tokens,
                max_trajectory_length=args.max_trajectory_length,
                show_token_budget=show_token_budget,
            )
    else:

        def factory() -> WikiTokenBudgetRetrievalSubagent:  # type: ignore[misc]
            inference_model = inference_model_factory()
            return WikiTokenBudgetRetrievalSubagent(
                toolset=toolset,
                inference_model=inference_model,
                token_counter=token_counter,
                max_trajectory_length=args.max_trajectory_length,
                threshold_budget=args.threshold_budget,
                token_budget=args.token_budget,
            )

    return factory


def main() -> None:
    args = parse_args()
    max_query_retries = resolve_max_query_retries(args)

    dataset = get_dataset(args.dataset_name)
    max_fused_results = (
        None if args.max_fused_results is not None and args.max_fused_results <= 0
        else args.max_fused_results
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Create reranker with tiktoken-based token counter
    config = get_config()
    tiktoken_encoding = tiktoken.get_encoding("o200k_harmony")
    rerank_token_counter = lambda text: len(tiktoken_encoding.encode(text))
    reranker = BasetenReranker(
        token_counter=rerank_token_counter,
        max_tokens=args.rerank_max_tokens,
    )

    # Create wiki toolset
    toolset = WikiToolSet.create(
        reranker=reranker,
        token_counter=rerank_token_counter,
        max_tokens=args.rerank_max_tokens,
        openai_client=config.get_openai_client(),
        summary_only=args.summary_only,
    )

    if args.no_budget:
        toolset.remove_tool("prune_chunks")
        logger.info("no_budget mode enabled, removed PruneChunksTool from toolset")

    # Match compare_harnesses.py / eval_harness1.py behavior for Tinker parsing.
    inference_model_factory = create_inference_model_factory(args, strict_mode=False)
    agent_factory = build_wiki_agent_factory(
        args=args,
        inference_model_factory=inference_model_factory,
        toolset=toolset,
    )

    runner = WikiEvaluationRunner(
        dataset=dataset,
        agent_factory=agent_factory,
        sample_size=args.num_queries,
        seed=args.seed,
        num_workers=args.num_workers,
        output_dir=args.output_dir,
        split=args.split,
        num_output_docs=args.num_output_docs,
        retrieval_harness=args.retrieval_harness,
        num_rollouts=args.num_rollouts,
        rollout_workers=args.rollout_workers,
        rrf_k=args.rrf_k,
        max_fused_results=max_fused_results,
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
