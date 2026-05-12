"""Shared Harness-1 (multi-rollout + RRF) helpers for web/wiki evaluation.

Chroma harness comparisons use ``compare_harnesses.evaluate_harness1``, which
fuses *internal* chunk IDs. Web and wiki runs resolve synthetic IDs to URLs
before calling ``SearchDataset.evaluate_*``; this module applies the same RRF
fusion and metric aggregation on those resolved ID lists.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, List, Optional, Sequence, Set

from harness.tasks import SearchTaskEvaluationOutput, chunk_ids_to_doc_ids

if TYPE_CHECKING:
    from datagen.search_dataset import SearchDataset


def dedupe_preserve_order(values: Sequence[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[str]],
    *,
    rrf_k: int = 60,
    max_results: Optional[int] = None,
) -> List[str]:
    scores: dict[str, float] = defaultdict(float)
    for ranking in ranked_lists:
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] += 1.0 / (rrf_k + rank)

    fused = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    fused_chunk_ids = [chunk_id for chunk_id, _ in fused]
    if max_results is not None:
        fused_chunk_ids = fused_chunk_ids[:max_results]
    return fused_chunk_ids


def build_harness1_evaluation_output(
    *,
    query_id: str,
    dataset: "SearchDataset",
    per_rollout_evals: List[SearchTaskEvaluationOutput],
    ranked_resolved_lists: List[List[str]],
    trajectory_resolved_sorted: List[str],
    rrf_k: int,
    max_fused_results: Optional[int],
) -> SearchTaskEvaluationOutput:
    """Aggregate per-rollout web/wiki evaluations into a single Harness-1 row."""
    fused = reciprocal_rank_fusion(
        ranked_resolved_lists,
        rrf_k=rrf_k,
        max_results=max_fused_results,
    )
    recall = dataset.evaluate_results_recall(query_id, fused)
    precision = dataset.evaluate_results_precision(query_id, fused)
    f1 = dataset.evaluate_results_f1_score(query_id, fused)
    trajectory_recall = dataset.evaluate_results_recall(
        query_id, trajectory_resolved_sorted
    )
    final_answer_recall = dataset.evaluate_results_final_answer_recall(
        query_id, fused
    )

    num_turns_vals = [e.num_turns for e in per_rollout_evals if e.num_turns is not None]
    avg_turns = round(sum(num_turns_vals) / len(num_turns_vals)) if num_turns_vals else None

    prune_vals = [
        e.prune_accuracy for e in per_rollout_evals if e.prune_accuracy is not None
    ]
    rerank_vals = [
        e.rerank_recall for e in per_rollout_evals if e.rerank_recall is not None
    ]
    rerank_drop_vals = [
        e.rerank_dropped_relevant_count
        for e in per_rollout_evals
        if e.rerank_dropped_relevant_count is not None
    ]

    return SearchTaskEvaluationOutput(
        query_id=query_id,
        recall=recall,
        precision=precision,
        f1=f1,
        trajectory_recall=trajectory_recall,
        final_answer_recall=final_answer_recall,
        retrieved_document_ids=sorted(chunk_ids_to_doc_ids(set(fused))),
        num_turns=avg_turns,
        prune_accuracy=(sum(prune_vals) / len(prune_vals)) if prune_vals else None,
        rerank_recall=(sum(rerank_vals) / len(rerank_vals)) if rerank_vals else None,
        rerank_dropped_relevant_count=(
            round(sum(rerank_drop_vals) / len(rerank_drop_vals))
            if rerank_drop_vals
            else None
        ),
    )
