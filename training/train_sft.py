
# Allow direct execution from subdirectories while keeping imports package-relative.
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

"""SFT training for Ultra v3 — replays trajectories through WorkingMemory.

Key improvements over v2:
  - Uses build_context() from ultra_core (not manual message construction).
  - Replays result summaries during trajectory replay.
  - Correct normalize_ids per trajectory.
  - Analysis truncation automatically included via build_context().
  - Guaranteed identical context format to RL.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import chz
import structlog
import tiktoken
import tinker
import torch
from openai_harmony import (
    Author,
    Conversation,
    HarmonyEncodingName,
    Message,
    Role,
    load_harmony_encoding,
)
from tinker_cookbook.supervised import train
from tinker_cookbook.supervised.types import SupervisedDataset, SupervisedDatasetBuilder
from tinker_cookbook.supervised.common import datum_from_model_input_weights

from harness.trajectory import Action, Observation, ActionBuilder, ObservationBuilder
from harness.tools import ToolSet, UserTextTool
from harness.ultra_core import (
    WorkingMemory,
    build_context,
    build_result_summary,
    get_system_prompt,
    get_tool_descriptions,
    parse_doc_ids_from_observation,
    parse_doc_texts_from_observation,
    action_observation_to_messages,
    auto_populate_from_first_search,
    RECENT_K,
    MAX_OBS_CHARS,
    MAX_CURATED_DOCS,
    FAN_OUT_MAX_QUERIES,
    DOC_SNIPPET_CHARS,
    FAN_OUT_SEARCH_SCHEMA,
    CURATE_SCHEMA,
    END_SEARCH_SCHEMA,
    REVIEW_DOCS_SCHEMA,
    VERIFY_SCHEMA,
    V8D_AUTO_POPULATE_FIRST_SEARCH,
    V8D_IMPORTANCE_TAGGING,
    V8D_VERIFY_TOOL,
)
from harness.tools import (
    SEARCH_CORPUS_SCHEMA,
    GREP_CORPUS_SCHEMA,
    READ_DOCUMENT_SCHEMA,
)

logger = structlog.get_logger()


# ═══════════════════════════════════════════════════════════════════════════════
# Tool registry for converting trajectory dicts → Action objects
# ═══════════════════════════════════════════════════════════════════════════════

class _PlaceholderTool:
    """Lightweight tool object for replaying saved trajectories into Action objects."""
    def __init__(self, schema):
        self.tool_schema = schema

_TOOL_REGISTRY = {
    "fan_out_search": _PlaceholderTool(FAN_OUT_SEARCH_SCHEMA),
    "search_corpus": _PlaceholderTool(SEARCH_CORPUS_SCHEMA),
    "grep_corpus": _PlaceholderTool(GREP_CORPUS_SCHEMA),
    "read_document": _PlaceholderTool(READ_DOCUMENT_SCHEMA),
    "review_docs": _PlaceholderTool(REVIEW_DOCS_SCHEMA),
    "curate": _PlaceholderTool(CURATE_SCHEMA),
    "end_search": _PlaceholderTool(END_SEARCH_SCHEMA),
}
if V8D_VERIFY_TOOL:
    _TOOL_REGISTRY["verify"] = _PlaceholderTool(VERIFY_SCHEMA)


def _turn_to_action(turn: Dict) -> Action:
    """Convert a trajectory turn dict to an Action object."""
    tool_name = turn.get("tool_name", "")
    params = turn.get("params", {})
    reasoning = turn.get("reasoning", "")

    builder = ActionBuilder()
    if reasoning:
        builder.add_reasoning(reasoning)

    tool = _TOOL_REGISTRY.get(tool_name)
    if tool:
        builder.add_tool_call(tool=tool, params=params, source=f"functions.{tool_name}")
    return builder.build()


def _turn_to_observation(turn: Dict) -> Observation:
    """Convert a trajectory turn dict to an Observation object."""
    obs_text = turn.get("observation", "")
    tool_name = turn.get("tool_name", "")
    builder = ObservationBuilder()
    builder.add_observation(obs_text, source=f"functions.{tool_name}")
    return builder.build()


# ═══════════════════════════════════════════════════════════════════════════════
# Replay trajectory through WorkingMemory
# ═══════════════════════════════════════════════════════════════════════════════

def _replay_trajectory(
    query_text: str,
    turn_history: List[Dict],
    doc_store_data: Dict[str, Dict],
    normalize_ids: bool,
) -> Tuple[
    List[Action],
    List[Observation],
    List[str],
    List[str],
    WorkingMemory,
]:
    """Replay a trajectory through WorkingMemory, producing Actions, Observations,
    WM snapshots, and result summaries at each turn."""

    wm = WorkingMemory(query_text, normalize_ids=normalize_ids)
    actions: List[Action] = []
    observations: List[Observation] = []
    wm_snapshots: List[str] = [wm.to_text()]  # snapshot[0] = initial
    result_summaries: List[str] = []
    tool_types_used: set = set()
    turns_since_curate = 0
    total_curate_calls = 0
    first_search_done = False  # for v8d auto-populate replay

    for turn in turn_history:
        tool_name = turn.get("tool_name", "")
        params = turn.get("params", {})
        obs_text = turn.get("observation", "")

        tool_types_used.add(tool_name)
        pool_size_before = wm.get_pool_size()

        # Replay WM state changes
        if tool_name in ("fan_out_search", "search_corpus", "grep_corpus", "read_document"):
            doc_ids = parse_doc_ids_from_observation(obs_text)
            doc_texts = {}
            for did in doc_ids:
                if did in doc_store_data:
                    snippet = doc_store_data[did].get("snippet", "")
                    doc_texts[did] = snippet
            pool_before = wm.get_pool_size()
            wm.add_to_pool(doc_ids, doc_texts if doc_texts else None)
            num_new = wm.get_pool_size() - pool_before

            if tool_name == "fan_out_search":
                queries = params.get("queries", [])
                q_summary = "; ".join(str(q)[:30] for q in queries[:3])
                wm.add_search_record("fan_out", q_summary, len(doc_ids), num_new=num_new)
            elif tool_name == "search_corpus":
                wm.add_search_record("search", params.get("query", "")[:50], len(doc_ids), num_new=num_new)
            elif tool_name == "grep_corpus":
                wm.add_search_record("grep", params.get("pattern", "")[:50], len(doc_ids), num_new=num_new)
            elif tool_name == "read_document":
                wm.add_search_record("read", params.get("doc_id", ""), len(doc_ids), num_new=num_new)

            # v8d: replay auto-populate on first successful search so the replayed
            # curated set matches what the model actually saw at generation time.
            if (
                V8D_AUTO_POPULATE_FIRST_SEARCH
                and not first_search_done
                and tool_name in ("fan_out_search", "search_corpus")
                and doc_ids
            ):
                auto_populate_from_first_search(wm, doc_ids)
                first_search_done = True

        elif tool_name == "review_docs":
            doc_ids = params.get("doc_ids", [])
            wm.add_search_record("review", ", ".join(doc_ids[:3]), len(doc_ids))

        elif tool_name == "verify":
            # v8d: verify is compute-only, no pool change, but record it in history
            # so replay WM text matches generation-time WM text.
            v_doc_ids = params.get("doc_ids", []) or []
            claim = str(params.get("claim", ""))[:50]
            wm.add_search_record("verify", claim, len(v_doc_ids), num_new=0)

        elif tool_name == "curate":
            add_ids = params.get("add_ids", [])
            remove_ids = params.get("remove_ids", [])
            importance = params.get("importance") if V8D_IMPORTANCE_TAGGING else None
            if not isinstance(add_ids, list):
                add_ids = [add_ids] if add_ids else []
            if not isinstance(remove_ids, list):
                remove_ids = [remove_ids] if remove_ids else []
            wm.curate(add_ids, remove_ids, importance=importance)
            turns_since_curate = 0
            total_curate_calls += 1

        if tool_name == "curate":
            pass  # already handled above
        elif tool_name != "end_search":
            turns_since_curate += 1

        # Build result summary for this turn
        summary = build_result_summary(
            obs_text=obs_text,
            tool_names=[tool_name],
            wm=wm,
            turns_since_curate=turns_since_curate,
            tool_types_used=tool_types_used,
            current_turn=len(actions) + 1,
            pool_size_before=pool_size_before,
        )
        result_summaries.append(summary)

        # Convert to Action/Observation objects
        actions.append(_turn_to_action(turn))
        observations.append(_turn_to_observation(turn))

        wm.advance_turn()
        wm_snapshots.append(wm.to_text())

    return actions, observations, wm_snapshots, result_summaries, wm


# ═══════════════════════════════════════════════════════════════════════════════
# Build training samples
# ═══════════════════════════════════════════════════════════════════════════════

def build_training_samples(
    trajectory: Dict,
    enc,
    max_length: int = 32768,
    min_recall: float = 0.1,
) -> List[tinker.Datum]:
    """Convert one trajectory into training datums (one per turn).

    For each turn t, the context is everything the model would see at that point
    (built by build_context), and the target is the model's action (reasoning + tool call).
    """
    query_text = trajectory["query_text"]
    turn_history = trajectory["turn_history"]
    doc_store_data = trajectory.get("doc_store", {})
    normalize_ids = trajectory.get("normalize_ids", False)
    final_recall = trajectory.get("final_recall", 0.0)

    if final_recall < min_recall:
        return []
    if not turn_history:
        return []

    actions, observations, wm_snapshots, result_summaries, wm = _replay_trajectory(
        query_text, turn_history, doc_store_data, normalize_ids,
    )

    system_prompt = get_system_prompt(query_text)
    datums: List[tinker.Datum] = []

    for t_idx in range(len(actions)):
        n_turns = t_idx  # turns completed before this one

        if n_turns <= RECENT_K:
            wm_text = None
            recent_actions = actions[:t_idx]
            recent_obs = observations[:t_idx]
            recent_summaries = result_summaries[:t_idx]
        else:
            wm_boundary = n_turns - RECENT_K
            wm_text = wm_snapshots[wm_boundary]
            recent_actions = actions[wm_boundary:t_idx]
            recent_obs = observations[wm_boundary:t_idx]
            recent_summaries = result_summaries[wm_boundary:t_idx]

        # Build context (what the model sees)
        context_conv = build_context(
            system_prompt, wm_text, recent_actions, recent_obs, recent_summaries,
        )

        # Build target (what the model should produce)
        target_action = actions[t_idx]
        target_obs = observations[t_idx]
        target_msgs = action_observation_to_messages(target_action, target_obs, compress=False)

        # Full conversation = context + target action (without obs — model produces action)
        action_only_msgs = []
        for msg in target_msgs:
            if msg.author.role == Role.ASSISTANT:
                action_only_msgs.append(msg)
            else:
                break  # stop at first non-assistant (tool result)

        if not action_only_msgs:
            continue

        context_messages = list(context_conv.messages)
        full_messages = context_messages + action_only_msgs

        context_conversation = Conversation(messages=context_messages)
        full_conversation = Conversation(messages=full_messages)

        try:
            context_tokens = enc.render_conversation(context_conversation)
            full_tokens = enc.render_conversation_for_training(full_conversation)
        except Exception as e:
            logger.warning("tokenization_error", turn=t_idx, error=str(e)[:100])
            continue

        n_context = len(context_tokens)
        n_target = len(full_tokens) - n_context

        if n_target <= 0:
            continue

        tokens = list(full_tokens)
        weights = [0] * n_context + [1] * n_target

        if len(tokens) > max_length:
            continue

        # Check for stop token
        stop_tokens = {200002, 200012}
        if not any(t in stop_tokens for t in tokens[-5:]):
            continue

        model_input = tinker.ModelInput.from_ints(tokens)
        weights_tensor = torch.tensor(weights, dtype=torch.float32)
        datum = datum_from_model_input_weights(model_input, weights_tensor, max_length)
        if datum is not None:
            datums.append(datum)

    return datums


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset + Training
# ═══════════════════════════════════════════════════════════════════════════════

class UltraSFTDataset(SupervisedDataset):
    def __init__(self, datums: List[tinker.Datum], batch_size: int = 128):
        self._datums = datums
        self._batch_size = batch_size
        self._shuffled = list(datums)

    def __len__(self):
        return (len(self._shuffled) + self._batch_size - 1) // self._batch_size

    def get_batch(self, index: int) -> List[tinker.Datum]:
        start = index * self._batch_size
        end = min(start + self._batch_size, len(self._shuffled))
        return self._shuffled[start:end]

    def set_epoch(self, seed: int = 0):
        import random
        rng = random.Random(seed)
        self._shuffled = list(self._datums)
        rng.shuffle(self._shuffled)


def load_trajectories(data_dir: str) -> List[Dict]:
    """Load all trajectory JSON files from a directory."""
    trajectories = []
    data_path = Path(data_dir)
    for f in sorted(data_path.glob("*.json")):
        try:
            with open(f) as fh:
                traj = json.load(fh)
            trajectories.append(traj)
        except Exception as e:
            logger.warning("load_error", file=str(f), error=str(e)[:100])
    return trajectories


def main():
    parser = argparse.ArgumentParser(description="SFT training for Ultra v3")
    parser.add_argument("--data-dir", type=str, default="sft_ultra_v3_data")
    parser.add_argument("--log-path", type=str, default="./tmp/sft_ultra_v3")
    parser.add_argument("--model-name", type=str, default="openai/gpt-oss-20b")
    parser.add_argument("--load-checkpoint-path", type=str, default=None)
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=32768)
    parser.add_argument("--min-recall", type=float, default=0.1)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--eval-every", type=int, default=5)
    args = parser.parse_args()

    enc = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

    logger.info("loading_trajectories", data_dir=args.data_dir)
    trajectories = load_trajectories(args.data_dir)
    logger.info("loaded_trajectories", count=len(trajectories))

    all_datums: List[tinker.Datum] = []
    n_traj = len(trajectories)
    import time
    _build_start = time.time()
    _log_every = max(1, n_traj // 20)
    for i, traj in enumerate(trajectories):
        datums = build_training_samples(
            traj, enc, max_length=args.max_length, min_recall=args.min_recall,
        )
        all_datums.extend(datums)
        if (i + 1) % _log_every == 0 or (i + 1) == n_traj:
            pct = 100.0 * (i + 1) / n_traj
            elapsed = time.time() - _build_start
            eta = (elapsed / (i + 1)) * (n_traj - i - 1)
            logger.info(
                "building_datums",
                progress=f"{i+1}/{n_traj}",
                pct=f"{pct:.0f}%",
                datums_so_far=len(all_datums),
                elapsed_s=round(elapsed, 1),
                eta_s=round(eta, 1),
            )

    logger.info(
        "training_data_ready",
        num_datums=len(all_datums),
        num_trajectories=len(trajectories),
        avg_per_traj=len(all_datums) / max(len(trajectories), 1),
        total_build_s=round(time.time() - _build_start, 1),
    )

    if not all_datums:
        logger.error("no_training_data")
        sys.exit(1)

    dataset = UltraSFTDataset(all_datums, batch_size=args.batch_size)

    @chz.chz
    class UltraSFTDatasetBuilder(SupervisedDatasetBuilder):
        def __call__(self) -> Tuple[SupervisedDataset, Optional[SupervisedDataset]]:
            return dataset, None

    config_dict = {
        "log_path": args.log_path,
        "load_checkpoint_path": args.load_checkpoint_path,
        "model_name": args.model_name,
        "dataset_builder": UltraSFTDatasetBuilder(),
        "learning_rate": args.learning_rate,
        "lora_rank": args.lora_rank,
        "num_epochs": args.num_epochs,
        "save_every": args.save_every,
        "eval_every": args.eval_every,
    }
    blueprint = chz.Blueprint(train.Config).apply(config_dict)

    cfg = blueprint.make()
    logger.info("starting_sft_training", config=str(cfg)[:500])

    import asyncio
    asyncio.run(train.main(cfg))


if __name__ == "__main__":
    main()
