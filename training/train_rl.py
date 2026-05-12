
# Allow direct execution from subdirectories while keeping imports package-relative.
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

"""
Clean rewrite of gpt_oss_trainer_rl_ultra.py. Key changes:
  - Uses ultra_core for ALL context assembly (render_context_within_budget).
  - Uses ultra_core for build_result_summary.
  - Uses ultra_core for compute_reward.
  - Bounded consecutive search penalty (cap from CONSEC_SEARCH_PENALTY_CAP).
  - Error detection = format error only (not no-curate episodes).
  - No dead turn penalty code.
  - stride=5 (non-overlapping full coverage).
"""

import asyncio
import copy
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
)

import structlog
import tinker
from openai_harmony import (
    Conversation,
    HarmonyEncoding,
    HarmonyEncodingName,
    Message,
    Role,
    load_harmony_encoding,
)
from tinker_cookbook.rl.types import (
    Env,
    Observation as TinkerObservation,
    StopCondition,
    Action as TinkerAction,
    StepResult,
    EnvGroupBuilder,
    RLDataset,
    RLDatasetBuilder,
    Trajectory as TinkerTrajectory,
    Transition as TinkerTransition,
    TrajectoryGroup,
    Metrics,
)

from harness.agent import TinkerAgentInferenceModel
from datagen.search_dataset import SearchDataset, get_dataset
from harness.rerank import BasetenReranker, ContextualReranker, Reranker
from harness.trajectory import (
    Action,
    Observation,
    ActionBuilder,
    ObservationBuilder,
)
from harness.tools import (
    Tool,
    ToolSet,
    ToolSchema,
    ToolCallMetadata,
    SearchCorpusTool,
    SearchCorpusToolCallMetadata,
    GrepCorpusTool,
    GrepCorpusToolCallMetadata,
    ReadDocumentTool,
    PruneChunksTool,
    UserTextTool,
    SEARCH_CORPUS_SCHEMA,
    GREP_CORPUS_SCHEMA,
    READ_DOCUMENT_SCHEMA,
    MULTI_TOOL_USE_SCHEMA,
)

from harness.ultra_core import (
    WorkingMemory,
    WorkingMemorySnapshot,
    build_result_summary,
    compute_reward,
    get_system_prompt,
    render_context_within_budget,
    parse_doc_ids_from_observation,
    parse_doc_texts_from_observation,
    # Schemas
    FAN_OUT_SEARCH_SCHEMA,
    CURATE_SCHEMA,
    END_SEARCH_SCHEMA,
    REVIEW_DOCS_SCHEMA,
    VERIFY_SCHEMA,
    # v8d helpers
    append_token_marker,
    compress_search_observation,
    auto_populate_from_first_search,
    build_rerank_instruction,
    exec_verify_claim,
    AUTO_POPULATE_TOP_K,
    V8D_AUTO_POPULATE_FIRST_SEARCH,
    V8D_IMPORTANCE_TAGGING,
    V8D_SENTENCE_COMPRESS,
    V8D_TOKEN_BUDGET_MARKER,
    V8D_VERIFY_TOOL,
    V8D_ADAPTIVE_RERANK_INSTRUCTION,
    # Constants
    FAN_OUT_MAX_QUERIES,
    MAX_REVIEW_DOCS,
    MAX_FORMAT_RETRIES,
    CURATE_NUDGE_INTERVAL,
    CURATE_NUDGE_PROMPT,
    FORMAT_RETRY_PROMPT,
    FORMAT_ERROR_PENALTY,
    NO_CURATE_PENALTY,
    MIN_FORMAT_REWARD,
    RECENT_K,
    PROMPT_TOKEN_BUDGET,
    SEARCH_DISPLAY_LIMIT,
    WINDOW_SIZE,
    MAX_WINDOWS,
    CONSEC_SEARCH_PENALTY,
    MAX_CONSEC_BEFORE_PENALTY,
    CONSEC_SEARCH_PENALTY_CAP,
    MAX_TURNS,
)

logger = structlog.get_logger("ultra_rl_v3")

# Save trajectory details for debugging
SAVE_TRAJECTORIES = os.environ.get("SAVE_TRAJECTORIES", "1") == "1"
TRAJECTORY_SAVE_PATH = os.environ.get("TRAJECTORY_SAVE_PATH", None)
ABLATE_VERIFY_UNAVAILABLE = os.environ.get("ABLATE_VERIFY_UNAVAILABLE", "0") == "1"
ABLATE_REVIEW_DOCS_UNAVAILABLE = os.environ.get("ABLATE_REVIEW_DOCS_UNAVAILABLE", "0") == "1"

# Optional per-curate recall-delta shaping.
# v8d default is terminal-reward-centric, so keep this OFF unless explicitly enabled.
DELTA_RECALL_BONUS = float(os.environ.get("DELTA_RECALL_BONUS", "0.0"))
# Reward mode: terminal-only by default (Context-1 style).
# When enabled, per-turn snapshots are not used for window rewards.
USE_TERMINAL_ONLY_REWARD = os.environ.get("USE_TERMINAL_ONLY_REWARD", "1") == "1"
# Rollout mode: full-trajectory by default (Context-1 style).
# Set USE_WINDOW_SLICING=1 to enable legacy window-slicing training.
USE_WINDOW_SLICING = os.environ.get("USE_WINDOW_SLICING", "0") == "1"
# Optional debug gate: restrict training to a fixed set of query IDs.
# Example: FORCE_QUERY_IDS="1029,579,751,605,638"
FORCE_QUERY_IDS_RAW = os.environ.get("FORCE_QUERY_IDS", "").strip()
FORCE_QUERY_IDS = {
    q.strip() for q in FORCE_QUERY_IDS_RAW.split(",") if q.strip()
}


# ═══════════════════════════════════════════════════════════════════════════════
# Tool Stubs (for toolset registration — dispatch handled by env)
# ═══════════════════════════════════════════════════════════════════════════════

class FanOutSearchToolCallMetadata(ToolCallMetadata):
    returned_chunk_ids: List[str]
    queries_executed: int


class FanOutSearchTool(Tool):
    tool_schema: ToolSchema
    def __init__(self):
        super().__init__(tool_schema=FAN_OUT_SEARCH_SCHEMA)
    def __call__(self, params, overrides=None):
        raise NotImplementedError("Handled by env")


class CurateTool(Tool):
    tool_schema: ToolSchema
    def __init__(self):
        super().__init__(tool_schema=CURATE_SCHEMA)
    def __call__(self, params, overrides=None):
        raise NotImplementedError("Handled by env")


class EndSearchTool(Tool):
    tool_schema: ToolSchema
    def __init__(self):
        super().__init__(tool_schema=END_SEARCH_SCHEMA)
    def __call__(self, params, overrides=None):
        return "Search concluded.", None


class ReviewDocsTool(Tool):
    tool_schema: ToolSchema
    def __init__(self):
        super().__init__(tool_schema=REVIEW_DOCS_SCHEMA)
    def __call__(self, params, overrides=None):
        raise NotImplementedError("Handled by env")


class VerifyTool(Tool):
    """v8d: stub for the verify(doc_ids, claim) tool. Dispatched by env."""
    tool_schema: ToolSchema
    def __init__(self):
        super().__init__(tool_schema=VERIFY_SCHEMA)
    def __call__(self, params, overrides=None):
        raise NotImplementedError("Handled by env")


# ═══════════════════════════════════════════════════════════════════════════════
# RL Environment
# ═══════════════════════════════════════════════════════════════════════════════

class SlidingWindowSearchEnv(Env):
    """RL environment with two-tier memory and budget-enforced context rendering."""

    def __init__(
        self,
        toolset: ToolSet,
        search_tool: SearchCorpusTool,
        query_id: str,
        query_text: str,
        dataset: SearchDataset,
        text_token_counter: Optional[Callable[[str], int]] = None,
        max_turns: int = MAX_TURNS,
        rollout_idx: int = 0,
    ):
        self.toolset = toolset
        self.search_tool = search_tool
        self.query_id = query_id
        self.query_text = query_text
        self.dataset = dataset
        self.text_token_counter = text_token_counter
        self.max_turns = max_turns
        self.rollout_idx = rollout_idx

        self.enc = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        self.stop_condition: StopCondition = [200002, 200012]

        self._normalize_ids = (
            getattr(dataset, "evaluation_mode", "document") == "document"
        )

        self.wm = WorkingMemory(query_text, normalize_ids=self._normalize_ids)
        self.system_prompt = get_system_prompt(query_text)

        self._all_actions: List[Action] = []
        self._all_observations: List[Observation] = []
        self._wm_snapshots: List[WorkingMemorySnapshot] = []
        self._result_summaries: List[str] = []
        self._reward_at_turn: List[float] = []

        self._ids_seen: Set[str] = set()
        self._doc_id_to_query: Dict[str, str] = {}

        self._terminal_reward: float = 0.0
        self._terminal_metrics: Metrics = {}
        self._episode_ended: bool = False
        self._current_turn: int = 0
        self._format_retries: int = 0
        self._turns_since_curate: int = 0
        self._total_curate_calls: int = 0
        self._tool_types_used: Set[str] = set()
        self._prev_recall: float = 0.0
        self._curate_recall_deltas: List[float] = []

        # v8d: track token usage for [Context: X/Y] marker, dataset name for
        # adaptive rerank instruction, and first-search auto-populate state.
        self._approx_prompt_tokens: int = 0
        self._first_search_done: bool = False
        self._dataset_name: str = getattr(dataset, "name", "web")
        self._openai_client = None  # lazily acquired when needed
        # Build rerank instruction once per episode (cheap if LLM path disabled)
        self.wm.rerank_instruction = build_rerank_instruction(
            query=query_text,
            dataset_name=self._dataset_name,
            openai_client=None,
            use_llm=False,  # keep cheap for RL — all 512 episodes × 1 call = free tier
        )

        # Pre-compute gold data for fast per-curate recall checks
        self._gold_fact_chunk_sets: Optional[List[Set[str]]] = None
        self._gold_doc_ids: Optional[Set[str]] = None
        self._eval_mode = getattr(dataset, "evaluation_mode", "document")
        try:
            qi = dataset._query_index.get(query_id, {})
            if self._eval_mode == "fact":
                self._gold_fact_chunk_sets = [
                    set(f["chunk_ids"]) for f in qi.get("document_ids", [])
                ]
            else:
                self._gold_doc_ids = set(qi.get("document_ids", []))
        except Exception:
            pass

    # ── Environment Interface ──────────────────────────────────────────────

    async def initial_observation(self) -> Tuple[TinkerObservation, StopCondition]:
        self.wm = WorkingMemory(self.query_text, normalize_ids=self._normalize_ids)
        self._wm_snapshots.append(self.wm.snapshot())

        tokens = render_context_within_budget(
            system_prompt=self.system_prompt,
            wm_text=None,
            recent_actions=[],
            recent_observations=[],
            result_summaries=None,
            enc=self.enc,
        )
        return tinker.ModelInput.from_ints(tokens), self.stop_condition

    async def step(self, action_tokens: TinkerAction) -> StepResult:
        full_toolset = self._build_full_toolset()

        # Parse action tokens
        try:
            action = TinkerAgentInferenceModel.harmony_tinker_tokens_to_action(
                self.enc, action_tokens, full_toolset,
            )
        except Exception as e:
            return self._handle_format_error(str(e))

        if len(action.tools) == 0:
            return self._handle_format_error("Reasoning-only action with no tool calls")

        # Check for episode end
        has_end_search = any(
            t.tool_schema.name == "end_search" for t in action.tools
        )
        has_user_text = any(isinstance(t, UserTextTool) for t in action.tools)

        if has_end_search or has_user_text:
            self._terminal_reward, self._terminal_metrics = (
                self._compute_terminal_reward()
            )
            self._episode_ended = True
            self._save_trajectory()
            logger.info(
                "episode_done",
                reward=round(self._terminal_reward, 4),
                recall=round(self._terminal_metrics.get("recall", 0), 4),
                n_curated=len(self.wm.curated_ids),
                turns=self._current_turn,
                query_id=self.query_id,
            )
            return StepResult(
                reward=self._terminal_reward,
                episode_done=True,
                next_observation=tinker.ModelInput.empty(),
                next_stop_condition=self.stop_condition,
                metrics=self._terminal_metrics,
            )

        # Capture pool size BEFORE tool execution (for novel_count in result summary)
        pool_size_before = self.wm.get_pool_size()

        # Execute tools
        try:
            observation = await asyncio.to_thread(self._execute_tools, action)
        except Exception as e:
            logger.error("tool_exec_error", error=str(e)[:300], qid=self.query_id)
            self._terminal_reward = FORMAT_ERROR_PENALTY
            return StepResult(
                reward=FORMAT_ERROR_PENALTY,
                episode_done=True,
                next_observation=tinker.ModelInput.empty(),
                next_stop_condition=self.stop_condition,
                metrics={"no_error": 0.0, "tool_error": 1.0, "max_turns_reached": 0.0},
            )

        self._format_retries = 0

        # Track curate state
        has_curate = any(t.tool_schema.name == "curate" for t in action.tools)
        if has_curate:
            self._turns_since_curate = 0
            self._total_curate_calls += 1
        else:
            self._turns_since_curate += 1

        for t in action.tools:
            self._tool_types_used.add(t.tool_schema.name)

        self._all_actions.append(action)
        self._all_observations.append(observation)
        self.wm.advance_turn()
        self._current_turn += 1
        self._wm_snapshots.append(self.wm.snapshot())

        # Build result summary
        tool_names = [
            t.tool_schema.name for t in action.tools
            if not isinstance(t, UserTextTool)
        ]
        obs_text = "\n".join(observation.observations) if observation.observations else ""
        summary = build_result_summary(
            obs_text=obs_text,
            tool_names=tool_names,
            wm=self.wm,
            turns_since_curate=self._turns_since_curate,
            tool_types_used=self._tool_types_used,
            current_turn=self._current_turn,
            pool_size_before=pool_size_before,
        )
        self._result_summaries.append(summary)

        # Optional per-turn reward snapshots (legacy sliding-window shaping).
        # v8d default uses terminal-only reward.
        if not USE_TERMINAL_ONLY_REWARD:
            turn_reward = self._compute_reward_at_current_state()

            # Bounded per-turn shaping: consecutive search penalty
            if (not has_curate
                    and self._turns_since_curate > MAX_CONSEC_BEFORE_PENALTY
                    and self.wm.get_pool_size() > 0):
                penalty = min(
                    CONSEC_SEARCH_PENALTY * (self._turns_since_curate - MAX_CONSEC_BEFORE_PENALTY),
                    CONSEC_SEARCH_PENALTY_CAP,
                )
                turn_reward -= penalty

            # Optional per-curate recall-delta shaping (disabled by default for v8d).
            if has_curate and len(self.wm.curated_ids) > 0:
                cur_recall = self._fast_recall(self.wm.curated_ids)
                delta = cur_recall - self._prev_recall
                self._curate_recall_deltas.append(delta)
                if DELTA_RECALL_BONUS != 0.0:
                    turn_reward += DELTA_RECALL_BONUS * delta
                self._prev_recall = cur_recall
                if delta > 0:
                    logger.info(
                        "curate_delta_positive", qid=self.query_id,
                        delta=round(delta, 4), recall=round(cur_recall, 4),
                        turn=self._current_turn,
                    )

            self._reward_at_turn.append(turn_reward)

        # Max turns check
        if self._current_turn >= self.max_turns:
            self._terminal_reward, self._terminal_metrics = (
                self._compute_terminal_reward()
            )
            self._terminal_metrics["max_turns_reached"] = 1.0
            self._episode_ended = True
            self._save_trajectory()
            return StepResult(
                reward=self._terminal_reward,
                episode_done=True,
                next_observation=tinker.ModelInput.empty(),
                next_stop_condition=self.stop_condition,
                metrics=self._terminal_metrics,
            )

        # Render context for next turn (budget-enforced)
        try:
            tokens = self._render_next_context()
        except Exception as e:
            logger.error("render_error", error=str(e)[:300], qid=self.query_id)
            self._terminal_reward = 0.0
            return StepResult(
                reward=0.0,
                episode_done=True,
                next_observation=tinker.ModelInput.empty(),
                next_stop_condition=self.stop_condition,
                metrics={"no_error": 0.0, "max_turns_reached": 0.0},
            )

        return StepResult(
            reward=0.0,
            episode_done=False,
            next_observation=tinker.ModelInput.from_ints(tokens),
            next_stop_condition=self.stop_condition,
        )

    # ── Context Rendering (single pathway via ultra_core) ──────────────────

    def _render_next_context(self) -> List[int]:
        """Render context for the next turn using render_context_within_budget."""
        n_turns = len(self._all_actions)

        if n_turns <= RECENT_K:
            wm_text = None
            recent_actions = self._all_actions
            recent_obs = self._all_observations
            recent_summaries = self._result_summaries
        else:
            wm_boundary = n_turns - RECENT_K
            wm_text = self._wm_snapshots[wm_boundary].text
            recent_actions = self._all_actions[-RECENT_K:]
            recent_obs = self._all_observations[-RECENT_K:]
            recent_summaries = self._result_summaries[-RECENT_K:]

        nudge = None
        if (self._turns_since_curate >= CURATE_NUDGE_INTERVAL
                and self.wm.get_pool_size() > 0):
            nudge = CURATE_NUDGE_PROMPT

        tokens = render_context_within_budget(
            system_prompt=self.system_prompt,
            wm_text=wm_text,
            recent_actions=recent_actions,
            recent_observations=recent_obs,
            result_summaries=recent_summaries,
            enc=self.enc,
            nudge_prompt=nudge,
        )
        # v8d: stash size so the next tool output can append an accurate marker.
        self._approx_prompt_tokens = len(tokens)
        return tokens

    def _render_retry_context(self) -> List[int]:
        """Re-render current context with retry prompt appended."""
        n_turns = len(self._all_actions)

        if n_turns <= RECENT_K:
            wm_text = None
            recent_actions = self._all_actions
            recent_obs = self._all_observations
            recent_summaries = self._result_summaries
        else:
            wm_boundary = n_turns - RECENT_K
            wm_text = self._wm_snapshots[wm_boundary].text
            recent_actions = self._all_actions[-RECENT_K:]
            recent_obs = self._all_observations[-RECENT_K:]
            recent_summaries = self._result_summaries[-RECENT_K:]

        return render_context_within_budget(
            system_prompt=self.system_prompt,
            wm_text=wm_text,
            recent_actions=recent_actions,
            recent_observations=recent_obs,
            result_summaries=recent_summaries,
            enc=self.enc,
            retry_prompt=FORMAT_RETRY_PROMPT,
        )

    # ── Format Error Handling ──────────────────────────────────────────────

    def _handle_format_error(self, error_msg: str) -> StepResult:
        self._format_retries += 1
        if self._format_retries <= MAX_FORMAT_RETRIES:
            logger.warning(
                "format_retry",
                error=error_msg[:200],
                retry=self._format_retries,
                qid=self.query_id,
            )
            try:
                tokens = self._render_retry_context()
            except Exception:
                tokens = render_context_within_budget(
                    self.system_prompt, None, [], [], None,
                    self.enc, retry_prompt=FORMAT_RETRY_PROMPT,
                )
            return StepResult(
                reward=0.0,
                episode_done=False,
                next_observation=tinker.ModelInput.from_ints(tokens),
                next_stop_condition=self.stop_condition,
                metrics={"format_retry": float(self._format_retries)},
            )
        else:
            logger.error(
                "format_error_final",
                error=error_msg[:300],
                retries=self._format_retries,
                qid=self.query_id,
            )
            self._terminal_reward = FORMAT_ERROR_PENALTY
            return StepResult(
                reward=FORMAT_ERROR_PENALTY,
                episode_done=True,
                next_observation=tinker.ModelInput.empty(),
                next_stop_condition=self.stop_condition,
                metrics={
                    "no_error": 0.0,
                    "format_error": 1.0,
                    "max_turns_reached": 0.0,
                },
            )

    # ── Tool Dispatch ──────────────────────────────────────────────────────

    def _build_full_toolset(self) -> ToolSet:
        ts = ToolSet(name="ultra_v3_toolset")
        for name, tool in self.toolset.tools.items():
            ts.tools[name] = tool
        ts.tools["fan_out_search"] = FanOutSearchTool()
        ts.tools["curate"] = CurateTool()
        ts.tools["end_search"] = EndSearchTool()
        ts.tools["review_docs"] = ReviewDocsTool()
        if V8D_VERIFY_TOOL:
            ts.tools["verify"] = VerifyTool()
        return ts

    def _execute_tools(self, action: Action) -> Observation:
        obs_builder = ObservationBuilder()

        for tool, params, source in zip(action.tools, action.params, action.sources):
            if isinstance(tool, UserTextTool):
                obs_builder.add_observation("", source=source, tool_metadata=None)
                continue

            name = tool.tool_schema.name
            logger.info("tool_call", tool=name, qid=self.query_id, turn=self._current_turn)
            try:
                if name == "fan_out_search":
                    output, meta = self._exec_fan_out_search(params)
                    obs_builder.add_observation(output, source=source, tool_metadata=meta)
                elif name == "search_corpus":
                    output, meta = self._exec_search(params)
                    obs_builder.add_observation(output, source=source, tool_metadata=meta)
                elif name == "grep_corpus":
                    output, meta = self._exec_grep(params)
                    obs_builder.add_observation(output, source=source, tool_metadata=meta)
                elif name == "read_document":
                    output, meta = self._exec_read_doc(params)
                    obs_builder.add_observation(output, source=source, tool_metadata=meta)
                elif name == "curate":
                    output = self._exec_curate(params)
                    obs_builder.add_observation(output, source=source, tool_metadata=None)
                elif name == "review_docs":
                    output = self._exec_review_docs(params)
                    obs_builder.add_observation(output, source=source, tool_metadata=None)
                elif name == "verify" and V8D_VERIFY_TOOL:
                    output = self._exec_verify(params)
                    obs_builder.add_observation(output, source=source, tool_metadata=None)
                elif name == "end_search":
                    obs_builder.add_observation("Search concluded.", source=source, tool_metadata=None)
                elif name == "prune_chunks":
                    obs_builder.add_observation(
                        "Context is managed via working memory. No pruning needed.",
                        source=source, tool_metadata=None,
                    )
                else:
                    obs_builder.add_observation(
                        f"Unknown tool: {name}", source=source, tool_metadata=None,
                    )
            except Exception as e:
                logger.warning("tool_error", tool=name, error=str(e)[:200], qid=self.query_id)
                obs_builder.add_observation(
                    f"Error executing {name}: {str(e)[:200]}",
                    source=source, tool_metadata=None,
                )

        return obs_builder.build()

    def _maybe_wrap_search_output(
        self,
        output: str,
        query_for_compress: str,
        first_search_ranked_ids: Optional[List[str]] = None,
    ) -> str:
        """v8d wrapper: BM25 compress + auto-populate + token marker."""
        # 1. Sentence-level compression (no-op unless flag on)
        if V8D_SENTENCE_COMPRESS and query_for_compress:
            output = compress_search_observation(query_for_compress, output)

        # 2. Auto-populate the curated set from the first search's top hits
        if (
            V8D_AUTO_POPULATE_FIRST_SEARCH
            and not self._first_search_done
            and first_search_ranked_ids
        ):
            added = auto_populate_from_first_search(
                self.wm, first_search_ranked_ids, top_k=AUTO_POPULATE_TOP_K,
            )
            self._first_search_done = True
            if added > 0:
                output = (
                    output
                    + f"\n\n[AUTO-POPULATED] Top {added} docs from this search have been "
                    "added to your curated set at 'fair' importance. Use `curate` with "
                    "`importance` to promote/demote and `remove_ids` to drop irrelevant ones."
                )

        # 3. Token budget marker (no-op unless flag on)
        if V8D_TOKEN_BUDGET_MARKER and self.text_token_counter is not None:
            try:
                used = self._approx_prompt_tokens + self.text_token_counter(output)
                output = append_token_marker(output, used)
            except Exception:
                pass

        return output

    def _exec_search(self, params: Dict) -> Tuple[str, Optional[ToolCallMetadata]]:
        query = params.get("query") or params.get("q", "")
        pool_before = self.wm.get_pool_size()
        # v8d: pipe per-episode rerank instruction through to the search tool.
        overrides: Dict[str, Any] = {"ignore_ids": list(self._ids_seen)}
        if V8D_ADAPTIVE_RERANK_INSTRUCTION and self.wm.rerank_instruction:
            overrides["rerank_instruction"] = self.wm.rerank_instruction
        output, meta = self.search_tool(params, overrides)
        ranked_ids: List[str] = []
        if meta and isinstance(meta, SearchCorpusToolCallMetadata):
            ranked_ids = list(meta.returned_chunk_ids)
            self._ids_seen.update(meta.returned_chunk_ids)
            doc_texts = parse_doc_texts_from_observation(output)
            self.wm.add_to_pool(meta.returned_chunk_ids, doc_texts)
            for cid in meta.returned_chunk_ids:
                doc_id = cid.split("_")[0] if "_" in cid else cid
                self._doc_id_to_query.setdefault(doc_id, str(query))
            num_new = self.wm.get_pool_size() - pool_before
            self.wm.add_search_record(
                "search", str(query)[:60], len(meta.returned_chunk_ids),
                num_new=num_new,
            )
        output = self._maybe_wrap_search_output(
            output, query_for_compress=str(query),
            first_search_ranked_ids=ranked_ids,
        )
        return output, meta

    def _exec_fan_out_search(self, params: Dict) -> Tuple[str, Optional[FanOutSearchToolCallMetadata]]:
        queries = params.get("queries", [])
        if not isinstance(queries, list) or not queries:
            return "No queries provided.", FanOutSearchToolCallMetadata(
                returned_chunk_ids=[], queries_executed=0,
            )

        queries = queries[:FAN_OUT_MAX_QUERIES]
        all_results: List[str] = []
        all_chunk_ids: List[str] = []
        pool_before = self.wm.get_pool_size()

        for q in queries:
            if not isinstance(q, str) or not q.strip():
                continue
            try:
                overrides: Dict[str, Any] = {"ignore_ids": list(self._ids_seen)}
                if V8D_ADAPTIVE_RERANK_INSTRUCTION and self.wm.rerank_instruction:
                    overrides["rerank_instruction"] = self.wm.rerank_instruction
                output, meta = self.search_tool({"query": q}, overrides)
                all_results.append(output)
                if meta and isinstance(meta, SearchCorpusToolCallMetadata):
                    self._ids_seen.update(meta.returned_chunk_ids)
                    doc_texts = parse_doc_texts_from_observation(output)
                    self.wm.add_to_pool(meta.returned_chunk_ids, doc_texts)
                    all_chunk_ids.extend(meta.returned_chunk_ids)
                    for cid in meta.returned_chunk_ids:
                        doc_id = cid.split("_")[0] if "_" in cid else cid
                        self._doc_id_to_query.setdefault(doc_id, str(q))
            except Exception as e:
                logger.warning("fan_out_error", query=str(q)[:100], error=str(e)[:200])
                all_results.append("No results.")

        q_summary = "; ".join(str(q)[:30] for q in queries[:3])
        num_new = self.wm.get_pool_size() - pool_before
        self.wm.add_search_record(
            "fan_out", q_summary, len(all_chunk_ids), num_new=num_new,
        )
        combined = "\n".join(all_results) if all_results else "No results found."
        # v8d: compress (using concatenated query string), auto-populate, token marker
        concat_query = " ".join(str(q) for q in queries if isinstance(q, str))
        combined = self._maybe_wrap_search_output(
            combined,
            query_for_compress=concat_query,
            first_search_ranked_ids=all_chunk_ids,
        )
        return combined, FanOutSearchToolCallMetadata(
            returned_chunk_ids=all_chunk_ids, queries_executed=len(queries),
        )

    def _exec_grep(self, params: Dict) -> Tuple[str, Optional[ToolCallMetadata]]:
        grep_tool = self.toolset.get_tool("grep_corpus")
        if grep_tool is None:
            return "grep_corpus not available.", None
        pool_before = self.wm.get_pool_size()
        output, meta = grep_tool(params)
        if meta and isinstance(meta, GrepCorpusToolCallMetadata):
            doc_texts = parse_doc_texts_from_observation(output)
            self.wm.add_to_pool(meta.returned_chunk_ids, doc_texts)
            num_new = self.wm.get_pool_size() - pool_before
            self.wm.add_search_record(
                "grep", str(params.get("pattern", ""))[:60],
                len(meta.returned_chunk_ids), num_new=num_new,
            )
        # v8d: grep results can still benefit from sentence-level compression and token marker
        output = self._maybe_wrap_search_output(
            output, query_for_compress=str(params.get("pattern", "")),
            first_search_ranked_ids=None,
        )
        return output, meta

    def _exec_read_doc(self, params: Dict) -> Tuple[str, Optional[ToolCallMetadata]]:
        read_tool = self.toolset.get_tool("read_document")
        if read_tool is None:
            return "read_document not available.", None
        doc_id = params.get("doc_id") or params.get("id", "")
        if self._normalize_ids and "_" in doc_id:
            doc_id = doc_id.split("_")[0]
        overrides = {}
        if doc_id in self._doc_id_to_query:
            overrides["query"] = self._doc_id_to_query[doc_id]
        pool_before = self.wm.get_pool_size()
        output, meta = read_tool(params, overrides or None)
        doc_texts = parse_doc_texts_from_observation(output)
        if doc_texts:
            self.wm.add_to_pool(list(doc_texts.keys()), doc_texts)
        num_new = self.wm.get_pool_size() - pool_before
        self.wm.add_search_record(
            "read", str(doc_id)[:30],
            len(doc_texts) if doc_texts else 1, num_new=num_new,
        )
        # v8d: read_document returns full text — compression is too aggressive here,
        # but still append token marker.
        if V8D_TOKEN_BUDGET_MARKER and self.text_token_counter is not None:
            try:
                used = self._approx_prompt_tokens + self.text_token_counter(output)
                output = append_token_marker(output, used)
            except Exception:
                pass
        return output, meta

    def _exec_curate(self, params: Dict) -> str:
        add_ids = params.get("add_ids", [])
        remove_ids = params.get("remove_ids", [])
        if not isinstance(add_ids, list):
            add_ids = [str(add_ids)] if add_ids else []
        if not isinstance(remove_ids, list):
            remove_ids = [str(remove_ids)] if remove_ids else []

        importance: Optional[Dict[str, str]] = None
        if V8D_IMPORTANCE_TAGGING:
            raw = params.get("importance")
            if isinstance(raw, dict):
                importance = {str(k): str(v) for k, v in raw.items()}

        return self.wm.curate(add_ids, remove_ids, importance=importance)

    def _exec_verify(self, params: Dict) -> str:
        """v8d: verify claim against specific docs via LLM. No corpus call."""
        if ABLATE_VERIFY_UNAVAILABLE:
            self.wm.add_search_record("verify", "unavailable", 0, num_new=0)
            return "verify: unavailable in this ablation."

        doc_ids = params.get("doc_ids", [])
        claim = str(params.get("claim", "")).strip()
        if not isinstance(doc_ids, list):
            doc_ids = [str(doc_ids)] if doc_ids else []
        doc_ids = [str(d).strip() for d in doc_ids if d][:5]
        if not doc_ids or not claim:
            return "verify: doc_ids or claim missing."

        # Resolve full text from WM's doc_store (verify does NOT re-query the corpus).
        doc_texts: Dict[str, str] = {}
        for did in doc_ids:
            norm = self.wm._normalize_id(did)
            store = self.wm.doc_store.get(norm, {})
            txt = store.get("full_text") or store.get("snippet") or ""
            if txt:
                doc_texts[norm] = txt

        if self._openai_client is None:
            try:
                from harness.config import get_config
                self._openai_client = get_config().get_openai_client()
            except Exception as e:
                return f"verify: openai client unavailable ({str(e)[:80]})"

        self.wm.add_search_record(
            "verify", claim[:50], len(doc_ids), num_new=0,
        )
        return exec_verify_claim(self._openai_client, doc_texts, claim)

    def _exec_review_docs(self, params: Dict) -> str:
        if ABLATE_REVIEW_DOCS_UNAVAILABLE:
            self.wm.add_search_record("review", "unavailable", 0)
            return "review_docs: unavailable in this ablation."

        doc_ids = params.get("doc_ids", [])
        if not isinstance(doc_ids, list):
            doc_ids = [str(doc_ids)] if doc_ids else []
        doc_ids = [str(x).strip() for x in doc_ids if x][:MAX_REVIEW_DOCS]
        if not doc_ids:
            return "No doc_ids provided."
        result = self.wm.review_docs(doc_ids)
        self.wm.add_search_record("review", ", ".join(doc_ids[:3]), len(doc_ids))
        return result

    # ── Reward ─────────────────────────────────────────────────────────────

    def _fast_recall(self, curated_ids: List[str]) -> float:
        """Lightweight recall using pre-computed gold data. Avoids repeated
        _query_index lookups and set constructions on the hot path."""
        if not curated_ids:
            return 0.0
        if self._eval_mode == "fact" and self._gold_fact_chunk_sets is not None:
            if not self._gold_fact_chunk_sets:
                return 0.0
            curated_set = set(curated_ids)
            found = sum(
                1 for gset in self._gold_fact_chunk_sets
                if gset & curated_set
            )
            return found / len(self._gold_fact_chunk_sets)
        if self._gold_doc_ids is not None:
            if not self._gold_doc_ids:
                return 0.0
            # curated_ids are already normalized to doc IDs by WorkingMemory,
            # so no further rsplit needed (avoids double-stripping doc IDs
            # that contain underscores, e.g. web datasets).
            found = len(set(curated_ids) & self._gold_doc_ids)
            return found / len(self._gold_doc_ids)
        return self.dataset.evaluate_results_recall(self.query_id, curated_ids)

    def _evaluate_and_compute_reward(self, is_terminal: bool) -> Tuple[float, Metrics]:
        """Evaluate recall/precision via dataset, then delegate to compute_reward."""
        curated = self.wm.curated_ids
        pool = self.wm.pool_ids

        if curated:
            recall = self.dataset.evaluate_results_recall(self.query_id, curated)
            precision = self.dataset.evaluate_results_precision(self.query_id, curated)
            fa_recall = self.dataset.evaluate_results_final_answer_recall(self.query_id, curated)
        else:
            recall = precision = fa_recall = 0.0

        traj_recall = (
            self.dataset.evaluate_results_recall(self.query_id, pool)
            if pool else 0.0
        )
        traj_fa_recall = (
            self.dataset.evaluate_results_final_answer_recall(self.query_id, pool)
            if pool else 0.0
        )

        return compute_reward(
            recall=recall,
            precision=precision,
            final_answer_recall=fa_recall,
            trajectory_recall=traj_recall,
            n_curated=len(curated),
            turn=self._current_turn,
            total_curate_calls=self._total_curate_calls,
            n_unique_tools=len(self._tool_types_used - {"end_search"}),
            is_terminal=is_terminal,
            trajectory_fa_recall=traj_fa_recall,
        )

    def _compute_reward_at_current_state(self) -> float:
        reward, _ = self._evaluate_and_compute_reward(is_terminal=False)
        return reward

    def _compute_terminal_reward(self) -> Tuple[float, Metrics]:
        return self._evaluate_and_compute_reward(is_terminal=True)

    # ── Trajectory Saving ──────────────────────────────────────────────────

    def _save_trajectory(self) -> None:
        if not SAVE_TRAJECTORIES:
            return
        try:
            save_dir = TRAJECTORY_SAVE_PATH or os.environ.get("LOG_PATH", "./tmp/rl_ultra_v3")
            save_dir = os.path.join(save_dir, "trajectories")
            os.makedirs(save_dir, exist_ok=True)

            record = {
                "query_id": self.query_id,
                "dataset": self.dataset.name,
                "normalize_ids": self._normalize_ids,
                "reward": self._terminal_reward,
                "metrics": {
                    k: v for k, v in self._terminal_metrics.items()
                    if isinstance(v, (int, float, str, bool))
                },
                "turns": self._current_turn,
                "curated_ids": self.wm.curated_ids,
                # Persist v8d per-doc tags so downstream analysis can filter
                # to high-confidence subsets (e.g., very_high/high only).
                "curated_importance": dict(self.wm.curated_importance),
                "pool_ids": self.wm.pool_ids[:50],
                "pool_size": len(self.wm.pool_ids),
                "reward_at_turn": [round(r, 4) for r in self._reward_at_turn],
                "search_history": self.wm.search_history,
                "curate_recall_deltas": [round(d, 4) for d in self._curate_recall_deltas],
            }
            save_file = os.path.join(save_dir, "episodes.jsonl")
            with open(save_file, "a") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            logger.warning("save_error", error=str(e)[:200])


# ═══════════════════════════════════════════════════════════════════════════════
# Window Slicing
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class WindowSlice:
    trajectory: TinkerTrajectory
    reward: float
    stage: int  # start_turn // window_size — used for normalization bucketing


def slice_trajectory_into_windows(
    trajectory: TinkerTrajectory,
    terminal_reward: float,
    reward_at_turn: Optional[List[float]] = None,
    window_size: int = WINDOW_SIZE,
    max_windows: int = MAX_WINDOWS,
) -> List[WindowSlice]:
    """Slice a full-episode trajectory into up to *max_windows* non-overlapping
    windows, evenly distributed across the episode.  The last window always
    covers the final turn so the terminal reward signal is never lost.

    Each window gets the reward at its END turn (curated set quality at that
    point).  Per-transition rewards are zeroed to avoid double-counting with
    final_rewards_G.

    Returns WindowSlice objects carrying a ``stage`` field (start_turn //
    window_size) so callers can normalize rewards across rollouts by episode
    stage rather than ordinal window index.
    """
    transitions = trajectory.transitions
    n = len(transitions)

    if n == 0:
        return []

    def _zero_rewards(ts):
        return [
            TinkerTransition(
                ob=t.ob, ac=t.ac, reward=0.0,
                episode_done=t.episode_done, metrics=t.metrics, logs=t.logs,
            )
            for t in ts
        ]

    def _make_window(start: int, end: int) -> WindowSlice:
        window_transitions = _zero_rewards(transitions[start:end])
        final_ob = transitions[end].ob if end < n else trajectory.final_ob
        if reward_at_turn is not None and end - 1 < len(reward_at_turn):
            window_reward = reward_at_turn[end - 1]
        else:
            window_reward = terminal_reward
        return WindowSlice(
            trajectory=TinkerTrajectory(transitions=window_transitions, final_ob=final_ob),
            reward=window_reward,
            stage=start // window_size,
        )

    if n <= window_size:
        return [_make_window(0, n)]

    # Build all non-overlapping windows first
    all_starts: List[int] = []
    start = 0
    while start < n:
        all_starts.append(start)
        start += window_size
        if start + 1 >= n:
            break

    if len(all_starts) <= max_windows:
        return [
            _make_window(s, min(s + window_size, n)) for s in all_starts
        ]

    # Too many windows — subsample to max_windows.
    # Always keep the last window; pick (max_windows - 1) evenly from the rest.
    last_idx = len(all_starts) - 1
    rest_indices = list(range(last_idx))
    num_pick = max_windows - 1

    if num_pick <= 0:
        picked = []
    elif num_pick >= len(rest_indices):
        picked = rest_indices
    else:
        picked = [
            rest_indices[round(i * (len(rest_indices) - 1) / (num_pick - 1))]
            for i in range(num_pick)
        ]
        seen: set = set()
        unique: List[int] = []
        for idx in picked:
            if idx not in seen:
                seen.add(idx)
                unique.append(idx)
        picked = unique

    selected_indices = picked + [last_idx]
    return [
        _make_window(all_starts[i], min(all_starts[i] + window_size, n))
        for i in selected_indices
    ]


# ═══════════════════════════════════════════════════════════════════════════════
# Group Builder + Rollout
# ═══════════════════════════════════════════════════════════════════════════════

class SlidingWindowGroupBuilder(EnvGroupBuilder):
    def __init__(
        self,
        query_id: str,
        query_text: str,
        dataset: SearchDataset,
        toolset: ToolSet,
        search_tool: SearchCorpusTool,
        group_size: int,
        text_token_counter: Optional[Callable[[str], int]] = None,
    ):
        self.query_id = query_id
        self.query_text = query_text
        self.dataset = dataset
        self.toolset = toolset
        self.search_tool = search_tool
        self.group_size = group_size
        self.text_token_counter = text_token_counter
        self._envs: List[SlidingWindowSearchEnv] = []

    async def make_envs(self) -> Sequence[SlidingWindowSearchEnv]:
        self._envs = [
            SlidingWindowSearchEnv(
                toolset=self.toolset,
                search_tool=self.search_tool,
                query_id=self.query_id,
                query_text=self.query_text,
                dataset=self.dataset,
                text_token_counter=self.text_token_counter,
                rollout_idx=i,
            )
            for i in range(self.group_size)
        ]
        return self._envs

    async def compute_group_rewards(
        self, trajectory_group: List[TinkerTrajectory], env_group: Sequence[Env],
    ) -> List[Tuple[float, Metrics]]:
        return [(0.0, {}) for _ in trajectory_group]

    def logging_tags(self) -> List[str]:
        return [self.dataset.name, "search"]


async def do_group_rollout_with_windows(
    env_group_builder: SlidingWindowGroupBuilder,
    policy,
) -> TrajectoryGroup:
    """Run full rollouts, then slice windows and position-normalize rewards."""
    from tinker_cookbook.rl.rollouts import do_single_rollout

    envs = await env_group_builder.make_envs()
    full_trajectories = await asyncio.gather(
        *[do_single_rollout(policy, env) for env in envs]
    )

    terminal_rewards = [env._terminal_reward for env in envs]
    terminal_metrics = [env._terminal_metrics for env in envs]
    per_turn_rewards = [env._reward_at_turn for env in envs]

    # Error = format error only (negative terminal reward from parse failure).
    # No-curate episodes (terminal reward = -0.2) are NOT errors — they're
    # valid episodes that should participate in position normalization.
    is_format_error = [env._terminal_reward == FORMAT_ERROR_PENALTY for env in envs]

    # Slice into windows
    all_rollout_windows: List[List[WindowSlice]] = []
    for traj, reward, turn_rewards in zip(
        full_trajectories, terminal_rewards, per_turn_rewards,
    ):
        windows = slice_trajectory_into_windows(
            traj,
            reward,
            reward_at_turn=None if USE_TERMINAL_ONLY_REWARD else turn_rewards,
        )
        all_rollout_windows.append(windows)

    max_windows = max(len(w) for w in all_rollout_windows) if all_rollout_windows else 0

    # Normalize rewards using two kinds of groups:
    #   "final"  — last window of every rollout (the terminal‐reward signal,
    #              always compared across rollouts regardless of episode length)
    #   stage N  — non‐final windows bucketed by episode stage
    #              (start_turn // window_size) so windows covering similar
    #              turns are compared even when subsampling differs.
    FINAL_KEY = -1  # sentinel for the "final" normalization group
    group_rewards: Dict[int, List[float]] = {}
    for rollout_idx, windows in enumerate(all_rollout_windows):
        if is_format_error[rollout_idx]:
            continue
        for i, ws in enumerate(windows):
            key = FINAL_KEY if i == len(windows) - 1 else ws.stage
            group_rewards.setdefault(key, []).append(ws.reward)

    group_means: Dict[int, float] = {
        k: sum(rs) / len(rs) for k, rs in group_rewards.items()
    } if group_rewards else {}

    all_means = list(group_means.values())
    global_mean = sum(all_means) / len(all_means) if all_means else 0.0

    all_window_trajectories: List[TinkerTrajectory] = []
    all_window_rewards: List[float] = []
    all_window_metrics: List[Metrics] = []

    for rollout_idx, windows in enumerate(all_rollout_windows):
        for i, ws in enumerate(windows):
            if is_format_error[rollout_idx]:
                normalized = ws.reward
            else:
                key = FINAL_KEY if i == len(windows) - 1 else ws.stage
                if key in group_means:
                    normalized = ws.reward - group_means[key] + global_mean
                else:
                    normalized = ws.reward
            all_window_trajectories.append(ws.trajectory)
            all_window_rewards.append(normalized)
            all_window_metrics.append(terminal_metrics[rollout_idx])

    stage_log = {
        ("final" if k == FINAL_KEY else f"stg{k}"): round(m, 3)
        for k, m in sorted(group_means.items())
    }
    logger.info(
        "windows_created",
        query_id=env_group_builder.query_id,
        reward_mode="terminal_only" if USE_TERMINAL_ONLY_REWARD else "per_turn_window",
        rollouts=len(envs),
        errors=sum(is_format_error),
        max_windows=max_windows,
        windows=[len(w) for w in all_rollout_windows],
        group_means=stage_log,
        terminal=[round(r, 3) for r in terminal_rewards],
    )

    return TrajectoryGroup(
        trajectories_G=all_window_trajectories,
        final_rewards_G=all_window_rewards,
        metrics_G=all_window_metrics,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class DatasetToolsetPair:
    dataset: SearchDataset
    toolset: ToolSet
    search_tool: SearchCorpusTool
    max_train_queries: Optional[int] = None

    @property
    def name(self) -> str:
        return self.dataset.name


class SlidingWindowRLDataset(RLDataset):
    def __init__(
        self,
        dataset_pairs: List[DatasetToolsetPair],
        batch_size: int,
        group_size: int,
        text_token_counter: Optional[Callable[[str], int]] = None,
        seed: int = 0,
        epochs: int = 1,
        query_split: str = "train",
    ):
        self.dataset_pairs = dataset_pairs
        self.batch_size = batch_size
        self.group_size = group_size
        self.text_token_counter = text_token_counter
        self.seed = seed
        self.epochs = epochs
        self.query_split = query_split

        import random as rng_module
        self._query_to_pair: Dict[str, DatasetToolsetPair] = {}
        self._base_query_ids: List[str] = []

        for pair in dataset_pairs:
            query_ids = pair.dataset.get_all_query_ids(split=self.query_split)

            # Optional debug mode for exact query-set parity checks.
            if FORCE_QUERY_IDS:
                before = len(query_ids)
                query_ids = [qid for qid in query_ids if str(qid) in FORCE_QUERY_IDS]
                logger.info(
                    "force_query_filter",
                    dataset=pair.name,
                    before=before,
                    after=len(query_ids),
                    requested=len(FORCE_QUERY_IDS),
                )

            if pair.max_train_queries and len(query_ids) > pair.max_train_queries:
                rng = rng_module.Random(seed)
                query_ids = rng.sample(query_ids, pair.max_train_queries)
            for qid in query_ids:
                unique_qid = f"{pair.name}::{qid}"
                self._query_to_pair[unique_qid] = pair
                self._base_query_ids.append(unique_qid)

        if not self._base_query_ids:
            raise ValueError(
                "No training queries selected. Check TRAIN_DATASETS / FORCE_QUERY_IDS."
            )

        self._batches_per_epoch = max(1, len(self._base_query_ids) // self.batch_size)
        self._epoch_query_ids: List[List[str]] = []
        for epoch in range(self.epochs):
            epoch_ids = self._base_query_ids.copy()
            rng = rng_module.Random(seed + epoch)
            rng.shuffle(epoch_ids)
            self._epoch_query_ids.append(epoch_ids)

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        epoch = index // self._batches_per_epoch
        batch_in_epoch = index % self._batches_per_epoch
        query_ids = self._epoch_query_ids[epoch]
        start = batch_in_epoch * self.batch_size
        end = min(start + self.batch_size, len(query_ids))

        builders: List[EnvGroupBuilder] = []
        for unique_qid in query_ids[start:end]:
            pair = self._query_to_pair[unique_qid]
            _, original_qid = unique_qid.split("::", 1)
            _, query_text = pair.dataset.get_query_by_id(original_qid)
            builder = SlidingWindowGroupBuilder(
                query_id=original_qid,
                query_text=query_text,
                dataset=pair.dataset,
                toolset=pair.toolset,
                search_tool=pair.search_tool,
                group_size=self.group_size,
                text_token_counter=self.text_token_counter,
            )
            builders.append(builder)
        return builders

    def __len__(self) -> int:
        return self._batches_per_epoch * self.epochs


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import chz
    import tiktoken
    from tinker_cookbook.rl import train
    from tinker_cookbook.rl.rollouts import do_group_rollout
    from harness.config import get_config

    DATASETS = os.environ.get("TRAIN_DATASETS", "browsecompplus,sec").split(",")
    SIMPLE_DATASET_MAX_QUERIES = int(os.environ.get("SIMPLE_DATASET_MAX_QUERIES", "150"))
    # Backward compatible split handling:
    # - RL_QUERY_SPLIT controls which query IDs are sampled for rollouts.
    # - RL_COLLECTION_SPLIT controls which Chroma collections back tools.
    # - RL_DATA_SPLIT remains accepted as a legacy alias for RL_QUERY_SPLIT.
    RL_QUERY_SPLIT = (
        os.environ.get("RL_QUERY_SPLIT")
        or os.environ.get("RL_DATA_SPLIT")
        or "train"
    ).strip() or "train"
    RL_COLLECTION_SPLIT = (
        os.environ.get("RL_COLLECTION_SPLIT")
        or RL_QUERY_SPLIT
    ).strip() or RL_QUERY_SPLIT
    LOG_PATH = os.environ.get("LOG_PATH", "./tmp/rl_ultra_v3")
    LOAD_CHECKPOINT_PATH = os.environ.get("LOAD_CHECKPOINT_PATH", "")
    SFT_CHECKPOINT_PATH = os.environ.get("SFT_CHECKPOINT_PATH", "")
    TTL_SECONDS_RAW = os.environ.get("TTL_SECONDS", "none").strip().lower()
    if TTL_SECONDS_RAW in ("", "none", "null"):
        TTL_SECONDS: Optional[int] = None
    else:
        TTL_SECONDS = int(TTL_SECONDS_RAW)
    MODEL_NAME = "openai/gpt-oss-20b"
    KL_PENALTY_COEF = float(os.environ.get("KL_PENALTY_COEF", "0.005"))

    # Guardrails for checkpoint type wiring:
    # - policy init should come from /weights/
    # - KL reference should come from /sampler_weights/
    if LOAD_CHECKPOINT_PATH and "/sampler_weights/" in LOAD_CHECKPOINT_PATH:
        inferred_policy_ckpt = LOAD_CHECKPOINT_PATH.replace(
            "/sampler_weights/", "/weights/"
        )
        logger.warning(
            "policy_checkpoint_looks_like_sampler_path",
            provided=LOAD_CHECKPOINT_PATH,
            inferred=inferred_policy_ckpt,
            action="using_inferred_weights_path",
        )
        LOAD_CHECKPOINT_PATH = inferred_policy_ckpt

    if SFT_CHECKPOINT_PATH and "/weights/" in SFT_CHECKPOINT_PATH:
        inferred_kl_ref = SFT_CHECKPOINT_PATH.replace(
            "/weights/", "/sampler_weights/"
        )
        logger.warning(
            "kl_reference_looks_like_weights_path",
            provided=SFT_CHECKPOINT_PATH,
            inferred=inferred_kl_ref,
            action="using_inferred_sampler_path",
        )
        SFT_CHECKPOINT_PATH = inferred_kl_ref

    BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "32"))
    GROUP_SIZE = int(os.environ.get("GROUP_SIZE", "8"))
    LEARNING_RATE = 1e-5
    TEMPERATURE = float(os.environ.get("ROLLOUT_TEMPERATURE", "1.0"))
    LOSS_FN = "cispo"
    LORA_RANK = 32
    MAX_TOKENS = 2048
    SEED = 42
    EPOCHS = int(os.environ.get("EPOCHS", "2"))
    THREAD_POOL_SIZE = 256
    NUM_SUBSTEPS = 4

    if os.environ.get("SMALL_SCALE_TEST"):
        BATCH_SIZE = 4
        GROUP_SIZE = 2
        EPOCHS = 1

    config = get_config()
    tiktoken_encoding = tiktoken.get_encoding("o200k_harmony")
    text_token_counter = lambda text: len(tiktoken_encoding.encode(text))

    # Reranker backend for search/read tools (v8d expects this on in full runs).
    reranker_backend = os.environ.get("RERANKER_BACKEND", "baseten").strip().lower()
    reranker_max_tokens = int(os.environ.get("RERANKER_MAX_TOKENS", "4096"))
    reranker: Optional[Reranker] = None
    if reranker_backend in ("", "none", "off", "disabled"):
        reranker = None
    elif reranker_backend == "baseten":
        reranker = BasetenReranker(
            token_counter=text_token_counter,
            max_tokens=reranker_max_tokens,
        )
    elif reranker_backend == "contextual":
        reranker = ContextualReranker(
            token_counter=text_token_counter,
            max_tokens=reranker_max_tokens,
        )
    else:
        raise ValueError(
            f"Unsupported RERANKER_BACKEND='{reranker_backend}'. "
            "Expected one of: none, baseten, contextual."
        )
    logger.info(
        "reranker_configured",
        backend=reranker_backend,
        enabled=bool(reranker),
        max_tokens=reranker_max_tokens,
    )
    logger.info(
        "checkpoint_wiring",
        policy_checkpoint=LOAD_CHECKPOINT_PATH or None,
        kl_reference_checkpoint=SFT_CHECKPOINT_PATH or None,
        ttl_seconds=TTL_SECONDS,
        kl_penalty_coef=KL_PENALTY_COEF,
    )

    dataset_pairs: List[DatasetToolsetPair] = []
    for dataset_name in DATASETS:
        dataset_name = dataset_name.strip()
        if not dataset_name:
            continue
        dataset = get_dataset(dataset_name)
        collection_names = dataset.get_chroma_collections(split=RL_COLLECTION_SPLIT)

        chroma_client = config.get_chroma_client()
        openai_client = config.get_openai_client()

        search_tool = SearchCorpusTool(
            chroma_client=chroma_client,
            openai_client=openai_client,
            chroma_collection_name=collection_names,
            reranker=reranker,
            snippet_max_chars=2048,
            display_limit=SEARCH_DISPLAY_LIMIT,
        )
        toolset = ToolSet(name=f"{dataset_name}_toolset")
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

        max_q = SIMPLE_DATASET_MAX_QUERIES if dataset_name.endswith("_simple") else None
        pair = DatasetToolsetPair(
            dataset=dataset, toolset=toolset, search_tool=search_tool,
            max_train_queries=max_q,
        )
        dataset_pairs.append(pair)
        logger.info(
            "dataset_ready",
            dataset=dataset_name,
            query_split=RL_QUERY_SPLIT,
            collection_split=RL_COLLECTION_SPLIT,
            queries=len(dataset.get_all_query_ids(split=RL_QUERY_SPLIT)),
            max_train_queries=max_q,
        )

    rl_dataset = SlidingWindowRLDataset(
        dataset_pairs=dataset_pairs,
        batch_size=BATCH_SIZE,
        group_size=GROUP_SIZE,
        text_token_counter=text_token_counter,
        seed=SEED,
        epochs=EPOCHS,
        query_split=RL_QUERY_SPLIT,
    )
    logger.info(
        "dataset_created",
        batches=len(rl_dataset),
        batch_size=BATCH_SIZE,
        group_size=GROUP_SIZE,
        query_split=RL_QUERY_SPLIT,
        collection_split=RL_COLLECTION_SPLIT,
        rollout_temperature=TEMPERATURE,
        use_window_slicing=USE_WINDOW_SLICING,
        window_size=WINDOW_SIZE,
        max_windows=MAX_WINDOWS,
        recent_k=RECENT_K,
    )

    if USE_WINDOW_SLICING:
        # Optional legacy path: monkey-patch do_group_rollout for window slicing.
        import tinker_cookbook.rl.rollouts as rollouts_module
        _original_do_group_rollout = rollouts_module.do_group_rollout

        async def _patched_do_group_rollout(env_group_builder, policy):
            if isinstance(env_group_builder, SlidingWindowGroupBuilder):
                return await do_group_rollout_with_windows(env_group_builder, policy)
            return await _original_do_group_rollout(env_group_builder, policy)

        rollouts_module.do_group_rollout = _patched_do_group_rollout
        logger.info("patched_rollout", rollout_mode="window_slicing")
    else:
        # Default path: context-1 style full trajectories with terminal reward.
        logger.info("using_default_rollout", rollout_mode="full_trajectory")

    @chz.chz
    class UltraRLDatasetBuilder(RLDatasetBuilder):
        async def __call__(self) -> Tuple[RLDataset, Optional[RLDataset]]:
            return rl_dataset, None

    # KL reference is only needed when KL penalty is enabled.
    use_kl_reference = KL_PENALTY_COEF > 0 and bool(SFT_CHECKPOINT_PATH)
    if KL_PENALTY_COEF > 0 and not SFT_CHECKPOINT_PATH:
        logger.warning(
            "kl_penalty_enabled_without_reference",
            kl_penalty_coef=KL_PENALTY_COEF,
            message="KL penalty > 0 but SFT_CHECKPOINT_PATH is empty; training will fail.",
        )

    kl_config = None
    if use_kl_reference:
        kl_config = train.KLReferenceConfig(
            base_model=MODEL_NAME,
            load_checkpoint_path=SFT_CHECKPOINT_PATH,
        )

    blueprint = chz.Blueprint(train.Config).apply({
        "log_path": LOG_PATH,
        "load_checkpoint_path": LOAD_CHECKPOINT_PATH or None,
        "model_name": MODEL_NAME,
        "dataset_builder": UltraRLDatasetBuilder(),
        "learning_rate": LEARNING_RATE,
        "lora_rank": LORA_RANK,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "num_substeps": NUM_SUBSTEPS,
        "loss_fn": LOSS_FN,
        "loss_fn_config": {"clip_low_threshold": 0, "clip_high_threshold": 5},
        "eval_every": 2,
        "save_every": 1,
        "ttl_seconds": TTL_SECONDS,
        "remove_constant_reward_groups": True,
        "kl_penalty_coef": KL_PENALTY_COEF,
        "kl_reference_config": kl_config,
        "compute_post_kl": bool(use_kl_reference),
    })

    cfg = blueprint.make()
    logger.info("starting_rl_v3", thread_pool_size=THREAD_POOL_SIZE)

    async def run():
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor(max_workers=THREAD_POOL_SIZE) as executor:
            loop.set_default_executor(executor)
            await train.main(cfg)

    asyncio.run(run())
