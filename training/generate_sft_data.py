
# Allow direct execution from subdirectories while keeping imports package-relative.
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

"""Generate SFT trajectories using GPT-5.4 as a live single-phase agent.

Key changes from v2:
  - Single-phase: GPT-5.4 gets all 7 tools and picks freely each turn.
    No forced curate_and_decide compound tool.
  - Uses WorkingMemory from ultra_core for state tracking.
  - Generates result summaries during trajectory creation (included in prompt).
  - read_document is available and handled.
  - Stores normalize_ids per trajectory for train_sft_v3 alignment.
"""

import argparse
import json
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import structlog
import tiktoken

from harness.config import get_config
from datagen.search_dataset import SearchDataset, get_dataset
from harness.tools import (
    SearchCorpusTool,
    GrepCorpusTool,
    ReadDocumentTool,
)
from harness.ultra_core import (
    WorkingMemory,
    build_result_summary,
    get_system_prompt,
    parse_doc_texts_from_observation,
    parse_doc_ids_from_observation,
    FAN_OUT_MAX_QUERIES,
    MAX_CURATED_DOCS,
    MAX_OBS_CHARS,
    MAX_REVIEW_DOCS,
    MAX_TURNS,
    RECENT_K,
    SEARCH_DISPLAY_LIMIT,
    SEARCH_TOKEN_BUDGET,
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
    VALID_IMPORTANCE,
)

logger = structlog.get_logger()

GPT5_MODEL = os.environ.get("GPT5_MODEL", "gpt-5.4")
MIN_TURNS_BEFORE_END = 3

_REASONING_PROP = {
    "reasoning": {
        "type": "string",
        "description": (
            "Your step-by-step analysis BEFORE acting. Include: "
            "(1) what you learned from the last result, "
            "(2) what gaps remain, "
            "(3) why you chose this tool and parameters. "
            "If your current approach isn't working, explain what you'll try differently."
        ),
    }
}

TOOL_DEFS = [
    {
        "type": "function",
        "function": {
            "name": "fan_out_search",
            "description": (
                f"Run up to {FAN_OUT_MAX_QUERIES} diverse search queries in parallel. "
                "Returns combined results. Best for broad exploration."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    **_REASONING_PROP,
                    "queries": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": f"List of search queries (max {FAN_OUT_MAX_QUERIES}).",
                    },
                },
                "required": ["reasoning", "queries"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_corpus",
            "description": "Deep single-query search. Returns more results per query. Best for targeted follow-up.",
            "parameters": {
                "type": "object",
                "properties": {
                    **_REASONING_PROP,
                    "query": {
                        "type": "string",
                        "description": "The search query.",
                    },
                },
                "required": ["reasoning", "query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep_corpus",
            "description": "Exact regex pattern matching. Best for specific names, dates, numbers, codes, or exact phrases from the query. Often finds what semantic search misses.",
            "parameters": {
                "type": "object",
                "properties": {
                    **_REASONING_PROP,
                    "pattern": {
                        "type": "string",
                        "description": "Regex pattern to search for.",
                    },
                },
                "required": ["reasoning", "pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_document",
            "description": "Read a document's full content from the corpus. Use liberally — full text reveals connections that snippets miss. Read partially-matching docs before deciding relevance.",
            "parameters": {
                "type": "object",
                "properties": {
                    **_REASONING_PROP,
                    "doc_id": {
                        "type": "string",
                        "description": "The document ID to read.",
                    },
                },
                "required": ["reasoning", "doc_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "review_docs",
            "description": (
                "Re-read documents from your memory. Shows full text of "
                "previously-found documents without re-searching. Free."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    **_REASONING_PROP,
                    "doc_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": f"Document IDs to review (max {MAX_REVIEW_DOCS}).",
                    },
                },
                "required": ["reasoning", "doc_ids"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "curate",
            "description": (
                f"Update your curated set (max {MAX_CURATED_DOCS}). "
                "These documents are your final output. "
                "MUST be called after every search — review results and add ALL plausibly "
                "relevant docs. Err on the side of including more. Typically add 3-8 docs per call."
                + (
                    " v8d: you MUST include an `importance` map tagging EVERY doc_id in "
                    "add_ids with exactly one of 'very_high'|'high'|'fair'|'low'. When the "
                    "curated set is full, the lowest-importance docs are evicted first. "
                    "Empty/omitted importance map is a format error."
                    if V8D_IMPORTANCE_TAGGING else ""
                )
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    **_REASONING_PROP,
                    "add_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Document IDs to add to curated set. Include ALL plausibly relevant docs from recent results.",
                    },
                    "remove_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Document IDs to remove.",
                    },
                    **(
                        {
                            "importance": {
                                "type": "object",
                                "description": (
                                    "v8d REQUIRED: per-doc importance tag for every doc in add_ids. "
                                    "Keys are doc IDs from add_ids, values ∈ {'very_high','high','fair','low'}. "
                                    "'very_high' = verified directly answers the query (preferably after a verify call). "
                                    "'high' = strongly relevant, matches most query constraints. "
                                    "'fair' = plausible but unconfirmed (the safe default). "
                                    "'low' = marginal; will be evicted first. "
                                    "Example: {\"doc_id_1\": \"high\", \"doc_id_2\": \"fair\"}."
                                ),
                                "additionalProperties": {
                                    "type": "string",
                                    "enum": ["very_high", "high", "fair", "low"],
                                },
                            }
                        }
                        if V8D_IMPORTANCE_TAGGING else {}
                    ),
                },
                "required": (
                    ["reasoning", "add_ids", "importance"]
                    if V8D_IMPORTANCE_TAGGING
                    else ["reasoning", "add_ids"]
                ),
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "end_search",
            "description": (
                "End your search and submit your curated set as your final answer. "
                "Call this when you've thoroughly explored the corpus and curated all relevant documents."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "reasoning": {
                        "type": "string",
                        "description": (
                            "Explain why you're concluding: what you found, "
                            "what searches you've exhausted, and your confidence level."
                        ),
                    }
                },
                "required": ["reasoning"],
            },
        },
    },
]

if V8D_VERIFY_TOOL:
    TOOL_DEFS.append({
        "type": "function",
        "function": {
            "name": "verify",
            "description": (
                "v8d: Check whether specific documents support a concrete claim. Returns "
                "a yes/no judgement per doc with a short rationale. Use BEFORE tagging docs "
                "as 'very_high' on multi-constraint queries. Does NOT cost corpus tokens."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    **_REASONING_PROP,
                    "doc_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Doc IDs to verify (max 5).",
                    },
                    "claim": {
                        "type": "string",
                        "description": (
                            "A concrete checkable claim derived from the query. E.g.: "
                            "'This filing is a 10-K for Tesla filed in 2023 mentioning an SEC fine.'"
                        ),
                    },
                },
                "required": ["reasoning", "doc_ids", "claim"],
            },
        },
    })


# ═══════════════════════════════════════════════════════════════════════════════
# GPT-5 API
# ═══════════════════════════════════════════════════════════════════════════════

def call_gpt5(
    client, prompt: str, tools: List[Dict], max_retries: int = 3,
) -> Tuple[str, str, Dict]:
    """Call GPT-5.4 with tools. Returns (tool_name, reasoning_text, params).

    Reasoning is extracted from the 'reasoning' field in tool arguments,
    since tool_choice='required' causes models to put all text in args.
    """
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=GPT5_MODEL,
                messages=[{"role": "user", "content": prompt}],
                tools=tools,
                tool_choice="required",
                max_completion_tokens=4096,
            )
            msg = resp.choices[0].message
            if msg.tool_calls:
                tc = msg.tool_calls[0]
                params = json.loads(tc.function.arguments)
                reasoning = params.pop("reasoning", "") or (msg.content or "")
                return tc.function.name, reasoning, params
            logger.warning("gpt5_no_tool_call", attempt=attempt)
            if attempt < max_retries - 1:
                time.sleep(1)
        except Exception as e:
            logger.warning("gpt5_error", attempt=attempt, error=str(e)[:200])
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
    return "", "", {}


# ═══════════════════════════════════════════════════════════════════════════════
# Prompt builder
# ═══════════════════════════════════════════════════════════════════════════════

def build_gpt5_prompt(
    query: str,
    wm: WorkingMemory,
    recent_turns: List[Dict],
    result_summaries: List[str],
    turn_idx: int,
    max_turns: int = MAX_TURNS,
) -> str:
    """Build a plain-text prompt for GPT-5.4 that mirrors the Harmony context structure."""
    parts = [get_system_prompt(query)]

    if turn_idx >= RECENT_K and wm.turn_number > 0:
        parts.append(f"\n{wm.to_text()}\n")

    start = max(0, len(recent_turns) - RECENT_K)
    for i, rt in enumerate(recent_turns[start:], start=start):
        is_latest = (i == len(recent_turns) - 1)
        parts.append(f"\n--- Turn {rt['turn_idx']} ---")
        if rt.get("reasoning"):
            analysis = rt['reasoning']
            if not is_latest and len(analysis) > 600:
                analysis = analysis[:600] + "..."
            parts.append(f"[Analysis]: {analysis}")
        parts.append(f"[Tool]: {rt['tool_name']}({json.dumps(rt['params'])[:400]})")

        obs = rt.get("observation", "")
        obs_limit = 15000 if is_latest else 4000
        if len(obs) > obs_limit:
            obs = obs[:obs_limit] + "\n... (truncated)"
        parts.append(f"[Result]: {obs}")

        if i < len(result_summaries) and not is_latest:
            parts.append(f"\n{result_summaries[i]}")

    parts.append(f"\n--- Turn {turn_idx + 1} of {max_turns}: Your action ---")

    turns_left = max_turns - turn_idx
    curated_count = len(wm.curated_ids)

    guidance = []
    guidance.append(
        "In the 'reasoning' field, explain: (1) what you learned, "
        "(2) what gaps remain, (3) why you chose this tool."
    )

    # Determine what the last action was
    last_tool = recent_turns[-1]['tool_name'] if recent_turns else None
    last_was_search = last_tool in ('fan_out_search', 'search_corpus', 'grep_corpus', 'read_document')
    all_tools_used = set(rt['tool_name'] for rt in recent_turns)

    # First turn: decompose the query
    if turn_idx == 0:
        guidance.append(
            "Start by identifying the key aspects/constraints in the query "
            "(entities, dates, relationships, distinctive facts). "
            "Then search for the most specific/unique aspect first."
        )

    # Core rhythm enforcement: curate after every search
    if last_was_search:
        curate_msg = (
            "IMPORTANT: You just searched. Follow the search → curate rhythm: "
            "call curate NOW to add ALL plausibly relevant docs from your last results. "
            "Include borderline docs — it's always better to over-curate than under-curate. "
            "Do NOT search again until you've curated."
        )
        if V8D_IMPORTANCE_TAGGING:
            curate_msg += (
                " **REQUIRED:** every call to `curate` MUST include an `importance` "
                "map with one tag per added doc_id, drawn from "
                "{very_high, high, fair, low}. Default to 'fair' if uncertain, "
                "but DO NOT omit the field — empty importance means the eviction "
                "policy can't distinguish your docs and you will lose your best finds."
            )
        guidance.append(curate_msg)

    # v8d: verify-tool nudge — fire once per trajectory when the model has some
    # candidates but hasn't used verify yet, and the query looks multi-constraint.
    if V8D_VERIFY_TOOL and 'verify' not in all_tools_used and turn_idx >= 6 and curated_count >= 3:
        q_lower = (query or "").lower()
        looks_multi = (
            len(q_lower.split()) >= 20
            or any(kw in q_lower for kw in (" and ", " specific ", " same ", " both ", " each "))
        )
        if looks_multi or turn_idx >= 12:
            guidance.append(
                "TIP (v8d): You have several candidate docs but haven't called `verify` yet. "
                "For multi-constraint queries, call `verify(doc_ids, claim)` on your "
                "2-4 most promising candidates with a concrete claim that combines ALL "
                "query constraints. Only after verify returns YES for a doc should you "
                "tag it as `very_high`."
            )

    if turns_left <= 5 and curated_count == 0:
        guidance.append(
            "URGENT: Very few turns left and NO curated documents! "
            "Curate the most relevant docs from your pool immediately."
        )
    elif turns_left <= 5:
        guidance.append(
            f"FINAL STRETCH: Only {turns_left} turns left. "
            "Curate any remaining relevant docs and call end_search."
        )
    elif turns_left <= 12:
        guidance.append(
            f"{turns_left} turns remaining. Consider: have you tried grep_corpus "
            "for specific names/dates? Have you used read_document on partially-matching docs?"
        )

    if turn_idx > 4 and curated_count < 3:
        guidance.append(
            f"You've only curated {curated_count} docs so far. "
            "Curate more aggressively — add all plausibly relevant docs."
        )

    # Encourage grep_corpus for queries with specific facts
    if turn_idx >= 4 and 'grep_corpus' not in all_tools_used and not last_was_search:
        guidance.append(
            "TIP: Try grep_corpus with specific names, dates, numbers, "
            "or exact phrases from the query."
        )

    # Encourage read_document
    if turn_idx >= 6 and 'read_document' not in all_tools_used and not last_was_search:
        guidance.append(
            "TIP: Use read_document on partially-matching docs — "
            "full text often reveals connections that snippets miss."
        )

    # Consecutive search warning
    recent_tools = [rt['tool_name'] for rt in recent_turns[-2:]]
    search_tools = {'fan_out_search', 'search_corpus', 'grep_corpus', 'read_document'}
    consec_searches = sum(1 for t in recent_tools if t in search_tools)
    if consec_searches >= 2:
        guidance.append(
            "WARNING: Multiple consecutive searches without curating. "
            "You MUST curate before searching again."
        )

    # --- Backtracking triggers ---
    # Detect stale pool: recent searches yielded few new docs
    if len(wm.search_history) >= 3:
        recent_entries = wm.search_history[-3:]
        low_yield = sum(1 for e in recent_entries if ", 0 new" in e or ", 1 new" in e)
        if low_yield >= 2:
            guidance.append(
                "BACKTRACK: Your last few searches found almost no new documents. "
                "Your current search angle is exhausted. In your reasoning, explicitly: "
                "(1) state what isn't working, (2) re-read the query for missed facets, "
                "(3) try a COMPLETELY different decomposition — different entities, "
                "synonyms, related concepts, or indirect connections."
            )

    # Detect re-reading loop: same doc read/reviewed multiple times
    if turn_idx >= 8:
        recent_doc_actions = []
        for rt in recent_turns[-6:]:
            if rt['tool_name'] in ('read_document', 'review_docs'):
                doc_id = rt['params'].get('doc_id', '') or str(rt['params'].get('doc_ids', ''))
                recent_doc_actions.append(doc_id)
        if len(recent_doc_actions) >= 3 and len(set(recent_doc_actions)) <= len(recent_doc_actions) // 2:
            guidance.append(
                "BACKTRACK: You are re-reading the same documents repeatedly. "
                "This is a sign you're stuck. In your reasoning: (1) identify what "
                "specific information you're still missing, (2) search for that "
                "information directly with new queries instead of re-reading."
            )

    # Detect stagnant pool size over many turns
    pool_size = wm.get_pool_size()
    if turn_idx >= 10 and pool_size > 0 and not last_was_search:
        searches_done = len(wm.search_history)
        if searches_done >= 5 and curated_count < 5:
            guidance.append(
                f"BACKTRACK: You've done {searches_done} searches and found "
                f"{pool_size} docs but only curated {curated_count}. "
                "Either curate more aggressively from your existing pool, "
                "or reason about whether your query decomposition is fundamentally wrong."
            )

    parts.append(" ".join(guidance))
    return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════════════
# Tool execution
# ═══════════════════════════════════════════════════════════════════════════════

def execute_tool(
    tool_name: str,
    params: Dict,
    wm: WorkingMemory,
    search_tool: SearchCorpusTool,
    grep_tool: Optional[GrepCorpusTool],
    read_tool: Optional[ReadDocumentTool],
    ids_seen: Set[str],
) -> str:
    """Execute a tool call and return the observation text."""

    def _v8d_wrap_search_obs(
        obs_text: str,
        query_for_compress: str,
        ranked_doc_ids: Optional[List[str]] = None,
        allow_auto_populate: bool = True,
    ) -> str:
        """Apply BM25 compression + auto-populate + token marker to a search-like obs.

        Auto-populate uses wm.auto_populated to track first-search per trajectory,
        so this is safe across multiple concurrent trajectories.
        """
        if V8D_SENTENCE_COMPRESS and query_for_compress:
            obs_text = compress_search_observation(query_for_compress, obs_text)
        if (
            V8D_AUTO_POPULATE_FIRST_SEARCH
            and allow_auto_populate
            and ranked_doc_ids
            and not wm.auto_populated
        ):
            added = auto_populate_from_first_search(
                wm, ranked_doc_ids, top_k=AUTO_POPULATE_TOP_K
            )
            if added > 0:
                obs_text += (
                    f"\n\n[AUTO-POPULATED] Top {added} docs from this search have been "
                    "added to your curated set at 'fair' importance. Promote good ones "
                    "to 'high'/'very_high' and REMOVE irrelevant ones via curate."
                )
        if V8D_TOKEN_BUDGET_MARKER:
            # Approx: obs length as cheap proxy for used context (avoids re-tokenizing).
            approx_used = min(max(len(obs_text) // 3, 500), 30000)
            obs_text = append_token_marker(obs_text, approx_used)
        return obs_text

    if tool_name == "fan_out_search":
        queries = params.get("queries", [])[:FAN_OUT_MAX_QUERIES]
        all_results = []
        total_ids = []
        pool_before = wm.get_pool_size()
        per_query_token_budget = max(512, SEARCH_TOKEN_BUDGET // max(len(queries), 1))
        overrides_base = {
            "ignore_ids": list(ids_seen),
            "max_tokens": per_query_token_budget,
        }
        if V8D_ADAPTIVE_RERANK_INSTRUCTION and wm.rerank_instruction:
            overrides_base["rerank_instruction"] = wm.rerank_instruction
        for q in queries:
            if not isinstance(q, str) or not q.strip():
                continue
            try:
                result_text, meta = search_tool({"query": q}, overrides_base)
                all_results.append(result_text)
                if meta and hasattr(meta, "returned_chunk_ids"):
                    ids_seen.update(meta.returned_chunk_ids)
                    total_ids.extend(meta.returned_chunk_ids)
                doc_texts = parse_doc_texts_from_observation(result_text)
                wm.add_to_pool(list(doc_texts.keys()), doc_texts)
            except Exception as e:
                all_results.append(f"Error: {str(e)[:100]}")
        n = max(len(all_results), 1)
        per_query_budget = MAX_OBS_CHARS // n
        truncated_results = []
        for r in all_results:
            if len(r) > per_query_budget:
                r = r[:per_query_budget] + f"\n... (truncated to {per_query_budget} chars)"
            truncated_results.append(r)
        obs = "\n".join(truncated_results) if truncated_results else "No results."
        q_summary = "; ".join(str(q)[:30] for q in queries[:3])
        num_new = wm.get_pool_size() - pool_before
        wm.add_search_record("fan_out", q_summary, len(total_ids), num_new=num_new)
        concat_q = " ".join(str(q) for q in queries if isinstance(q, str))
        obs = _v8d_wrap_search_obs(
            obs, concat_q, ranked_doc_ids=total_ids, allow_auto_populate=True,
        )
        return obs

    elif tool_name == "search_corpus":
        query = params.get("query", "")
        if not query:
            return "No query provided."
        try:
            pool_before = wm.get_pool_size()
            overrides = {"ignore_ids": list(ids_seen), "max_tokens": SEARCH_TOKEN_BUDGET}
            if V8D_ADAPTIVE_RERANK_INSTRUCTION and wm.rerank_instruction:
                overrides["rerank_instruction"] = wm.rerank_instruction
            result_text, meta = search_tool({"query": query}, overrides)
            ranked = list(meta.returned_chunk_ids) if meta and hasattr(meta, "returned_chunk_ids") else []
            if meta and hasattr(meta, "returned_chunk_ids"):
                ids_seen.update(meta.returned_chunk_ids)
            doc_texts = parse_doc_texts_from_observation(result_text)
            wm.add_to_pool(list(doc_texts.keys()), doc_texts)
            num_new = wm.get_pool_size() - pool_before
            wm.add_search_record("search", query[:50], len(doc_texts), num_new=num_new)
            result_text = _v8d_wrap_search_obs(
                result_text, query, ranked_doc_ids=ranked, allow_auto_populate=True,
            )
            return result_text
        except Exception as e:
            return f"Error: {str(e)[:200]}"

    elif tool_name == "grep_corpus":
        pattern = params.get("pattern", "")
        if not pattern or not grep_tool:
            return "grep_corpus not available or empty pattern."
        try:
            pool_before = wm.get_pool_size()
            result_text, meta = grep_tool({"pattern": pattern})
            doc_texts = parse_doc_texts_from_observation(result_text)
            wm.add_to_pool(list(doc_texts.keys()), doc_texts)
            num_new = wm.get_pool_size() - pool_before
            wm.add_search_record("grep", pattern[:50], len(doc_texts), num_new=num_new)
            result_text = _v8d_wrap_search_obs(
                result_text, pattern, ranked_doc_ids=None, allow_auto_populate=False,
            )
            return result_text
        except Exception as e:
            return f"Error: {str(e)[:200]}"

    elif tool_name == "read_document":
        doc_id = params.get("doc_id", "")
        if not doc_id:
            return "No doc_id provided."
        if not read_tool:
            return "read_document not available."
        try:
            pool_before = wm.get_pool_size()
            result_text, meta = read_tool(
                {"doc_id": doc_id},
                {"query": wm.query, "max_tokens": SEARCH_TOKEN_BUDGET},
            )
            doc_texts = parse_doc_texts_from_observation(result_text)
            wm.add_to_pool(list(doc_texts.keys()), doc_texts)
            num_new = wm.get_pool_size() - pool_before
            wm.add_search_record("read", doc_id, len(doc_texts), num_new=num_new)
            # read_document returns full text — skip compression, just add token marker
            if V8D_TOKEN_BUDGET_MARKER:
                approx_used = min(max(len(result_text) // 3, 500), 30000)
                result_text = append_token_marker(result_text, approx_used)
            return result_text
        except Exception as e:
            return f"Error: {str(e)[:200]}"

    elif tool_name == "review_docs":
        doc_ids = params.get("doc_ids", [])[:MAX_REVIEW_DOCS]
        if not doc_ids:
            return "No doc_ids provided."
        result = wm.review_docs(doc_ids)
        wm.add_search_record("review", ", ".join(doc_ids[:3]), len(doc_ids))
        return result

    elif tool_name == "curate":
        add_ids = params.get("add_ids", [])
        remove_ids = params.get("remove_ids", [])
        if not isinstance(add_ids, list):
            add_ids = [add_ids] if add_ids else []
        if not isinstance(remove_ids, list):
            remove_ids = [remove_ids] if remove_ids else []
        importance = None
        if V8D_IMPORTANCE_TAGGING:
            raw = params.get("importance")
            if isinstance(raw, dict):
                importance = {str(k): str(v) for k, v in raw.items()}
        return wm.curate(add_ids, remove_ids, importance=importance)

    elif tool_name == "verify" and V8D_VERIFY_TOOL:
        doc_ids = params.get("doc_ids", [])[:5]
        claim = str(params.get("claim", "")).strip()
        if not isinstance(doc_ids, list):
            doc_ids = [str(doc_ids)] if doc_ids else []
        if not doc_ids or not claim:
            return "verify: doc_ids or claim missing."
        doc_texts: Dict[str, str] = {}
        for did in doc_ids:
            norm = wm._normalize_id(str(did).strip())
            store = wm.doc_store.get(norm, {})
            txt = store.get("full_text") or store.get("snippet") or ""
            if txt:
                doc_texts[norm] = txt
        if not doc_texts:
            return "verify: no matching docs in memory (call read_document first)."
        # Use the same openai client (passed via module-level attribute, see generate_trajectory)
        import openai as _openai_mod  # noqa
        global _VERIFY_CLIENT
        cli = _VERIFY_CLIENT
        if cli is None:
            try:
                cli = get_config().get_openai_client()
                _VERIFY_CLIENT = cli
            except Exception as e:
                return f"verify: openai client unavailable ({str(e)[:80]})"
        wm.add_search_record("verify", claim[:50], len(doc_ids), num_new=0)
        return exec_verify_claim(cli, doc_texts, claim)

    elif tool_name == "end_search":
        return "Search concluded."

    else:
        return f"Unknown tool: {tool_name}"


# v8d: module-level state for verify client cache only.
# (First-search marker is per-trajectory via wm.auto_populated.)
_VERIFY_CLIENT = None


# ═══════════════════════════════════════════════════════════════════════════════
# Trajectory generation
# ═══════════════════════════════════════════════════════════════════════════════

def generate_trajectory(
    query_id: str,
    query_text: str,
    dataset: SearchDataset,
    search_tool: SearchCorpusTool,
    grep_tool: Optional[GrepCorpusTool],
    read_tool: Optional[ReadDocumentTool],
    client,
    normalize_ids: bool,
) -> Optional[Dict]:
    """Run GPT-5.4 as a single-phase agent with all 7 tools."""
    wm = WorkingMemory(query_text, normalize_ids=normalize_ids)
    # v8d: build per-episode rerank instruction (domain preset, cheap/deterministic).
    wm.rerank_instruction = build_rerank_instruction(
        query=query_text,
        dataset_name=getattr(dataset, "name", None),
        openai_client=None,
        use_llm=False,
    )
    ids_seen: Set[str] = set()
    turn_history: List[Dict[str, Any]] = []
    result_summaries: List[str] = []
    tool_types_used: Set[str] = set()
    turns_since_curate = 0
    total_curate_calls = 0

    ground_truth_ids = list(dataset.get_expected_document_ids(query_id))

    for turn_idx in range(MAX_TURNS):
        prompt = build_gpt5_prompt(
            query_text, wm, turn_history, result_summaries, turn_idx,
        )

        tool_name, reasoning, params = call_gpt5(client, prompt, TOOL_DEFS)
        if not tool_name:
            logger.warning("no_tool_call", turn=turn_idx, query_id=query_id)
            continue

        tool_types_used.add(tool_name)
        pool_size_before = wm.get_pool_size()

        obs = execute_tool(
            tool_name, params, wm, search_tool, grep_tool, read_tool, ids_seen,
        )

        turn_history.append({
            "turn_idx": turn_idx + 1,
            "tool_name": tool_name,
            "params": params,
            "reasoning": reasoning,
            "observation": obs,
        })

        # Track curate state
        if tool_name == "curate":
            turns_since_curate = 0
            total_curate_calls += 1
        elif tool_name != "end_search":
            turns_since_curate += 1

        # Build result summary
        summary = build_result_summary(
            obs_text=obs,
            tool_names=[tool_name],
            wm=wm,
            turns_since_curate=turns_since_curate,
            tool_types_used=tool_types_used,
            current_turn=turn_idx + 1,
            pool_size_before=pool_size_before,
        )
        result_summaries.append(summary)

        wm.advance_turn()

        if tool_name == "end_search" and turn_idx >= MIN_TURNS_BEFORE_END:
            break

    curated_ids = wm.curated_ids
    pool_ids = wm.pool_ids
    recall = dataset.evaluate_results_recall(query_id, curated_ids) if curated_ids else 0.0
    precision = dataset.evaluate_results_precision(query_id, curated_ids) if curated_ids else 0.0
    pool_recall = dataset.evaluate_results_recall(query_id, pool_ids) if pool_ids else 0.0

    logger.info(
        "trajectory_done",
        query_id=query_id,
        turns=len(turn_history),
        curated=len(curated_ids),
        pool=len(pool_ids),
        recall=round(recall, 3),
        precision=round(precision, 3),
        pool_recall=round(pool_recall, 3),
        tools_used=sorted(tool_types_used),
    )

    # Serialize doc_store (snippets only, not full text — full text is too large)
    doc_store_data = {}
    for doc_id, info in wm.doc_store.items():
        doc_store_data[doc_id] = {
            "snippet": info.get("snippet", ""),
        }

    return {
        "format_version": "ultra_v3",
        "generator": f"{GPT5_MODEL}_single_phase",
        "query_id": query_id,
        "query_text": query_text,
        "dataset_name": dataset.name,
        "normalize_ids": normalize_ids,
        "num_turns": len(turn_history),
        "final_recall": recall,
        "final_precision": precision,
        "pool_recall": pool_recall,
        "curated_ids": curated_ids,
        "pool_ids": pool_ids[:100],
        "ground_truth_ids": ground_truth_ids,
        "turn_history": turn_history,
        "search_history": wm.search_history,
        "doc_store": doc_store_data,
        "result_summaries": result_summaries,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Generate SFT trajectories for Ultra v3"
    )
    parser.add_argument("--num-queries", type=int, default=50)
    parser.add_argument("--datasets", type=str, default="browsecompplus,sec")
    parser.add_argument("--output-dir", type=str, default="sft_ultra_v3_data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--split", type=str, default="sft")
    parser.add_argument(
        "--query-ids", type=str, default=None,
        help="Comma-separated list of specific query IDs to generate (overrides --num-queries)",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = get_config()
    client = config.get_openai_client()
    tiktoken_enc = tiktoken.get_encoding("o200k_harmony")
    text_token_counter = lambda text: len(tiktoken_enc.encode(text))

    dataset_names = [d.strip() for d in args.datasets.split(",") if d.strip()]
    total_generated = 0
    total_failed = 0

    for ds_name in dataset_names:
        dataset = get_dataset(ds_name)
        normalize_ids = (
            getattr(dataset, "evaluation_mode", "document") == "document"
        )

        collection_names = dataset.get_chroma_collections(split="train")
        chroma_client = config.get_chroma_client()
        openai_client = config.get_openai_client()

        search_tool = SearchCorpusTool(
            chroma_client=chroma_client,
            openai_client=openai_client,
            chroma_collection_name=collection_names,
            reranker=None,
            snippet_max_chars=2048,
            display_limit=SEARCH_DISPLAY_LIMIT,
        )
        grep_tool = GrepCorpusTool(
            chroma_client=chroma_client,
            chroma_collection_name=collection_names,
            token_counter=text_token_counter,
        )
        read_tool = ReadDocumentTool(
            chroma_client=chroma_client,
            chroma_collection_name=collection_names,
            reranker=None,
            token_counter=text_token_counter,
            max_tokens=4096,
        )

        if args.query_ids:
            requested = set(q.strip() for q in args.query_ids.split(","))
            all_ids = dataset.get_all_query_ids(split=args.split)
            selected = [q for q in all_ids if q in requested]
        else:
            query_ids = dataset.get_all_query_ids(split=args.split)
            random.shuffle(query_ids)
            selected = query_ids[:args.num_queries]

        logger.info(
            "starting_dataset",
            dataset=ds_name,
            queries=len(selected),
            normalize_ids=normalize_ids,
        )

        ds_generated = 0

        def _process(qi_qid):
            qi, qid = qi_qid
            out_path = output_dir / f"ultra_v3_{ds_name}_{qid}.json"
            if out_path.exists():
                logger.info("skipping_existing", query_id=qid, path=str(out_path))
                return True
            _, query_text = dataset.get_query_by_id(qid)
            traj = generate_trajectory(
                qid, query_text, dataset, search_tool, grep_tool, read_tool,
                client, normalize_ids,
            )
            if traj and traj["final_recall"] >= 0.0:
                with open(out_path, "w") as f:
                    json.dump(traj, f, indent=2, default=str)
                return True
            return False

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(_process, (qi, qid)): qid
                for qi, qid in enumerate(selected)
            }
            for future in as_completed(futures):
                try:
                    if future.result():
                        ds_generated += 1
                    else:
                        total_failed += 1
                except Exception as e:
                    logger.error("trajectory_error", error=str(e)[:200])
                    total_failed += 1

        total_generated += ds_generated
        logger.info(
            "dataset_done", dataset=ds_name, generated=ds_generated,
        )

    logger.info(
        "all_done",
        total_generated=total_generated,
        total_failed=total_failed,
    )


if __name__ == "__main__":
    main()
