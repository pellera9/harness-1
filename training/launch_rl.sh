#!/usr/bin/env bash
# =============================================================================
# v8d RL — match Context-1 training scale.
#
# Context-1 scale (per published report):
#   - 128 prompts/step × 8 rollouts/prompt = 1024 rollouts/step
#   - ~230 training steps
#   - ~5 epochs over the training set
#   - Total: ~235K rollouts
#
# We match this with:
#   - BATCH_SIZE=128, GROUP_SIZE=8
#   - EPOCHS=5
#   - All 4 main domains + 2 simple variants (curriculum)
#   - v8d harness (importance tagging, subtractive curation, evidence graph,
#     BM25 sentence compression, chunk-neighbor expansion, verify tool)
#   - Baseten reranker (Qwen3-Reranker-8B) for chunk rerank + verify
#
# Usage:
#   bash launch_rl.sh                    # full run
#   SMOKE_TEST=1 bash launch_rl.sh       # small debug run
#   tmux new -s rl_v8d 'bash launch_rl.sh'
# =============================================================================
set -euo pipefail

# Load API keys (TINKER_API_KEY, OPENAI_API_KEY, BASETEN_API_KEY, ...)
if [ -f .env.local ]; then
    set -a
    source .env.local
    set +a
fi

if [ -z "${TINKER_API_KEY:-}" ]; then
    echo "ERROR: TINKER_API_KEY is not set. Ensure .env.local exists and defines it." >&2
    exit 1
fi

RUN_NAME="${RUN_NAME:-rl_v8d_full}"
LOG_DIR="./tmp/${RUN_NAME}"
mkdir -p "$LOG_DIR" ./logs

# ----------------------------------------------------------------------------
# Scale — match Context-1
# ----------------------------------------------------------------------------
export TRAIN_DATASETS="${TRAIN_DATASETS:-browsecompplus,sec}"
export SIMPLE_DATASET_MAX_QUERIES="${SIMPLE_DATASET_MAX_QUERIES:-150}"
# Query/corpus split knobs:
# - RL_QUERY_SPLIT: dataset split for rollout query IDs
# - RL_COLLECTION_SPLIT: dataset split for retrieval collections
# - RL_DATA_SPLIT: legacy alias for RL_QUERY_SPLIT
export RL_QUERY_SPLIT="${RL_QUERY_SPLIT:-${RL_DATA_SPLIT:-train}}"
export RL_COLLECTION_SPLIT="${RL_COLLECTION_SPLIT:-$RL_QUERY_SPLIT}"
export RL_DATA_SPLIT="$RL_QUERY_SPLIT"
export BATCH_SIZE="${BATCH_SIZE:-128}"
export GROUP_SIZE="${GROUP_SIZE:-8}"
export EPOCHS="${EPOCHS:-5}"

# ----------------------------------------------------------------------------
# v8d harness flags (consumed by ultra_core / train_rl)
# ----------------------------------------------------------------------------
export V8D_SUBTRACTIVE_CURATION="${V8D_SUBTRACTIVE_CURATION:-1}"
export V8D_IMPORTANCE_TAGGING="${V8D_IMPORTANCE_TAGGING:-1}"
export V8D_AUTO_POPULATE_FIRST_SEARCH="${V8D_AUTO_POPULATE_FIRST_SEARCH:-1}"
export V8D_EVIDENCE_GRAPH="${V8D_EVIDENCE_GRAPH:-1}"
export V8D_SENTENCE_COMPRESS="${V8D_SENTENCE_COMPRESS:-1}"
export SENTENCE_COMPRESS_K="${SENTENCE_COMPRESS_K:-4}"
export V8D_CHUNK_NEIGHBORS="${V8D_CHUNK_NEIGHBORS:-0}"
export V8D_CONTENT_DEDUP="${V8D_CONTENT_DEDUP:-1}"
export V8D_VERIFY_TOOL="${V8D_VERIFY_TOOL:-1}"
export V8D_TOKEN_BUDGET_MARKER="${V8D_TOKEN_BUDGET_MARKER:-1}"
export V8D_ADAPTIVE_RERANK_INSTRUCTION="${V8D_ADAPTIVE_RERANK_INSTRUCTION:-0}"
export AUTO_POPULATE_TOP_K="${AUTO_POPULATE_TOP_K:-8}"

# ----------------------------------------------------------------------------
# Reward / rollout defaults
#
# Default to the same rollout regime used for the selected SFT checkpoint
# comparison so RL warm-start behavior is directly comparable before training:
#   - MAX_TURNS=128 (avoid short hard cap during quality-focused runs)
#   - turn penalty disabled by default (quality-first phase)
#   - rollout temperature 1.0
#   - KL anchor on by default for stability
# ----------------------------------------------------------------------------
export RECALL_BETA="${RECALL_BETA:-2.0}"
export FINAL_ANSWER_BONUS="${FINAL_ANSWER_BONUS:-1.0}"
export FINAL_ANSWER_RECALL_WEIGHT="${FINAL_ANSWER_RECALL_WEIGHT:-0.8}"
export TRAJECTORY_FA_RECALL_WEIGHT="${TRAJECTORY_FA_RECALL_WEIGHT:-0.4}"
export FA_MISS_PENALTY_WEIGHT="${FA_MISS_PENALTY_WEIGHT:-0.35}"
export TURN_PENALTY_MIN_TURNS="${TURN_PENALTY_MIN_TURNS:-24}"
export TURN_PENALTY_MAX="${TURN_PENALTY_MAX:-0.0}"
export MAX_TURNS="${MAX_TURNS:-128}"
export ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-1.0}"
export KL_PENALTY_COEF="${KL_PENALTY_COEF:-0.005}"
# Default to Context-1 style full-trajectory training (no window slicing).
export USE_WINDOW_SLICING="${USE_WINDOW_SLICING:-0}"
# Use terminal-only reward (no per-turn window rewards/shaping) for v8d.
export USE_TERMINAL_ONLY_REWARD="${USE_TERMINAL_ONLY_REWARD:-1}"
# Keep per-turn dense shaping terms off.
export DELTA_RECALL_BONUS="${DELTA_RECALL_BONUS:-0.0}"
export CONSEC_SEARCH_PENALTY="${CONSEC_SEARCH_PENALTY:-0.0}"
export TOOL_DIVERSITY_BONUS="${TOOL_DIVERSITY_BONUS:-0.5}"
export TOOL_DIVERSITY_TARGET="${TOOL_DIVERSITY_TARGET:-6}"
export TOOL_DIVERSITY_SHORTFALL_PENALTY="${TOOL_DIVERSITY_SHORTFALL_PENALTY:-0.08}"

# ----------------------------------------------------------------------------
# Chroma Cloud throttling
# ----------------------------------------------------------------------------
export CHROMA_SEARCH_MAX_CONCURRENCY="${CHROMA_SEARCH_MAX_CONCURRENCY:-8}"

# ----------------------------------------------------------------------------
# Reranker — Baseten (high throughput for BATCH=128)
# ----------------------------------------------------------------------------
export RERANKER_BACKEND="${RERANKER_BACKEND:-baseten}"
export RERANKER_MODEL="${RERANKER_MODEL:-qwen3-reranker-8b}"
# BASETEN_API_KEY is expected to be set in environment

# ----------------------------------------------------------------------------
# SFT warmup checkpoint wiring
#
# IMPORTANT:
# - RL policy init (train.Config.load_checkpoint_path) expects a WEIGHTS state path:
#     <training-weights-uri>
# - KL reference sampling client expects a SAMPLER path:
#     <sampler-weights-uri>
#
# We auto-read both from tmp/sft_v8d_warmup/checkpoints.jsonl.
# Default selection is the final user-trained SFT checkpoint; override with
# SFT_SELECTION_NAME=000550 or pass LOAD_CHECKPOINT_PATH/SFT_CHECKPOINT_PATH
# explicitly.
# ----------------------------------------------------------------------------
_SFT_LOG="${SFT_LOG_DIR:-./tmp/sft_v8d_warmup}"
_SFT_SELECTION_RAW="${SFT_SELECTION_NAME:-final}"
if [[ "${_SFT_SELECTION_RAW}" =~ ^step([0-9]+)$ ]]; then
    _SFT_SELECTION_NAME="$(printf "%06d" "${BASH_REMATCH[1]}")"
else
    _SFT_SELECTION_NAME="${_SFT_SELECTION_RAW}"
fi

_SELECTED_STATE_URI=""
_SELECTED_SAMPLER_URI=""
if [ -f "${_SFT_LOG}/checkpoints.jsonl" ]; then
    _SELECTED_STATE_URI="$(python3 -c "
import json
from pathlib import Path
p = Path('${_SFT_LOG}/checkpoints.jsonl')
target = '${_SFT_SELECTION_NAME}'
fallback = None
for line in p.read_text().splitlines():
    o = json.loads(line)
    if o.get('name') == target and o.get('state_path'):
        print(o['state_path'])
        raise SystemExit(0)
    if o.get('name') == 'final' and o.get('state_path'):
        fallback = o['state_path']
if fallback:
    print(fallback)
" 2>/dev/null || true)"
    _SELECTED_SAMPLER_URI="$(python3 -c "
import json
from pathlib import Path
p = Path('${_SFT_LOG}/checkpoints.jsonl')
target = '${_SFT_SELECTION_NAME}'
fallback = None
for line in p.read_text().splitlines():
    o = json.loads(line)
    if o.get('name') == target and o.get('sampler_path'):
        print(o['sampler_path'])
        raise SystemExit(0)
    if o.get('name') == 'final' and o.get('sampler_path'):
        fallback = o['sampler_path']
if fallback:
    print(fallback)
" 2>/dev/null || true)"
fi

# Policy init checkpoint (weights path) for RL training client.
if [ -z "${LOAD_CHECKPOINT_PATH:-}" ] && [ -n "${_SELECTED_STATE_URI}" ]; then
    export LOAD_CHECKPOINT_PATH="${_SELECTED_STATE_URI}"
fi
# KL reference checkpoint (sampler path) for optional KL penalty.
if [ -z "${SFT_CHECKPOINT_PATH:-}" ] && [ -n "${_SELECTED_SAMPLER_URI}" ]; then
    export SFT_CHECKPOINT_PATH="${_SELECTED_SAMPLER_URI}"
fi

# If KL ref is still unset, infer sampler path from the policy weights path.
if [ -z "${SFT_CHECKPOINT_PATH:-}" ] && [ -n "${LOAD_CHECKPOINT_PATH:-}" ]; then
    if [[ "${LOAD_CHECKPOINT_PATH}" == *"/weights/"* ]]; then
        export SFT_CHECKPOINT_PATH="${LOAD_CHECKPOINT_PATH/\/weights\//\/sampler_weights\/}"
    fi
fi

# If policy init is unset but KL ref is known, infer weights path from sampler path.
if [ -z "${LOAD_CHECKPOINT_PATH:-}" ] && [ -n "${SFT_CHECKPOINT_PATH:-}" ]; then
    if [[ "${SFT_CHECKPOINT_PATH}" == *"/sampler_weights/"* ]]; then
        export LOAD_CHECKPOINT_PATH="${SFT_CHECKPOINT_PATH/\/sampler_weights\//\/weights\/}"
    fi
fi

# Guardrail: KL reference must be a sampler path.
if [[ "${SFT_CHECKPOINT_PATH:-}" == *"/weights/"* ]]; then
    _KL_AUTO_SAMPLER_PATH="${SFT_CHECKPOINT_PATH/\/weights\//\/sampler_weights\/}"
    echo "WARN: SFT_CHECKPOINT_PATH looks like a weights path; switching to sampler path for KL reference."
    echo "      ${_KL_AUTO_SAMPLER_PATH}"
    export SFT_CHECKPOINT_PATH="${_KL_AUTO_SAMPLER_PATH}"
fi

# Keep explicit env values if provided, otherwise leave empty.
export LOAD_CHECKPOINT_PATH="${LOAD_CHECKPOINT_PATH:-}"
export SFT_CHECKPOINT_PATH="${SFT_CHECKPOINT_PATH:-}"

# ----------------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------------
export LOG_PATH="$LOG_DIR"

# ----------------------------------------------------------------------------
# Smoke test overrides
# ----------------------------------------------------------------------------
if [ -n "${SMOKE_TEST:-}" ]; then
    export SMALL_SCALE_TEST=1
    export BATCH_SIZE=4
    export GROUP_SIZE=2
    export EPOCHS=1
    export TRAIN_DATASETS="browsecompplus,sec"
    RUN_NAME="rl_v8d_smoke"
    LOG_DIR="./tmp/${RUN_NAME}"
    export LOG_PATH="$LOG_DIR"
    mkdir -p "$LOG_DIR"
fi

echo "=================================================================="
echo "v8d RL launch"
echo "=================================================================="
echo "Run name      : $RUN_NAME"
echo "Log dir       : $LOG_DIR"
echo "Datasets      : $TRAIN_DATASETS"
echo "Query split   : $RL_QUERY_SPLIT"
echo "Corpus split  : $RL_COLLECTION_SPLIT"
echo "Batch / Group : $BATCH_SIZE / $GROUP_SIZE"
echo "Epochs        : $EPOCHS"
echo "Temperature   : $ROLLOUT_TEMPERATURE"
if [ -n "${FORCE_QUERY_IDS:-}" ]; then
echo "Force queries : $FORCE_QUERY_IDS"
fi
echo "Rollout mode  : $([ \"$USE_WINDOW_SLICING\" = \"1\" ] && echo window_slicing || echo full_trajectory)"
echo "Reranker      : $RERANKER_BACKEND / $RERANKER_MODEL"
echo "SFT pick name : ${_SFT_SELECTION_NAME:-<manual override>}"
echo "Policy ckpt   : ${LOAD_CHECKPOINT_PATH:-<base model>}"
echo "KL penalty    : $KL_PENALTY_COEF"
echo "KL ref ckpt   : ${SFT_CHECKPOINT_PATH:-<none>}"
echo "FA weights    : final=$FINAL_ANSWER_RECALL_WEIGHT pool=$TRAJECTORY_FA_RECALL_WEIGHT miss_penalty=$FA_MISS_PENALTY_WEIGHT"
echo "Turn penalty  : start=$TURN_PENALTY_MIN_TURNS max=$TURN_PENALTY_MAX"
echo ""
echo "Expected: ${BATCH_SIZE}x${GROUP_SIZE}=$((BATCH_SIZE*GROUP_SIZE)) rollouts/step"
echo "Target steps : ~230  (~$((230 * BATCH_SIZE * GROUP_SIZE)) total rollouts)"
echo ""

PYTHONPATH=. uv run python training/train_rl.py 2>&1 | tee "./logs/${RUN_NAME}.log"
