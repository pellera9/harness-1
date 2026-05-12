#!/usr/bin/env bash
# =============================================================================
# v8d SFT training — warmup GPT-OSS-20B on v8d-generated trajectories.
#
# Goal: teach the model the v8d tool-call format (incl. importance tagging and
# verify), reasoning style, and search/curate cadence so RL starts from a solid
# format prior instead of burning steps on malformed calls.
#
# Data: ./sft_ultra_v8d_data/*.json   (899 raw GPT-5.4 trajectories)
#   - With MIN_RECALL=0.1 we keep ~717 (drop ~182 zero-recall garbage).
#   - Roughly 30–35 turns/traj × 717 trajs ≈ 23K training datums.
#
# IMPORTANT: All V8D_* flags MUST match the flags used at data-generation time
# (see launch_sft_generation.sh) AND the flags used at RL time (launch_rl.sh).
# Otherwise the tool schemas / system prompt / observation wrapping will diverge.
#
# Usage:
#   bash launch_sft_training.sh                       # full run
#   SMOKE_TEST=1 bash launch_sft_training.sh          # dry-run sanity
#   MIN_RECALL=0.0 bash launch_sft_training.sh        # use ALL 899 raw
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

RUN_NAME="${RUN_NAME:-sft_v8d_warmup}"
LOG_DIR="./tmp/${RUN_NAME}"
mkdir -p "$LOG_DIR" ./logs

# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
export DATA_DIR="${DATA_DIR:-./sft_ultra_v8d_data}"
export MIN_RECALL="${MIN_RECALL:-0.1}"      # drops 0-recall failures; set 0.0 for ALL
export MAX_LENGTH="${MAX_LENGTH:-32768}"

# ----------------------------------------------------------------------------
# Training config
# ----------------------------------------------------------------------------
export MODEL_NAME="${MODEL_NAME:-openai/gpt-oss-20b}"
export NUM_EPOCHS="${NUM_EPOCHS:-3}"
export BATCH_SIZE="${BATCH_SIZE:-128}"
export LEARNING_RATE="${LEARNING_RATE:-5e-6}"
export LORA_RANK="${LORA_RANK:-32}"
export SAVE_EVERY="${SAVE_EVERY:-50}"
export EVAL_EVERY="${EVAL_EVERY:-50}"

# ----------------------------------------------------------------------------
# v8d harness flags — MUST match SFT-gen and RL launchers exactly.
# ----------------------------------------------------------------------------
export V8D_SUBTRACTIVE_CURATION=1
export V8D_IMPORTANCE_TAGGING=1
export V8D_AUTO_POPULATE_FIRST_SEARCH=1
export V8D_EVIDENCE_GRAPH=1
export V8D_SENTENCE_COMPRESS=1
export V8D_CONTENT_DEDUP=1
export V8D_VERIFY_TOOL=1
export V8D_TOKEN_BUDGET_MARKER=1
export V8D_ADAPTIVE_RERANK_INSTRUCTION=1
export AUTO_POPULATE_TOP_K="${AUTO_POPULATE_TOP_K:-8}"

# ----------------------------------------------------------------------------
# Smoke test overrides
# ----------------------------------------------------------------------------
if [ -n "${SMOKE_TEST:-}" ]; then
    export NUM_EPOCHS=1
    export BATCH_SIZE=4
    RUN_NAME="sft_v8d_smoke"
    LOG_DIR="./tmp/${RUN_NAME}"
    mkdir -p "$LOG_DIR"
fi

echo "=================================================================="
echo "v8d SFT training"
echo "=================================================================="
echo "Run           : $RUN_NAME"
echo "Model         : $MODEL_NAME"
echo "Data dir      : $DATA_DIR"
echo "Min recall    : $MIN_RECALL"
echo "Epochs        : $NUM_EPOCHS"
echo "Batch size    : $BATCH_SIZE"
echo "LR / LoRA     : $LEARNING_RATE / r=$LORA_RANK"
echo "Max length    : $MAX_LENGTH"
echo ""

PYTHONPATH=. uv run python training/train_sft.py \
    --data-dir "$DATA_DIR" \
    --log-path "$LOG_DIR" \
    --model-name "$MODEL_NAME" \
    --num-epochs "$NUM_EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --learning-rate "$LEARNING_RATE" \
    --lora-rank "$LORA_RANK" \
    --max-length "$MAX_LENGTH" \
    --min-recall "$MIN_RECALL" \
    --save-every "$SAVE_EVERY" \
    --eval-every "$EVAL_EVERY" \
    2>&1 | tee "./logs/${RUN_NAME}.log"
