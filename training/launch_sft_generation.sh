#!/usr/bin/env bash
# =============================================================================
# v8d SFT data generation — small, high-quality format-bootstrap SFT.
#
# Rationale (see docs/v8d_design_plan.md §11): v8d does NOT use SFT for behavior
# cloning. The harness (evidence graph, importance tags, auto-populate, verify)
# carries the cognitive load, and RL has 235K rollouts to learn behavior.
# SFT only needs to teach the model the output format and v8d's tool-call schema
# so RL doesn't burn early steps on malformed calls.
#
# Target: ~500 FILTERED trajectories after strict quality gates.
# Generation budget: ~1,000 raw (50-60% will pass filters in filter_sft_v8d.py).
#
# Raw distribution (total: 1,000 raw → ~500 filtered):
#   - browsecompplus : 300  (hardest, biggest share)
#   - sec            : 250
#   - patents        : 150
#   - web            : 150
#   - web_simple     :  75  (curriculum)
#   - sec_simple     :  75
#
# Expected runtime on 16 workers: ~3-5 hours (vs 24-36h at full scale).
# Resumable: generate_sft_data.py skips already-written trajectories (idempotent).
#
# Usage:
#   bash launch_sft_generation.sh                # full (1000 raw → ~500 filtered)
#   SMOKE_TEST=1 bash launch_sft_generation.sh   # tiny sanity: 10 queries per dataset
#   DATASETS="patents,web" bash launch_sft_generation.sh  # subset
# =============================================================================
set -euo pipefail

OUT_DIR="${OUT_DIR:-./sft_ultra_v8d_data}"
WORKERS="${WORKERS:-16}"
SPLIT="${SPLIT:-sft}"
GPT5_MODEL="${GPT5_MODEL:-gpt-5.4}"
MAX_TURNS="${MAX_TURNS:-40}"

mkdir -p "$OUT_DIR"
mkdir -p ./logs

export GPT5_MODEL MAX_TURNS

# v8d harness features (see docs/v8d_design_plan.md).
# These must be set or the SFT data will not exercise v8d's tool schema.
export V8D_SUBTRACTIVE_CURATION="${V8D_SUBTRACTIVE_CURATION:-1}"
export V8D_IMPORTANCE_TAGGING="${V8D_IMPORTANCE_TAGGING:-1}"
export V8D_AUTO_POPULATE_FIRST_SEARCH="${V8D_AUTO_POPULATE_FIRST_SEARCH:-1}"
export V8D_EVIDENCE_GRAPH="${V8D_EVIDENCE_GRAPH:-1}"
export V8D_SENTENCE_COMPRESS="${V8D_SENTENCE_COMPRESS:-1}"
export V8D_CONTENT_DEDUP="${V8D_CONTENT_DEDUP:-1}"
export V8D_VERIFY_TOOL="${V8D_VERIFY_TOOL:-1}"
export V8D_TOKEN_BUDGET_MARKER="${V8D_TOKEN_BUDGET_MARKER:-1}"
export V8D_ADAPTIVE_RERANK_INSTRUCTION="${V8D_ADAPTIVE_RERANK_INSTRUCTION:-1}"
export AUTO_POPULATE_TOP_K="${AUTO_POPULATE_TOP_K:-8}"

# Per-dataset raw targets (aim for ~500 after filter_sft_v8d.py)
if [ -n "${SMOKE_TEST:-}" ]; then
    declare -A TARGETS=(
        [browsecompplus]=10
        [sec]=10
        [patents]=10
        [web]=10
        [web_simple]=5
        [sec_simple]=5
    )
else
    declare -A TARGETS=(
        [browsecompplus]=300
        [sec]=250
        [patents]=150
        [web]=150
        [web_simple]=75
        [sec_simple]=75
    )
fi

# Default to all 6 datasets; allow override via DATASETS env var
ALL_DATASETS="${DATASETS:-browsecompplus,sec,patents,web,web_simple,sec_simple}"
IFS=',' read -ra DATASET_LIST <<< "$ALL_DATASETS"

echo "=================================================================="
echo "v8d SFT generation"
echo "=================================================================="
echo "Teacher model : $GPT5_MODEL"
echo "Output dir    : $OUT_DIR"
echo "Workers       : $WORKERS"
echo "Split         : $SPLIT"
echo "Datasets      : ${DATASET_LIST[*]}"
echo ""

for ds in "${DATASET_LIST[@]}"; do
    ds="$(echo "$ds" | xargs)"  # trim whitespace
    if [ -z "$ds" ]; then continue; fi
    target="${TARGETS[$ds]:-}"
    if [ -z "$target" ]; then
        echo "[$ds] SKIP: no target defined"
        continue
    fi
    log_file="./logs/sft_v8d_${ds}.log"
    echo "[$ds] target=$target, log=$log_file"

    PYTHONPATH=. uv run python training/generate_sft_data.py \
        --num-queries "$target" \
        --datasets "$ds" \
        --output-dir "$OUT_DIR" \
        --workers "$WORKERS" \
        --split "$SPLIT" \
        --seed 42 \
        2>&1 | tee "$log_file"
    echo ""
done

echo "=================================================================="
echo "Raw generation done. Count per dataset:"
total_raw=0
for ds in "${DATASET_LIST[@]}"; do
    ds="$(echo "$ds" | xargs)"
    if [ -z "$ds" ]; then continue; fi
    n=$(ls "$OUT_DIR"/ultra_v3_"${ds}"_*.json 2>/dev/null | wc -l)
    total_raw=$((total_raw + n))
    echo "  $ds : $n"
done
echo "  total raw: $total_raw"
echo ""
echo "Next step: filter to ~500 high-quality trajectories:"
echo "  uv run python filter_sft_v8d.py --input $OUT_DIR --output ${OUT_DIR}_filtered --target 500"
