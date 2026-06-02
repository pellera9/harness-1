# Training

This folder contains the Harness-1 training pipeline:

1. `generate_sft_data.py`: generate Harness-1 SFT trajectories.
2. `train_sft.py`: train the SFT warm-start checkpoint with Tinker.
3. `train_rl.py`: run RL with the stateful Harness-1 environment.
4. `launch_*.sh`: launch scripts with the default project settings.

Run commands from the repository root so imports and relative paths resolve.

## SFT Data Generation

```bash
set -a && source .env.local && set +a
uv run python training/generate_sft_data.py \
  --datasets browsecompplus,sec,patents,web \
  --num-queries 50 \
  --output-dir tmp/sft_data
```

For a small wiring check:

```bash
uv run python training/generate_sft_data.py --help
```

## SFT Training

```bash
set -a && source .env.local && set +a
uv run python training/train_sft.py \
  --data-dir tmp/sft_data \
  --log-path tmp/sft_v8d_warmup \
  --model-name openai/gpt-oss-20b \
  --num-epochs 3 \
  --batch-size 128
```

The Tinker checkpoint log is written to:

```bash
tmp/sft_v8d_warmup/checkpoints.jsonl
```

## RL Training

RL should start from the actual user-trained SFT checkpoint. The launcher
defaults to selecting `final` from `tmp/sft_v8d_warmup/checkpoints.jsonl`.
You can also provide explicit checkpoint URIs.

```bash
set -a && source .env.local && set +a
RUN_NAME=rl_ultra_0424_div \
MAX_TURNS=40 \
TURN_PENALTY_MIN_TURNS=20 \
TURN_PENALTY_MAX=0.02 \
TOOL_DIVERSITY_BONUS=0.25 \
TOOL_DIVERSITY_TARGET=6 \
GAP_PENALTY_WEIGHT=0.0 \
KL_PENALTY_COEF=0.0 \
TRAIN_DATASETS=sec \
SFT_SELECTION_NAME=final \
bash training/launch_rl.sh
```

If you know the exact Tinker URIs, pass them directly:

```bash
LOAD_CHECKPOINT_PATH="$HARNESS1_TINKER_TRAINING_WEIGHTS" \
SFT_CHECKPOINT_PATH="$HARNESS1_TINKER_SAMPLER_WEIGHTS" \
bash training/launch_rl.sh
```

For a small training smoke run:

```bash
SMOKE_TEST=1 TRAIN_DATASETS=sec bash training/launch_rl.sh
```

This still contacts Tinker and the retrieval stack, so it requires valid keys.
