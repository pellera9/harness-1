# Inference And Evaluation

This folder contains HarnesS-1 inference and evaluation scripts. The default
released model is `$HARNESS1_HF_MODEL`.

The public merged checkpoint is hosted at
[pat-jj/harness-1](https://huggingface.co/pat-jj/harness-1).

## Standard HF Inference

Use this for a direct Transformers load/generation smoke test:

```bash
uv run python inference/hf_inference.py \
  --model ${HARNESS1_HF_MODEL:-harness-1} \
  --prompt "Briefly describe HarnesS-1 in one sentence."
```

## vLLM Inference

Install the optional vLLM dependency before serving locally:

```bash
uv sync --extra vllm
```

Start a local OpenAI-compatible server:

```bash
uv run python inference/vllm_local_inference.py serve \
  --model ${HARNESS1_HF_MODEL:-harness-1} \
  --served-model-name harness-1 \
  --tensor-parallel-size 1
```

Query the server:

```bash
uv run python inference/vllm_local_inference.py query \
  --url http://localhost:8000 \
  --served-model-name harness-1 \
  --prompt "What is HarnesS-1?"
```

`vllm_modal_inference.py` provides the Modal deployment version.

## Harness-1 Search Evaluation

The Harness-1 operating point is temperature `1.0`. The eval scripts default to
that value.

BrowseComp+ is the public ready-to-run dataset path documented in
`datagen/README.md`. The `web`, `sec`, and `patents` settings require users to
first build the matching Chroma-backed corpora, for example using the
[Context-1 data-generation repository](https://github.com/chroma-core/context-1-data-gen).

```bash
set -a && source .env.local && set +a
PYTHONPATH=. uv run python inference/evaluate_harness1.py \
  --dataset browsecompplus \
  --split test \
  --collection-split test \
  --max-turns 40 \
  --temperature 1.0 \
  --checkpoints harness1="$HARNESS1_TINKER_CHECKPOINT" \
  --output tmp/eval_harness1_browsecompplus.json
```

Transfer datasets use `evaluate_transfer.py`.

## BrowseComp+ Component Ablation

```bash
set -a && source .env.local && set +a
PYTHONPATH=. uv run python inference/queue_browsecomp_ablation.py \
  --temperature 1.0 \
  --max-turns 40 \
  --limit 100 \
  --max-parallel 2
```

The full Harness-1 flags are enabled by default. Ablation conditions disable one
mechanism at a time while keeping the same checkpoint and query IDs.

## Baselines

Baseline scripts live in `inference/baselines/`: `eval.py` is the in-domain
baseline runner, while `inference/baselines/transfer/` contains the web and
Wikipedia transfer-dataset baseline runners.
