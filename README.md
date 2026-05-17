# HarnesS-1

[![Tinker Inference](https://img.shields.io/badge/Tinker-Inference-073f3d?labelColor=white)](https://github.com/pat-jj/harness-1/blob/main/inference/tinker_inference.md)
[![Model Checkpoint](https://img.shields.io/badge/Hugging%20Face-Checkpoint-FFCA03?logo=huggingface&logoColor=FFCA03)](https://huggingface.co/pat-jj/harness-1)
[![arXiv](https://img.shields.io/badge/arXiv-coming%20soon-b31b1b.svg?logo=arxiv&logoColor=white)](https://arxiv.org/)

HarnesS-1 is a 20B search agent trained with reinforcement learning inside a
stateful retrieval harness. The harness maintains the recoverable search state:
candidate documents, curated evidence, evidence links, verification records, and
budget-aware context.

The policy keeps the semantic decisions: what to search, which documents to
inspect or curate, what claims to verify, and when the evidence is sufficient.

![HarnesS-1 average search performance](assets/teaser_recall_barchart.png)

## Checkpoint

The released HarnesS-1 checkpoint is available on Hugging Face:
[pat-jj/harness-1](https://huggingface.co/pat-jj/harness-1).

Set the model repository once and reuse it across inference and export scripts:

```bash
HARNESS1_HF_MODEL=pat-jj/harness-1
```

## Repository Layout

- `training/`: SFT data generation, SFT training, RL training, and launch scripts.
- `inference/`: Harness-1 evaluation, component ablations, HF inference, and vLLM inference.
- `inference/baselines/`: in-domain and transfer baseline evaluation runners.
- `harness/`: shared harness, tool, trajectory, task, reranking, and config modules.
- `model_export/`: helper scripts for merging a private Tinker adapter into a Hugging Face model.
- `datagen/` and `eval_scripts/`: dataset and auxiliary evaluation code.
- `tinker-cookbook/`: local Tinker cookbook dependency used by the training scripts.
- `tests/`: lightweight import and CLI smoke tests.

## Dataset Availability

The public, ready-to-run evaluation path in this repository is
BrowseComp+ (`browsecompplus`). It uses the public BrowseComp+ release plus the
local qrel/query files described in `datagen/README.md`.

The other in-domain corpora used in the paper (`web`, `sec`, and `patents`) are
not packaged here as public ready-made indexes. To evaluate those settings, first
construct the corresponding corpora and Chroma collections by following the
Context-1 data-generation pipeline:
[chroma-core/context-1-data-gen](https://github.com/chroma-core/context-1-data-gen).
After those corpora are available in your Chroma deployment, the evaluation
entrypoints in `inference/` can target them.

## Setup

Use Python 3.11+ and install dependencies with `uv`:

```bash
uv sync
```

For local vLLM serving, install the optional extra:

```bash
uv sync --extra vllm
```

Create a local environment file from the template:

```bash
cp .env.example .env.local
```

Fill only the credentials needed for the workflow you plan to run. `.env.local`
is ignored by git; do not commit real keys or tokens.

Common requirements:

- Training and Tinker-hosted evaluation: `TINKER_API_KEY`.
- Corpus retrieval and embeddings: Chroma/OpenAI configuration.
- Reranking: `BASETEN_API_KEY`.
- Hugging Face upload: a write-capable Hugging Face token exported in your shell.

## Smoke Tests

These tests validate imports and command-line wiring without launching a full
training or evaluation run:

```bash
uv run python tests/smoke_imports.py
uv run python tests/smoke_cli.py
```

## Inference

The released merged checkpoint is published as
[`pat-jj/harness-1`](https://huggingface.co/pat-jj/harness-1) on Hugging Face.
Run a basic model-load test with:

```bash
uv run python inference/hf_inference.py \
  --model ${HARNESS1_HF_MODEL:-pat-jj/harness-1} \
  --prompt "Briefly describe HarnesS-1."
```

For Tinker-hosted inference with the published Tinker checkpoint, see
[`inference/tinker_inference.md`](inference/tinker_inference.md). That document
contains the public Tinker checkpoint path, required harness flags, and a
BrowseComp+ example run.

For local vLLM serving:

```bash
uv sync --extra vllm
uv run python inference/vllm_local_inference.py serve \
  --model ${HARNESS1_HF_MODEL:-pat-jj/harness-1} \
  --served-model-name harness-1
```

## Training Pipeline

1. Generate SFT trajectories in `training/`.
2. Train the SFT checkpoint in `training/`.
3. Launch RL from the user-trained SFT checkpoint with `training/launch_rl.sh`.

Start with the folder README:

```bash
less training/README.md
```

## Evaluation

Evaluate HarnesS-1 search behavior with:

```bash
PYTHONPATH=. uv run python inference/evaluate_harness1.py --help
```

For published-checkpoint evaluation, use either the Hugging Face model
[`pat-jj/harness-1`](https://huggingface.co/pat-jj/harness-1) with a local
serving backend, or the Tinker-hosted checkpoint documented in
[`inference/tinker_inference.md`](inference/tinker_inference.md). Transfer
evaluation, ablations, and baseline runners are documented in
`inference/README.md`. The default HarnesS-1 search operating point is
temperature `1.0`.

## Model Export

`model_export/` contains utilities for downloading a Tinker checkpoint adapter,
merging it into the base GPT-OSS model, and uploading the resulting full model
to Hugging Face. See `model_export/README.md`.
