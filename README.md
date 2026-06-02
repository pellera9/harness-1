# Harness-1

[![Tinker Inference](https://img.shields.io/badge/Tinker-Inference-073f3d?labelColor=white)](https://github.com/pat-jj/harness-1/blob/main/inference/tinker_inference.md)
[![Model Checkpoint](https://img.shields.io/badge/Hugging%20Face-Checkpoint-FFCA03?logo=huggingface&logoColor=FFCA03)](https://huggingface.co/pat-jj/harness-1)
[![arXiv](https://img.shields.io/badge/arXiv-2606.02373-b31b1b.svg?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2606.02373)

Harness-1 is a 20B search agent trained with reinforcement learning inside a
stateful retrieval harness. The harness maintains recoverable search state:
candidate documents, curated evidence, evidence links, verification records, and
budget-aware context. The policy keeps the semantic decisions: what to search,
which documents to inspect or curate, what claims to verify, and when the
evidence is sufficient.

This repository is intended for researchers, evaluation authors, and engineers
who want to serve the released checkpoint, inspect the search harness, or
reproduce BrowseComp+ evaluation runs. BrowseComp+ is a benchmark for difficult
browsing tasks where systems must find and curate supporting evidence; running
the evaluator produces recall, trajectory recall, final-answer recall,
precision, and per-query trajectory artifacts.

![HarnesS-1 average search performance](assets/teaser_recall_barchart.png)

## Quickstart

For a minimal local smoke test, you need:

- Linux with Python `3.11+`.
- `uv` installed.
- A CUDA-compatible NVIDIA GPU environment.
- vLLM with GPT-OSS support.
- Access to the released Hugging Face checkpoint.

Install dependencies:

```bash
uv sync --extra vllm
```

Set the checkpoint:

```bash
export HARNESS1_HF_MODEL=pat-jj/harness-1
```

Start with the detailed vLLM and BrowseComp+ guide:

```bash
less docs/run_vllm_browsecompplus.md
```

## Model Checkpoint

The released Harness-1 weights are hosted on Hugging Face:

```text
https://huggingface.co/pat-jj/harness-1
```

vLLM downloads the weights from Hugging Face on first use and then reuses the
local Hugging Face cache. See the Hugging Face model page for model-card details,
usage restrictions, and checkpoint metadata.

## What You Can Do

- Serve the released checkpoint locally with vLLM.
- Run raw `/v1/completions` smoke tests with token-id outputs.
- Evaluate Harness-1 search behavior on BrowseComp+ when a compatible retrieval
  backend is available.
- Run Tinker-hosted inference with the published checkpoint.
- Inspect and extend the stateful search harness, tool environment, training
  scripts, and evaluation runners.
- Run ablations and baselines for supported datasets.

## Repository Layout

- `docs/`: user-facing guides and runbooks.
- `harness/`: shared search harness, tools, trajectory, task, reranking, and
  configuration modules.
- `inference/`: HarnesS-1 evaluation, component ablations, HF inference, and
  vLLM inference utilities.
- `inference/baselines/`: in-domain and transfer baseline evaluation runners.
- `training/`: SFT data generation, SFT training, RL training, and launch scripts.
- `datagen/` and `eval_scripts/`: dataset and auxiliary evaluation code.
- `model_export/`: helper scripts for merging a private Tinker adapter into a
  Hugging Face model.
- `tinker-cookbook/`: local Tinker cookbook dependency used by the training scripts.
- `tests/`: lightweight import and CLI smoke tests.

## Setup Levels

### Minimum Model Serving

Use this if you only want to verify that the released checkpoint serves locally:

```bash
uv sync --extra vllm
export HARNESS1_HF_MODEL=pat-jj/harness-1
```

Then follow `docs/run_vllm_browsecompplus.md`.

### Full BrowseComp+ Evaluation

In addition to the minimum setup, BrowseComp+ evaluation requires:

- BrowseComp+ query, qrel, and answer files on disk.
- A Chroma collection containing BrowseComp+ corpus chunks with document IDs that
  match the qrels.
- OpenAI credentials for retrieval support used by the harness.
- Optional Baseten reranker credentials if reranking is enabled.

BrowseComp+ data setup is described in `datagen/README.md`. The end-to-end vLLM
evaluation path is documented in `docs/run_vllm_browsecompplus.md`.

### Development And Training

Use the base environment for lightweight tests and code development:

```bash
uv sync
uv run python tests/smoke_imports.py
uv run python tests/smoke_cli.py
```

Training scripts live in `training/`. Model export utilities live in
`model_export/`.

## Credentials And Security

Copy the environment template only when needed:

```bash
cp .env.example .env.local
```

Do not commit real credentials. `.env` and `.env.local` are ignored by this
repository.

Credential scope:

- `HUGGINGFACE_TOKEN`: used only if Hugging Face checkpoint access requires auth.
- `OPENAI_API_KEY`: used by retrieval/evaluation workflows.
- `CHROMA_API_KEY` and `CHROMA_DATABASE`: used by Chroma-backed evaluation.
- `BASETEN_API_KEY` and `BASETEN_MODEL_URL`: used only for the optional reranker.
- `TINKER_API_KEY`: used by Tinker-hosted training or evaluation paths.

## Dataset Availability

BrowseComp+ is the public evaluation path documented in this release. The code
expects local BrowseComp+ files plus a compatible Chroma retrieval collection.
See `datagen/README.md` and `docs/run_vllm_browsecompplus.md`.

The other in-domain corpora used in the paper, such as `web`, `sec`, and
`patents`, are not bundled as public ready-made indexes. To evaluate those
settings, first construct compatible corpora and Chroma collections, for example
with the Context-1 data-generation pipeline:
[chroma-core/context-1-data-gen](https://github.com/chroma-core/context-1-data-gen).

## Inference

Run a basic Hugging Face model-load test with:

```bash
uv run python inference/hf_inference.py \
  --model ${HARNESS1_HF_MODEL:-pat-jj/harness-1} \
  --prompt "Briefly describe HarnesS-1."
```

For Tinker-hosted inference with the published Tinker checkpoint, see
[`inference/tinker_inference.md`](inference/tinker_inference.md). That document
contains the public Tinker checkpoint path, required harness flags, and a
BrowseComp+ example run.

For local vLLM serving and BrowseComp+ evaluation, see
[`docs/run_vllm_browsecompplus.md`](docs/run_vllm_browsecompplus.md). The
end-to-end path uses `inference/evaluate_harness1_vllm.py` with raw
`/v1/completions` token-id prompts.

For a lightweight local vLLM server wrapper:

```bash
uv sync --extra vllm
uv run python inference/vllm_local_inference.py serve \
  --model ${HARNESS1_HF_MODEL:-pat-jj/harness-1} \
  --served-model-name harness-1
```

## Results And Reproducibility

Evaluation metrics depend on the query sample, Chroma index, reranker backend,
vLLM version, and GPU kernels. Small smoke tests are useful for validating setup
but have high variance. Larger query sets are more appropriate for reporting
aggregate metrics.

The detailed vLLM guide explains how to read final metrics including:

- `recall`: recall of evidence documents in the final curated set.
- `final_answer_recall`: recall over evidence tied to the final answer.
- `trajectory_recall`: evidence recall anywhere in the search trajectory.
- `precision`: precision of the final curated set.

## Glossary

- HarnesS-1 operating point: the component flags and generation settings used for
  the full search harness.
- BrowseComp+: a benchmark for browsing and evidence-seeking questions.
- qrels: relevance labels that map query IDs to gold or evidence document IDs.
- Curated evidence recall: how much gold evidence appears in the final curated
  document set.
- Trajectory recall: how much gold evidence appears anywhere during the search
  trajectory.
- Raw `/v1/completions`: the OpenAI-compatible completion endpoint used with
  pre-tokenized prompts.
- Integer token prompts: prompt inputs sent as token IDs instead of plain text.
- `V8D_` flags: environment flags that enable HarnesS-1 search components.

## Known Limitations

- Full BrowseComp+ evaluation requires a compatible Chroma retrieval backend; the
  large retrieval index is not bundled in this repository.
- Results can vary with external retrieval and reranking services.
- Local serving requires a CUDA GPU environment with enough memory for the
  checkpoint. Non-H100 GPUs may work with sufficient memory and vLLM support, but
  the documented path was validated on H100-class hardware.
- Some training and model-export workflows depend on private checkpoints or
  hosted services.

## Documentation

- `docs/run_vllm_browsecompplus.md`: detailed local vLLM and BrowseComp+ guide.
- `inference/tinker_inference.md`: Tinker-hosted inference guide.
- `datagen/README.md`: dataset setup notes.
- `inference/README.md`: inference, evaluation, ablation, and baseline entrypoints.
- `model_export/README.md`: model export utilities.

## Support And Contributing

Please use the repository issue tracker for bug reports, setup problems, and
feature requests. Contributions should keep public documentation free of private
paths, secrets, and service-specific assumptions unless they are clearly marked
as optional.

## License And Citation

The repository does not currently include a `LICENSE` file or BibTeX citation in
this checkout. Add the intended license, citation, and model usage terms before a
public release.

For checkpoint-specific metadata, see the Hugging Face model card at
`pat-jj/harness-1`.
