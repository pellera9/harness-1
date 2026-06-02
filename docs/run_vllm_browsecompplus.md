# Run Harness-1 With vLLM And BrowseComp+

This guide shows how to serve the released Harness-1 checkpoint with vLLM and,
optionally, run the BrowseComp+ search evaluation.

Use the minimum setup if you only want to confirm that the model serves locally.
Use the full evaluation setup if you also have BrowseComp+ files and a compatible
retrieval backend.

## What You Will Run

Harness-1 is a long-horizon search agent. At evaluation time, the model produces
tool calls, retrieves and reads documents, maintains a curated evidence set, and
is scored against evidence labels.

BrowseComp+ is a benchmark for difficult browsing and evidence-seeking tasks.
Running the evaluation produces aggregate metrics such as final curated evidence
recall, final-answer recall, trajectory recall, precision, and per-query
trajectory files.

## Prerequisites

Minimum model-serving requirements:

- Linux with Python `3.11+`.
- `uv` installed. See [the uv installation docs](https://docs.astral.sh/uv/getting-started/installation/).
- NVIDIA GPU environment with a CUDA-compatible driver.
- A recent vLLM version with GPT-OSS support. The repository currently declares
  `vllm>=0.13.0`; the tested environment used vLLM `0.20.2`.
- Access to the Hugging Face checkpoint `pat-jj/harness-1`.

Full BrowseComp+ evaluation additionally requires:

- BrowseComp+ query, qrel, and answer files on disk.
- A Chroma collection containing BrowseComp+ corpus chunks with document IDs that
  match the qrels.
- OpenAI credentials for embedding/search support used by the harness.
- Optional Baseten reranker credentials if reranking is enabled.

If you do not have a CUDA GPU, you can inspect the code and run lightweight
import/CLI tests, but local vLLM serving of the released checkpoint is not the
intended path.

## Hugging Face Weights

The released Harness-1 weights are hosted on Hugging Face:

```text
https://huggingface.co/pat-jj/harness-1
```

Set the model repository once:

```bash
export HARNESS1_HF_MODEL="${HARNESS1_HF_MODEL:-pat-jj/harness-1}"
```

vLLM downloads the weights from Hugging Face on first use and then reuses the
local Hugging Face cache. If the checkpoint access policy changes, authenticate
with Hugging Face before starting vLLM:

```bash
huggingface-cli login
```

## Install

From the repository root:

```bash
cd /path/to/harness-1
uv sync --extra vllm
export PYTHONPATH=.
```

## Secret Handling

Copy the environment template and fill in only the keys needed for your workflow:

```bash
cp .env.example .env.local
```

Do not commit real credentials. This repository ignores `.env` and `.env.local`
by default.

Common variables:

- `HUGGINGFACE_TOKEN`: used only if Hugging Face checkpoint access requires auth.
- `OPENAI_API_KEY`: used by the BrowseComp+ evaluation harness for retrieval
  support.
- `CHROMA_API_KEY` and `CHROMA_DATABASE`: used only for Chroma-backed evaluation.
- `BASETEN_API_KEY` and `BASETEN_MODEL_URL`: used only when the Baseten reranker
  is enabled.
- `BROWSECOMPPLUS_*_PATH`: local BrowseComp+ query, qrel, and answer files.

Load local configuration when needed:

```bash
set -a
source .env.local
set +a

export HF_TOKEN="${HUGGINGFACE_TOKEN:-$HF_TOKEN}"
export HARNESS1_HF_MODEL="${HARNESS1_HF_MODEL:-pat-jj/harness-1}"
export PYTHONPATH=.
```

## Minimum Setup: Serve The Model

Single-GPU example:

```bash
CUDA_VISIBLE_DEVICES=0 \
VLLM_USE_DEEP_GEMM=0 \
VLLM_MOE_USE_DEEP_GEMM=0 \
uv run vllm serve "$HARNESS1_HF_MODEL" \
  --served-model-name harness-1 \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 1 \
  --max-model-len 32768 \
  --max-num-batched-tokens 16384 \
  --trust-remote-code \
  --moe-backend triton
```

Multi-GPU example:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
VLLM_USE_DEEP_GEMM=0 \
VLLM_MOE_USE_DEEP_GEMM=0 \
uv run vllm serve "$HARNESS1_HF_MODEL" \
  --served-model-name harness-1 \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 2 \
  --max-model-len 32768 \
  --max-num-batched-tokens 16384 \
  --trust-remote-code \
  --moe-backend triton
```

Keep `--tensor-parallel-size` equal to the number of visible GPUs.

The Triton MoE and DeepGEMM settings avoid two common startup failures on hosts
where the default FlashInfer CUTLASS path needs `nvcc`, or DeepGEMM is not
available in the runtime.

Wait until the server is healthy:

```bash
curl -sS http://127.0.0.1:8000/health
```

The health endpoint returns an empty successful response when the server is
ready.

## Minimum Smoke Test

The evaluation harness uses raw `/v1/completions`, not chat completions. It also
needs vLLM to return generated token IDs.

Run this check before launching an evaluation:

```bash
python - <<'PY'
import json
import urllib.request

payload = {
    "model": "harness-1",
    "prompt": "Say OK.",
    "max_tokens": 4,
    "temperature": 0.0,
    "stream": False,
    "return_token_ids": True,
}
req = urllib.request.Request(
    "http://127.0.0.1:8000/v1/completions",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urllib.request.urlopen(req, timeout=120) as resp:
    data = json.loads(resp.read().decode("utf-8"))
choice = data["choices"][0]
tokens = choice.get("token_ids") or choice.get("tokens") or choice.get("text_token_ids")
print(choice.get("text", "").replace("\n", "\\n"))
print("token_ids_returned", bool(tokens))
PY
```

Continue to evaluation only if `token_ids_returned True` is printed.

## Full Evaluation Setup

Configure BrowseComp+ paths in `.env.local`:

```bash
BROWSECOMPPLUS_QUERIES_PATH=/path/to/BrowseComp-Plus/topics-qrels/queries.tsv
BROWSECOMPPLUS_QRELS_GOLD_PATH=/path/to/BrowseComp-Plus/topics-qrels/qrel_golds.txt
BROWSECOMPPLUS_QRELS_EVIDENCE_PATH=/path/to/BrowseComp-Plus/topics-qrels/qrel_evidence.txt
BROWSECOMPPLUS_ANSWERS_PATH=/path/to/BrowseComp-Plus/data/browsecomp_plus_decrypted.jsonl
```

The Chroma deployment must contain the BrowseComp+ test collection expected by
the dataset loader, normally `browsecomp_plus_test`, with document IDs matching
the BrowseComp+ qrels.

Enable the full Harness-1 operating point in the same shell that will run the
evaluation:

```bash
export V8D_SUBTRACTIVE_CURATION=1
export V8D_IMPORTANCE_TAGGING=1
export V8D_AUTO_POPULATE_FIRST_SEARCH=1
export V8D_EVIDENCE_GRAPH=1
export V8D_SENTENCE_COMPRESS=1
export V8D_CHUNK_NEIGHBORS=0
export V8D_CONTENT_DEDUP=1
export V8D_VERIFY_TOOL=1
export V8D_TOKEN_BUDGET_MARKER=1
export V8D_ADAPTIVE_RERANK_INSTRUCTION=0
export SENTENCE_COMPRESS_K=4
export AUTO_POPULATE_TOP_K=8

export SEARCH_DISPLAY_LIMIT=10
export SEARCH_TOKEN_BUDGET=4096
export MAX_OBS_CHARS=15000
export DOC_SNIPPET_CHARS=120
export CURATED_DOC_CHARS=0
export MAX_TURNS=35
```

`--max-turns 40` is still passed to the evaluator. The `MAX_TURNS=35`
environment variable is used by prompt and reward-shaping constants imported
from the training harness.

## Run BrowseComp+

Choose how many queries to run:

```bash
N_QUERIES=10
SEED=42
RUN_DIR=tmp/harness1_vllm_bcplus
mkdir -p "$RUN_DIR/trajectories"
```

A small `N_QUERIES` value is best for a smoke test. Larger values give more
stable aggregate metrics.

Run the evaluation:

```bash
export SAVE_TRAJECTORIES=1
export SAVE_FULL_TRAJECTORIES=1
export TRAJECTORY_SAVE_PATH="$RUN_DIR/trajectories"

PYTHONPATH=. uv run python inference/evaluate_harness1_vllm.py \
  --dataset browsecompplus \
  --split test \
  --collection-split test \
  --n-queries "$N_QUERIES" \
  --seed "$SEED" \
  --max-turns 40 \
  --temperature 1.0 \
  --max-tokens 2048 \
  --parallel 2 \
  --base-url http://127.0.0.1:8000/v1 \
  --model harness-1 \
  --partial-output "$RUN_DIR/partial_results.jsonl" \
  --output "$RUN_DIR/eval_results.json"
```

The evaluator samples query IDs from the BrowseComp+ test split using `--seed`.
Pass `--query-ids ...` to evaluate a fixed list instead.

## Monitor Progress

Summarize completed partial results:

```bash
python - <<'PY'
from pathlib import Path
import json
import statistics

run_dir = Path("tmp/harness1_vllm_bcplus")
path = run_dir / "partial_results.jsonl"
rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
print("completed", len(rows))
for key in ["recall", "final_answer_recall", "trajectory_recall", "precision", "turns"]:
    vals = [float(row.get(key, 0.0) or 0.0) for row in rows]
    print(
        key,
        "mean",
        round(sum(vals) / max(len(vals), 1), 4),
        "median",
        round(statistics.median(vals), 4) if vals else None,
    )
print("errors", sum(1 for row in rows if row.get("error")))
PY
```

Check GPU utilization:

```bash
nvidia-smi
```

## Read Final Results

When the run completes, `eval_results.json` contains per-query results plus a
top-level `_summary` object:

```bash
python - <<'PY'
from pathlib import Path
import json

path = Path("tmp/harness1_vllm_bcplus/eval_results.json")
data = json.loads(path.read_text())
print(json.dumps(data["_summary"], indent=2))
PY
```

Key metrics:

- `recall`: recall of evidence documents in the final curated set.
- `final_answer_recall`: recall over evidence tied to the final answer.
- `trajectory_recall`: evidence recall anywhere in the trajectory.
- `precision`: precision of the final curated set.
- `errors`: number of episodes that ended with a harness error.

Expected values vary with the query sample, reranker backend, Chroma index, vLLM
version, and GPU kernels. A small smoke test has high variance; use larger query
sets for stable reporting.

## Glossary

- Harness-1 operating point: the component flags and generation settings used for
  the full search harness, including curation, verification, evidence graph, and
  token-budget controls.
- BrowseComp+: a benchmark for browsing and evidence-seeking questions.
- qrels: relevance labels that map query IDs to gold/evidence document IDs.
- Curated evidence recall: how much gold evidence appears in the final curated
  document set.
- Trajectory recall: how much gold evidence appears anywhere during the search
  trajectory, even if it is not kept in the final set.
- Raw `/v1/completions`: the OpenAI-compatible completion endpoint used with
  pre-tokenized prompts.
- Integer token prompts: prompt inputs sent as token IDs rather than plain text.
- `V8D_` flags: environment flags that enable Harness-1 search components.

## Common Issues

### vLLM does not support `GptOssForCausalLM`

Use the repository environment and confirm the vLLM version:

```bash
uv sync --extra vllm
uv run python - <<'PY'
import vllm
print(vllm.__version__)
PY
```

### FlashInfer CUTLASS fails because `nvcc` is missing

Use:

```bash
--moe-backend triton
```

### DeepGEMM is unavailable or outdated

Set:

```bash
export VLLM_USE_DEEP_GEMM=0
export VLLM_MOE_USE_DEEP_GEMM=0
```

### `/v1/completions` returns text but no token IDs

Use a newer vLLM build and rerun the raw completion smoke test. The evaluator
requires token IDs to reconstruct the exact Harness-1 action tokens.

### Evaluation metrics are far below expectation

Check the following first:

- The full `V8D_` operating-point flags are set in the evaluation shell.
- The BrowseComp+ Chroma collection and qrel document IDs match.
- Optional reranker credentials point to the intended reranker backend.
- The run completed the requested number of queries instead of reporting a
  partial average.
- The evaluator is using raw `/v1/completions`, not chat completions.

## Limitations

- The full BrowseComp+ evaluation requires a compatible Chroma retrieval backend;
  the large corpus index is not bundled in this repository.
- Results can vary with external retrieval and reranking services.
- Non-H100 GPUs may work with vLLM if they have enough memory and kernel support,
  but the commands here were validated on H100-class hardware.
