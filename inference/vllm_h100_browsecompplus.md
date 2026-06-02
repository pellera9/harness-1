# H100 vLLM Inference Instructions

This guide describes how to serve the released HarnesS-1 Hugging Face checkpoint
with vLLM on local H100 GPUs and run the BrowseComp+ evaluation through the same
multi-turn search harness included in this repository.

The goal is to validate that the merged Hugging Face checkpoint
`pat-jj/harness-1` runs the HarnesS-1 operating point on BrowseComp+ through the
public vLLM evaluation entrypoint.

## What To Run

Use the repository root as the working directory:

```bash
cd /path/to/harness-1
```

The model to serve is:

```bash
export HARNESS1_HF_MODEL=pat-jj/harness-1
```

The parity evaluation should use:

- `temperature=1.0`
- `max_tokens=2048`
- `max_turns=40`
- BrowseComp+ test split
- the full HarnesS-1 component set enabled
- the raw vLLM `/v1/completions` API, not chat completions

## Environment

Use Python 3.11+ and install dependencies:

```bash
uv sync --extra vllm
```

If not using `uv`, install the equivalent Python dependencies plus a recent
vLLM build with GPT-OSS support. On H100s, prefer a recent vLLM release that
supports GPT-OSS and the OpenAI-compatible completions API with token-id prompts.

Export credentials and service configuration:

```bash
set -a
source .env.local
set +a

export HF_TOKEN="${HUGGINGFACE_TOKEN:-$HF_TOKEN}"
export PYTHONPATH=.
```

The evaluation requires Chroma/OpenAI/reranker configuration exactly as normal
HarnesS-1 evaluation does. At minimum, make sure `.env.local` contains working
values for the BrowseComp+ Chroma collections, OpenAI embeddings/search support,
and `BASETEN_API_KEY` if reranking is enabled.

## Serve HarnesS-1 With vLLM

Start vLLM on the H100 machine. For 1 H100:

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve "$HARNESS1_HF_MODEL" \
  --served-model-name harness-1 \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 1 \
  --max-model-len 32768 \
  --max-num-batched-tokens 16384 \
  --kv-cache-dtype fp8 \
  --trust-remote-code
```

For 2 or more H100s, increase tensor parallelism:

```bash
CUDA_VISIBLE_DEVICES=0,1 vllm serve "$HARNESS1_HF_MODEL" \
  --served-model-name harness-1 \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 2 \
  --max-model-len 32768 \
  --max-num-batched-tokens 16384 \
  --kv-cache-dtype fp8 \
  --trust-remote-code
```

If startup fails because a particular vLLM version does not accept one of these
flags, drop only that serving optimization flag and keep the model, served model
name, tensor parallel size, and max model length.

Wait until the server is healthy:

```bash
curl http://localhost:8000/health
```

## Raw Completion Smoke Test

The parity harness sends pre-tokenized Harmony prompt tokens to
`/v1/completions`, so a plain chat smoke test is not sufficient. Use the
repository's vLLM eval path for the real check. A lightweight server check is:

```bash
python - <<'PY'
import json
import urllib.request

payload = {
    "model": "harness-1",
    "prompt": "Say OK.",
    "max_tokens": 8,
    "temperature": 0.0,
    "stream": False,
}
req = urllib.request.Request(
    "http://localhost:8000/v1/completions",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urllib.request.urlopen(req, timeout=120) as resp:
    print(resp.read().decode("utf-8"))
PY
```

The real evaluation below additionally requires vLLM to accept integer token
arrays as `prompt` and to return generated token IDs when
`return_token_ids=true`.

## BrowseComp+ Parity Eval

Run this from the public `harness-1` repository root. The vLLM evaluator is
`inference/evaluate_harness1_vllm.py`; it uses `SlidingWindowSearchEnv` from
`training.train_rl` and sends raw action tokens returned by vLLM directly into
the environment.

Set the exact HarnesS-1 flags:

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
export SAVE_FULL_TRAJECTORIES=0
export SAVE_TRAJECTORIES=1

export SEARCH_DISPLAY_LIMIT=10
export SEARCH_TOKEN_BUDGET=4096
export MAX_OBS_CHARS=15000
export DOC_SNIPPET_CHARS=120
export CURATED_DOC_CHARS=0
export MAX_TURNS=35
```

Run a 10-query smoke/parity pass:

```bash
RUN_DIR=tmp/harness1_vllm_bcplus
mkdir -p "$RUN_DIR/trajectories"
export TRAJECTORY_SAVE_PATH="$RUN_DIR/trajectories"

PYTHONPATH=. uv run python inference/evaluate_harness1_vllm.py \
  --dataset browsecompplus \
  --split test \
  --collection-split test \
  --n-queries 10 \
  --seed 42 \
  --temperature 1.0 \
  --max-turns 40 \
  --max-tokens 2048 \
  --parallel 1 \
  --base-url http://127.0.0.1:8000/v1 \
  --model harness-1 \
  --partial-output "$RUN_DIR/partial_results.jsonl" \
  --output "$RUN_DIR/eval_results.json"
```

For a larger run, increase `--n-queries`. To evaluate an exact fixed set, pass
the IDs directly with `--query-ids 1029 579 ...` instead of `--n-queries`.

## Why This Matches The Tinker Harness

`inference/evaluate_harness1_vllm.py` is the public vLLM parity bridge:

- it imports `SlidingWindowSearchEnv` from `training.train_rl`
- it uses the same public tool loop as `inference/evaluate_harness1.py`
- it sends action tokens returned by the policy directly into `env.step(...)`
- it uses `temperature=1.0` and `max_tokens=2048`, matching the Tinker eval
- its vLLM backend sends raw token IDs to `/v1/completions`

The vLLM backend is therefore preferable to a chat API wrapper for parity.

## Expected Server Behavior

The vLLM request payload generated by the eval script has this shape:

```json
{
  "model": "harness-1",
  "prompt": [200006, 17360, "... token ids ..."],
  "max_tokens": 2048,
  "temperature": 1.0,
  "top_p": 0.9,
  "stream": false,
  "stop_token_ids": ["... assistant action stop token ids ..."],
  "return_token_ids": true
}
```

The response must include generated token IDs in one of the common vLLM fields:

- `choices[0].token_ids`
- `choices[0].tokens`
- `choices[0].text_token_ids`

If the response only includes text, the harness cannot reliably reconstruct the
exact action tokens.

## Reporting Results

Please report:

- vLLM version
- CUDA driver/runtime version
- GPU model and count
- exact serve command
- whether `/v1/completions` accepts integer token prompts
- whether `return_token_ids=true` returns token IDs
- final result table from the eval stdout
- `tmp/harness1_vllm_bcplus/eval_results.json`
- any server-side traceback or OOM logs

Useful quick checks:

```bash
python - <<'PY'
import torch
import vllm
print("torch", torch.__version__)
print("cuda available", torch.cuda.is_available())
print("gpu count", torch.cuda.device_count())
print("vllm", vllm.__version__)
PY

nvidia-smi
```

## Common Issues

If vLLM does not support GPT-OSS in the installed version, upgrade vLLM first.

If `/v1/completions` rejects token-array prompts, use a newer vLLM build. Do not
switch to chat completions for the parity run unless the eval script is modified
to preserve token-level stop behavior.

If the eval runs but gets all-zero metrics with `Errors: 10/10`, inspect the
tracebacks in stdout. That usually means the model server returned malformed
tokens, omitted token IDs, or hit an internal inference error.

If throughput is slow but not failing, keep the exact eval parameters unchanged
for the parity run. Serving-side optimizations such as tensor parallelism, FP8
KV cache, and CUDA graph compilation are fine as long as they do not alter the
sampling parameters or returned token sequence.
