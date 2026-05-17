# Tinker Inference for HarnesS-1

This document describes the Tinker-backed evaluation setup used for the
BrowseComp+ HarnesS-1 results reported in Table 2 ("Search quality across
benchmarks"). The BrowseComp+ run uses a fixed 100-query test subset and the
same search-agent harness settings as the original Tinker inference run.

## Checkpoint

Use the published sampler checkpoint:

```text
tinker://ed693b03-4126-5b46-92bd-4b888b55234a:train:0/sampler_weights/000029
```

The corresponding training weights path is:

```text
tinker://ed693b03-4126-5b46-92bd-4b888b55234a:train:0/weights/000029
```

Both paths should be published through Tinker before external users run the
evaluation:

```python
import tinker

rest = tinker.ServiceClient().create_rest_client()
for path in [
    "tinker://ed693b03-4126-5b46-92bd-4b888b55234a:train:0/sampler_weights/000029",
    "tinker://ed693b03-4126-5b46-92bd-4b888b55234a:train:0/weights/000029",
]:
    rest.publish_checkpoint_from_tinker_path(path).result()
```

## Harness Equivalence

For this release repo, use:

```bash
python inference/evaluate_harness1.py
```

This is the release-copy equivalent of the original
`eval_sft_ultra_0417.py` evaluator: it uses the same multi-turn
`SlidingWindowSearchEnv` behavior, but imports it from `training.train_rl`
instead of `train_rl_ultra_0417`. Do not rely on script defaults when
reproducing Table 2; pass all parameters explicitly as shown below.

## Environment

Run from the `harness-1` repository root after installing dependencies and
setting `TINKER_API_KEY` and the search/index credentials required by
`harness.config`.

Set the exact HarnesS-1 harness flags:

```bash
export PYTHONPATH=.

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

export SAVE_TRAJECTORIES=1
export SAVE_FULL_TRAJECTORIES=0
```

## Fixed BrowseComp+ Query Set

The example run below uses the first 50 query IDs from the fixed BrowseComp+
test-query set used for the Table 2-style BrowseComp+ evaluation. In the
original workspace, the 100-query source set was saved as
`tmp/browsecomp100_component_ablation_t1/query_ids.json`; this 50-query example
uses its first 50 entries.

```bash
mkdir -p tmp/browsecomp50_table2
cat > tmp/browsecomp50_table2/query_ids.json <<'JSON'
[
  "1029", "579", "751", "605", "638", "169", "681", "1110", "853", "732",
  "689", "610", "912", "869", "1207", "289", "30", "920", "747", "706",
  "653", "885", "1164", "972", "330", "1224", "1211", "985", "893", "934",
  "1257", "193", "469", "509", "1191", "797", "834", "1141", "1097", "830",
  "754", "959", "68", "26", "936", "168", "1065", "1217", "787", "132"
]
JSON
```

## BrowseComp+ Evaluation Command

Run exactly 50 BrowseComp+ test queries against the Tinker sampler checkpoint:

```bash
mkdir -p tmp/tinker_table2_browsecompplus_50/trajectories
export TRAJECTORY_SAVE_PATH=tmp/tinker_table2_browsecompplus_50/trajectories

python - <<'PY' | bash
import json
from pathlib import Path

qids = json.loads(Path("tmp/browsecomp50_table2/query_ids.json").read_text())
cmd = [
    "python", "inference/evaluate_harness1.py",
    "--dataset", "browsecompplus",
    "--split", "test",
    "--collection-split", "test",
    "--seed", "42",
    "--max-turns", "40",
    "--temperature", "1.0",
    "--max-tokens", "2048",
    "--parallel", "1",
    "--checkpoints",
    "harness1=tinker://ed693b03-4126-5b46-92bd-4b888b55234a:train:0/sampler_weights/000029",
    "--output", "tmp/tinker_table2_browsecompplus_50/eval_sft_results.json",
    "--query-ids",
    *qids,
]
print(" ".join(cmd))
PY
```

The Python snippet expands the fixed query list into CLI arguments and pipes the
result into `bash`.

The important inference parameters are:

- Dataset: `browsecompplus`
- Query split: `test`
- Retrieval collection split: `test`
- Query count: `50`
- Query IDs: the fixed list above
- Tinker model path:
  `tinker://ed693b03-4126-5b46-92bd-4b888b55234a:train:0/sampler_weights/000029`
- Temperature: `1.0`
- Max generation tokens per turn: `2048`
- Max turns: `40`
- Parallel episodes: `1`
- Search display limit: `10`
- Search token budget: `4096`
- Max observation chars: `15000`
- Search snippet chars: `120`
- Curated document chars: `0`

## Example 50-Query Result

We ran the 50-query example above and stopped after the 50th completed episode.
The summary was saved to:

```text
tmp/tinker_table2_browsecompplus/eval_sft_results_50_summary.json
```

Observed metrics over those 50 completed BrowseComp+ queries:

- Curated Set Recall (`recall`): `0.5947`
- Final-Answer Recall (`final_answer_recall`): `0.6645`
- Trajectory Recall (`trajectory_recall`): `0.6634`
- Trajectory Final-Answer Recall (`trajectory_fa_recall`): `0.7268`
- Precision: `0.1375`
- Reward: `1.8767`
- Average turns: `36.42`
- Errors: `0/50`
