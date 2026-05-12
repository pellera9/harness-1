# Baseline Evaluation

This folder contains the baseline evaluation entrypoints.

## In-Domain Corpora

Use `eval.py` for Chroma-backed in-domain datasets:

```bash
set -a && source .env.local && set +a
PYTHONPATH=. uv run python inference/baselines/eval.py --help
```

BrowseComp+ is the public path documented in `datagen/README.md`. The `web`,
`sec`, and `patents` corpora are not distributed as ready-made public indexes in
this repository; construct them first with a compatible data-generation and
indexing pipeline, such as
[chroma-core/context-1-data-gen](https://github.com/chroma-core/context-1-data-gen),
then point `.env.local` at your Chroma deployment.

## Transfer Datasets

Use `transfer/web_eval.py` and `transfer/wiki_eval.py` for transfer datasets
that use live web/Wikipedia retrieval tools:

```bash
set -a && source .env.local && set +a
PYTHONPATH=. uv run python inference/baselines/transfer/web_eval.py --help
PYTHONPATH=. uv run python inference/baselines/transfer/wiki_eval.py --help
```

The transfer folder includes the matching `web_tools.py`, `wiki_tools.py`, and
RRF helper used by those runners.
