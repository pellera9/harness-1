# Model Export

This folder contains utilities for merging a locally downloaded Tinker adapter
into the base GPT-OSS model and publishing the resulting full Hugging Face
checkpoint. Private Tinker checkpoint URIs are intentionally not stored in this
repository.

Set the target Hugging Face repository with:

```bash
HARNESS1_HF_MODEL=pat-jj/harness-1
```

## Export Flow

First download your Tinker sampler adapter weights locally using your private
checkpoint URI:

```bash
set -a && source .env.local && set +a
uv run tinker checkpoint download \
  "$HARNESS1_TINKER_CHECKPOINT" \
  --output model_export/adapter \
  --force
```

The downloaded adapter directory must contain:

```text
adapter_model.safetensors
adapter_config.json
```

Merge the adapter into the base GPT-OSS model before publishing the release
checkpoint:

```bash
uv run python model_export/export_tinker_checkpoint_to_hf.py \
  --adapter-path model_export/adapter/YOUR_DOWNLOADED_ADAPTER_DIR \
  --output-path model_export/merged_model
```

To upload the merged model, provide a write-capable Hugging Face token through
the environment and add `--push`:

```bash
uv run python model_export/export_tinker_checkpoint_to_hf.py \
  --adapter-path model_export/adapter/YOUR_DOWNLOADED_ADAPTER_DIR \
  --output-path model_export/merged_model \
  --push
```

After upload, verify standard HF inference:

```bash
uv run python inference/hf_inference.py --model ${HARNESS1_HF_MODEL:-harness-1}
```

Adapter-only uploads are useful for debugging, but the released artifact should
be the merged full model.
