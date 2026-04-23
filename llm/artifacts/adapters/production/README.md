# Production Adapter Bundle

This directory is the lightweight production artifact to share across machines without retraining.

Included:
- `adapter_config.json`
- `adapter_model.bin`

Base model:
- `Qwen/Qwen2.5-7B-Instruct`

What is intentionally excluded:
- merged full model weights
- training checkpoints
- optimizer states
- run logs

## Use On Another Machine

1. Clone the repository.
2. Install Git LFS and run `git lfs pull`.
3. Ensure the base model `Qwen/Qwen2.5-7B-Instruct` can be downloaded.
4. Start the backend normally.

The server loads the base model first, then attaches this adapter from `artifacts/adapters/production`.

## Notes

- This adapter is the promoted `production` variant.
- You do not need to rerun training to test it elsewhere.
- GPU and runtime requirements still depend on the base model.
