# Tounsi Call Center LLM

Production-oriented, configurable call-center LLM stack with:

- configurable dataset download
- dataset manifest/provenance generation
- raw dataset audit before preprocessing
- self-supervised language adaptation
- SFT on business dialogues
- DPO alignment
- FastAPI inference with few-shot reinforcement
- vector RAG with chunking and optional FAISS backend
- persistent conversation memory, ratings, admin corrections, and learning buffers
- optional SQL database persistence for sessions, conversations, feedbacks, ratings, and live corrections
- Angular frontend with model variants and runtime modes

## Real Pipeline

```text
download datasets
-> audit raw data
-> prepare data
-> self-supervised
-> sft
-> dpo
-> evaluate
-> promote / serve
```

Main files to adapt a new business later:

- `data/config/domain.json`
- `data/config/few_shots.jsonl`
- `data/eval/inference_cases.jsonl`
- `data/rag/` (official business corpus for chunking, embeddings, retrieval, and grounding)
- `data/kb/` (structured operational support for tools and mock business state)
- downloaded/raw datasets under `data/tounsi_raw/`

## Main Structure

```text
src/tounsi_llm/
|- config.py       # Paths, runtime, variants, domain config loader
|- data_sources.py # Download + manifest + extraction helpers
|- data_audit.py   # Raw data audit reports
|- data_prep.py    # Self-sup/SFT/DPO preparation
|- training.py     # Self-supervised, SFT, DPO, promote
|- rag.py          # Chunking, embeddings, vector retrieval, optional FAISS
|- tools.py        # Tool registry and business tool handlers
|- memory.py       # Persistent session memory + learning logs
|- corrections.py  # Live admin corrections without retraining
|- storage.py      # Optional SQL persistence layer
|- inference.py    # Few-shot + tools + RAG + memory + runtime modes
|- server.py       # FastAPI API
`- evaluation.py   # Data + inference evaluation with reports

run_pipeline.py
scripts/train.py
frontend/
docker-compose.yml
```

## Commands

```bash
# Download raw datasets defined in data/config/domain.json
python run_pipeline.py --stage download

# Audit raw datasets (reports/data_audit.*)
python run_pipeline.py --stage audit

# Prepare all training data
python run_pipeline.py --stage prepare

# Validate domain/tools/slots/data coherence
python run_pipeline.py --stage validate

# Self-supervised language adaptation
python run_pipeline.py --stage self-sup

# Supervised fine-tuning
python run_pipeline.py --stage sft

# DPO alignment
python run_pipeline.py --stage dpo

# Promote best stage to production adapter
python run_pipeline.py --stage promote --prod-source-variant dpo

# Evaluate data + inference
python run_pipeline.py --stage eval --eval-model-variant prod --eval-runtime-mode autonomous

# Final coherence + GO/NO_GO readiness report
python run_pipeline.py --stage validate

# Serve API
python run_pipeline.py --stage serve --host 0.0.0.0 --port 8000
```

Full default pipeline:

```bash
python run_pipeline.py --stage all
```

Strict run-from-zero recipe (recommended before production cut):

```bash
python run_pipeline.py --stage full --clean-before-run --reset-processed --fail-on-no-go
```

This command now executes: download -> audit -> prepare -> self-sup -> sft -> dpo -> promote -> eval -> validate.

To auto-serve after a successful full run:

```bash
python run_pipeline.py --stage full --clean-before-run --reset-processed --fail-on-no-go --serve-after-promote
```

## Production Recipe (Ubuntu + T4 16GB VRAM)

Recommended order for a stable call-center adaptation run:

```bash
python run_pipeline.py --stage reset --reset-processed
python run_pipeline.py --stage validate
python run_pipeline.py --stage download
python run_pipeline.py --stage audit
python run_pipeline.py --stage prepare

# SSF / self-supervised language adaptation (darja + code-switch + arabizi)
python run_pipeline.py --stage self-sup --self-sup-epochs 2 --self-sup-max-steps 1800 --self-sup-max-seq-len 1024 --self-sup-fresh-adapter

# SFT dialogue adaptation (multi-turn + business behavior)
python run_pipeline.py --stage sft --sft-epochs 3 --sft-max-steps 2400 --sft-max-seq-len 1536

# DPO alignment (safe, non-aggressive)
python run_pipeline.py --stage dpo --dpo-epochs 1 --dpo-max-steps 600 --dpo-beta 0.1

python run_pipeline.py --stage promote --prod-source-variant dpo
python run_pipeline.py --stage eval --eval-model-variant prod --eval-runtime-mode autonomous
python run_pipeline.py --stage validate
```

Default training profile in config is tuned for T4 16GB with non-aggressive LoRA and best-checkpoint retention.

### External SSF Enrichment (104k)

To mirror the large mixed-script SSF strategy, place the external enriched file at:

- `data/tounsi_raw/enrichment/ssf_llm_sentiment_104k.jsonl`

The dataset config is already wired (`ssf_llm_sentiment_104k`) and will be consumed during `--stage prepare` when present.

### Official RAG Integration (Delivery + Lens Catalog)

To integrate external RAG JSONL files without changing their content, run:

```bash
OFFICIAL_RAG_SOURCE_ROOT=/path/to/official_rag python scripts/sync_external_rag.py --overwrite
```

Or pass the source explicitly:

```bash
python scripts/sync_external_rag.py --source-root /path/to/official_rag --overwrite
```

Synced destination:

- `data/rag/external/`

Sync manifest:

- `reports/rag_external_sync_manifest.json`

The retriever and runtime now use `data/rag/` as the official business source for:

- delivery schedule hints by agency/sector (`get_delivery_schedule`)
- lens catalog lookup by code/name/material/diameter (`lookup_lens_catalog`)
- stronger availability and reference confirmation behavior
- train augmentation for SSF/SFT grounding

## Training Metrics and Convergence Logs

During training, each stage now writes:

- `reports/self_sup_metrics.json`
- `reports/sft_metrics.json`
- `reports/dpo_metrics.json`
- `reports/self_sup_train_log.jsonl`
- `reports/sft_train_log.jsonl`
- `reports/dpo_train_log.jsonl`

These files include convergence signals such as best eval loss, eval perplexity, train/eval final losses, and best checkpoint path.

Promote one adapter as production:

```bash
python run_pipeline.py --stage promote --prod-source-variant sft
python run_pipeline.py --stage promote --prod-source-variant dpo
```

## Runtime Modes

- `collect_execute`: collect fields, structure actions, keep output operational
- `speak`: natural conversation with controlled execution
- `autonomous`: natural conversation with maximum autonomy allowed by policy

## API

- `POST /chat`
- `POST /chat/recommendation`
- `GET /health`
- `GET /models`
- `GET /tools`
- `POST /reset`
- `POST /feedback`
- `POST /rating`
- `POST /admin/corrections`

`/chat` body:

```json
{
  "message": "قداش سوم البلارات 1.67؟",
  "session_id": "my-session-id",
  "model_variant": "prod",
  "runtime_mode": "autonomous"
}
```

`model_variant` can be `prod` or `dpo`.
`runtime_mode` can be `collect_execute`, `speak`, or `autonomous`.

## Docker / Database

```bash
copy .env.example .env
docker compose up -d
```

Then set:

```bash
CALL_CENTER_DATABASE_URL=postgresql+psycopg://callcenter:callcenter@localhost:5432/callcenter
```

### VM One-Command Bootstrap (DB + Training + Promote + Serve)

Linux VM:

```bash
bash scripts/vm_full_prod.sh
```

Windows VM/host:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/vm_full_prod.ps1
```

The bootstrap scripts do the following:

- start PostgreSQL/Adminer with Docker
- run `run_pipeline.py --stage full --fail-on-no-go`
- launch API server in production mode

You can override behavior through `.env`:

- `PIPELINE_STAGE` (default `full`)
- `SERVE_AFTER` (`1` or `0`)
- `CALL_CENTER_API_HOST`, `CALL_CENTER_API_PORT`

The application still works without a database, but production readiness validation now marks DB as required (`required_for_production=true` in domain config).

## Reports and Guides

- evaluation JSON: `reports/eval_full.json`
- evaluation Markdown: `reports/eval_full.md`
- validation JSON: `reports/validation_report.json`
- validation Markdown: `reports/validation_report.md`
- push summary (fr): `CHANGEMENT_AHMED.md`

## Notes

- The vector retriever uses FAISS when available and falls back to NumPy cosine retrieval otherwise.
- `data/rag/` is the official source for business retrieval and grounding; `data/kb/` remains structured support for tools, mocks, and operational fallbacks.
- Runtime data is split into pending learning candidates, approved SFT examples, approved DPO pairs, ratings, and live admin corrections.
- Live admin corrections can change behavior immediately without retraining.
- Chat and recommendation flows now keep endpoint-separated sessions (`chat-*` and `reco-*`) and recommendation conversations are persisted in both history files and PostgreSQL with endpoint metadata.
