# Tounsi Call Center LLM

Production-oriented, configurable call-center LLM stack with:

- configurable dataset download
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
- `data/kb/`
- `data/rag/`
- `data/templates/`
- downloaded/raw datasets under `data/tounsi_raw/`

## Main Structure

```text
src/tounsi_llm/
|- config.py       # Paths, runtime, variants, domain config loader
|- data_prep.py    # Download + self-sup/SFT/DPO preparation
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
scripts/generate_data_guide.py
frontend/
docker-compose.yml
```

## Commands

```bash
# Download raw datasets defined in data/config/domain.json
python run_pipeline.py --stage download

# Prepare all training data
python run_pipeline.py --stage prepare

# Self-supervised language adaptation
python run_pipeline.py --stage self-sup

# Supervised fine-tuning
python run_pipeline.py --stage sft

# DPO alignment
python run_pipeline.py --stage dpo

# Evaluate data + inference
python run_pipeline.py --stage eval --eval-model-variant prod --eval-runtime-mode autonomous

# Generate the Word data guide
python run_pipeline.py --stage docs

# Serve API
python run_pipeline.py --stage serve --host 0.0.0.0 --port 8000
```

Full default pipeline:

```bash
python run_pipeline.py --stage all
```

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

The application still works without a database, but production deployments should enable it.

## Reports and Guides

- evaluation JSON: `reports/eval_full.json`
- evaluation Markdown: `reports/eval_full.md`
- data preparation Word guide: `docs/Call_Center_LLM_Data_Guide.docx`
- templates for every data format: `data/templates/`

## Notes

- The vector retriever uses FAISS when available and falls back to NumPy cosine retrieval otherwise.
- Runtime data is split into pending learning candidates, approved SFT examples, approved DPO pairs, ratings, and live admin corrections.
- Live admin corrections can change behavior immediately without retraining.
