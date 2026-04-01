# Guide de lancement

## 1. Backend

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python run_pipeline.py --stage serve --host 0.0.0.0 --port 8000
```

## 2. Pipeline d'entrainement

```bash
python run_pipeline.py --stage download
python run_pipeline.py --stage prepare
python run_pipeline.py --stage self-sup
python run_pipeline.py --stage sft
python run_pipeline.py --stage dpo
python run_pipeline.py --stage eval --eval-model-variant prod --eval-runtime-mode autonomous
```

Pipeline complet par defaut:

```bash
python run_pipeline.py --stage all
```

## 3. Promotion d'une version production

```bash
python run_pipeline.py --stage promote --prod-source-variant sft
```

ou

```bash
python run_pipeline.py --stage promote --prod-source-variant dpo
```

## 4. Frontend

```bash
cd frontend
npm install
npm run build
npm start
```

## 5. Endpoints utiles

- `POST /chat`
- `GET /health`
- `GET /models`
- `GET /tools`
- `POST /reset`
- `POST /feedback`
- `POST /rating`
- `POST /admin/corrections`

## 6. Docker + base de donnees

```bash
copy .env.example .env
docker compose up -d
```

Puis definir:

```bash
CALL_CENTER_DATABASE_URL=postgresql+psycopg://callcenter:callcenter@localhost:5432/callcenter
```

## 7. Guide de donnees

```bash
python run_pipeline.py --stage docs
```

Le document Word sera genere dans `docs/Call_Center_LLM_Data_Guide.docx`.

## 8. Fichiers a changer plus tard pour adapter le contexte

- `data/config/domain.json`
- `data/config/few_shots.jsonl`
- `data/eval/inference_cases.jsonl`
- `data/templates/`
- `data/kb/`
- `data/rag/`
- datasets telecharges ou locaux
