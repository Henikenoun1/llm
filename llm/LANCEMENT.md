# Guide de lancement

## 1. Backend

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run_pipeline.py --stage serve --host 0.0.0.0 --port 8000
```

## 2. Pipeline d'entrainement

```bash
python run_pipeline.py --stage reset --reset-processed
python run_pipeline.py --stage validate
python run_pipeline.py --stage download
python run_pipeline.py --stage audit
python run_pipeline.py --stage prepare
python run_pipeline.py --stage self-sup --self-sup-epochs 2 --self-sup-max-steps 1800 --self-sup-max-seq-len 1024 --self-sup-fresh-adapter
python run_pipeline.py --stage sft --sft-epochs 3 --sft-max-steps 2400 --sft-max-seq-len 1536
python run_pipeline.py --stage dpo --dpo-epochs 1 --dpo-max-steps 600 --dpo-beta 0.1
python run_pipeline.py --stage promote --prod-source-variant dpo
python run_pipeline.py --stage eval --eval-model-variant prod --eval-runtime-mode autonomous
python run_pipeline.py --stage validate
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
- `POST /chat/recommendation`
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

## 7. Bootstrap VM en une commande

Linux VM:

```bash
bash scripts/vm_full_prod.sh
```

Windows:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/vm_full_prod.ps1
```

Variables utiles (dans `.env`):

- `PIPELINE_STAGE` (par defaut `full`)
- `SERVE_AFTER` (par defaut `1`)
- `CALL_CENTER_API_HOST` et `CALL_CENTER_API_PORT`

## 8. Fichiers a changer plus tard pour adapter le contexte

- `data/config/domain.json`
- `data/config/few_shots.jsonl`
- `data/eval/inference_cases.jsonl`
- `data/rag/` pour le contexte metier officiel utilise par chunking, embeddings et retrieval
- `data/kb/` pour les donnees structurees de support outils/suivi
- datasets telecharges ou locaux
