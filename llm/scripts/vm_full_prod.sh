#!/usr/bin/env bash
set -euo pipefail

# VM one-command bootstrap:
# 1) start PostgreSQL via Docker
# 2) run pipeline (download -> train -> promote -> validate by default)
# 3) serve API in production mode

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -f ".env" && -f ".env.example" ]]; then
  cp .env.example .env
  echo "[vm_full_prod] .env created from .env.example"
fi

if [[ -f ".env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

POSTGRES_DB="${POSTGRES_DB:-callcenter}"
POSTGRES_USER="${POSTGRES_USER:-callcenter}"
POSTGRES_PASSWORD="${POSTGRES_PASSWORD:-callcenter}"
POSTGRES_PORT="${POSTGRES_PORT:-5432}"

if [[ -z "${CALL_CENTER_DATABASE_URL:-}" ]]; then
  export CALL_CENTER_DATABASE_URL="postgresql+psycopg://${POSTGRES_USER}:${POSTGRES_PASSWORD}@localhost:${POSTGRES_PORT}/${POSTGRES_DB}"
fi

PIPELINE_STAGE="${PIPELINE_STAGE:-full}"
SERVE_AFTER="${SERVE_AFTER:-1}"
API_HOST="${CALL_CENTER_API_HOST:-0.0.0.0}"
API_PORT="${CALL_CENTER_API_PORT:-8000}"

echo "[vm_full_prod] Starting PostgreSQL and Adminer with Docker..."
docker compose up -d postgres adminer

echo "[vm_full_prod] Waiting for PostgreSQL healthcheck..."
READY=0
for _ in $(seq 1 40); do
  if docker compose exec -T postgres pg_isready -U "$POSTGRES_USER" -d "$POSTGRES_DB" >/dev/null 2>&1; then
    READY=1
    break
  fi
  sleep 2
done

if [[ "$READY" -ne 1 ]]; then
  echo "[vm_full_prod] PostgreSQL did not become healthy in time."
  exit 1
fi

echo "[vm_full_prod] Running pipeline stage '$PIPELINE_STAGE'..."
python run_pipeline.py --stage "$PIPELINE_STAGE" --fail-on-no-go

if [[ "$SERVE_AFTER" != "1" ]]; then
  echo "[vm_full_prod] Pipeline completed (SERVE_AFTER=0)."
  exit 0
fi

echo "[vm_full_prod] Starting API on ${API_HOST}:${API_PORT}"
python run_pipeline.py --stage serve --host "$API_HOST" --port "$API_PORT"
