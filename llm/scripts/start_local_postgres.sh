#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -f ".env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

POSTGRES_DB="${POSTGRES_DB:-callcenter}"
POSTGRES_USER="${POSTGRES_USER:-callcenter}"
POSTGRES_PORT="${POSTGRES_PORT:-5432}"

LOCAL_ROOT="${LOCAL_POSTGRES_ROOT:-$ROOT_DIR/.local/postgres-root}"
PKG_DIR="${LOCAL_POSTGRES_PKG_DIR:-$ROOT_DIR/.local/postgres_pkgs}"
DATA_DIR="${LOCAL_POSTGRES_DATA_DIR:-$ROOT_DIR/.local/postgres-data}"
RUN_DIR="${LOCAL_POSTGRES_RUN_DIR:-$ROOT_DIR/.local/postgres-run}"
LOG_FILE="${LOCAL_POSTGRES_LOG_FILE:-$ROOT_DIR/.local/postgres.log}"

BIN_DIR="$LOCAL_ROOT/usr/lib/postgresql/16/bin"
POSTGRES_BIN="$BIN_DIR/postgres"
INITDB_BIN="$BIN_DIR/initdb"
PG_CTL_BIN="$BIN_DIR/pg_ctl"
PG_ISREADY_BIN="$BIN_DIR/pg_isready"
PSQL_BIN="$BIN_DIR/psql"

ensure_binaries() {
  if [[ -x "$POSTGRES_BIN" && -x "$INITDB_BIN" && -x "$PG_CTL_BIN" && -x "$PSQL_BIN" ]]; then
    return 0
  fi

  mkdir -p "$PKG_DIR" "$LOCAL_ROOT"
  (
    cd "$PKG_DIR"
    apt download \
      postgresql-16 \
      postgresql-client-16 \
      libllvm17t64 \
      postgresql-common \
      postgresql-client-common
  )

  for deb in "$PKG_DIR"/*.deb; do
    dpkg-deb -x "$deb" "$LOCAL_ROOT"
  done
}

start_cluster() {
  mkdir -p "$RUN_DIR" "$(dirname "$LOG_FILE")"

  if [[ ! -s "$DATA_DIR/PG_VERSION" ]]; then
    rm -rf "$DATA_DIR"
    "$INITDB_BIN" -D "$DATA_DIR" -U "$POSTGRES_USER" -A trust --locale=C.UTF-8 >/dev/null
  fi

  if "$PG_ISREADY_BIN" -h 127.0.0.1 -p "$POSTGRES_PORT" -U "$POSTGRES_USER" >/dev/null 2>&1; then
    return 0
  fi

  "$PG_CTL_BIN" \
    -D "$DATA_DIR" \
    -l "$LOG_FILE" \
    -o "-h 127.0.0.1 -p ${POSTGRES_PORT} -k ${RUN_DIR}" \
    start \
    >/dev/null

  for _ in $(seq 1 30); do
    if "$PG_ISREADY_BIN" -h 127.0.0.1 -p "$POSTGRES_PORT" -U "$POSTGRES_USER" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done

  echo "[start_local_postgres] PostgreSQL did not become ready in time." >&2
  exit 1
}

ensure_database() {
  local escaped_db
  escaped_db="${POSTGRES_DB//\"/\"\"}"
  local exists
  exists="$("$PSQL_BIN" "postgresql://${POSTGRES_USER}@127.0.0.1:${POSTGRES_PORT}/postgres" -tAc "SELECT 1 FROM pg_database WHERE datname = '${escaped_db}'")"
  if [[ "$exists" != "1" ]]; then
    "$PSQL_BIN" "postgresql://${POSTGRES_USER}@127.0.0.1:${POSTGRES_PORT}/postgres" -c "CREATE DATABASE \"${escaped_db}\";" >/dev/null
  fi
}

ensure_binaries
start_cluster
ensure_database

echo "[start_local_postgres] PostgreSQL ready on 127.0.0.1:${POSTGRES_PORT}"
echo "[start_local_postgres] CALL_CENTER_DATABASE_URL=postgresql+psycopg://${POSTGRES_USER}:***@localhost:${POSTGRES_PORT}/${POSTGRES_DB}"
