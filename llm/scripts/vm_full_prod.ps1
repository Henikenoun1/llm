$ErrorActionPreference = 'Stop'

# Windows one-command bootstrap:
# 1) Start PostgreSQL via Docker
# 2) Run pipeline (full by default)
# 3) Start API server

$root = Split-Path -Path $PSScriptRoot -Parent
Set-Location $root

if (-not (Test-Path '.env') -and (Test-Path '.env.example')) {
  Copy-Item '.env.example' '.env'
  Write-Host '[vm_full_prod] .env created from .env.example'
}

if (Test-Path '.env') {
  Get-Content '.env' | ForEach-Object {
    if ($_ -and -not $_.StartsWith('#') -and $_.Contains('=')) {
      $parts = $_.Split('=', 2)
      [System.Environment]::SetEnvironmentVariable($parts[0], $parts[1])
    }
  }
}

if (-not $env:POSTGRES_DB) { $env:POSTGRES_DB = 'callcenter' }
if (-not $env:POSTGRES_USER) { $env:POSTGRES_USER = 'callcenter' }
if (-not $env:POSTGRES_PASSWORD) { $env:POSTGRES_PASSWORD = 'callcenter' }
if (-not $env:POSTGRES_PORT) { $env:POSTGRES_PORT = '5432' }
if (-not $env:CALL_CENTER_DATABASE_URL) {
  $env:CALL_CENTER_DATABASE_URL = "postgresql+psycopg://$($env:POSTGRES_USER):$($env:POSTGRES_PASSWORD)@localhost:$($env:POSTGRES_PORT)/$($env:POSTGRES_DB)"
}

if (-not $env:PIPELINE_STAGE) { $env:PIPELINE_STAGE = 'full' }
if (-not $env:SERVE_AFTER) { $env:SERVE_AFTER = '1' }
if (-not $env:CALL_CENTER_API_HOST) { $env:CALL_CENTER_API_HOST = '0.0.0.0' }
if (-not $env:CALL_CENTER_API_PORT) { $env:CALL_CENTER_API_PORT = '8000' }

Write-Host '[vm_full_prod] Starting PostgreSQL and Adminer with Docker...'
docker compose up -d postgres adminer

Write-Host '[vm_full_prod] Waiting for PostgreSQL healthcheck...'
$ready = $false
for ($i = 0; $i -lt 40; $i++) {
  try {
    docker compose exec -T postgres pg_isready -U $env:POSTGRES_USER -d $env:POSTGRES_DB | Out-Null
    if ($LASTEXITCODE -eq 0) {
      $ready = $true
      break
    }
  } catch {
    # keep waiting
  }
  Start-Sleep -Seconds 2
}

if (-not $ready) {
  throw '[vm_full_prod] PostgreSQL did not become healthy in time.'
}

Write-Host "[vm_full_prod] Running pipeline stage '$($env:PIPELINE_STAGE)'..."
python run_pipeline.py --stage $env:PIPELINE_STAGE --fail-on-no-go

if ($env:SERVE_AFTER -ne '1') {
  Write-Host '[vm_full_prod] Pipeline completed (SERVE_AFTER=0).'
  exit 0
}

Write-Host "[vm_full_prod] Starting API on $($env:CALL_CENTER_API_HOST):$($env:CALL_CENTER_API_PORT)"
python run_pipeline.py --stage serve --host $env:CALL_CENTER_API_HOST --port $env:CALL_CENTER_API_PORT
