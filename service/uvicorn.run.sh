#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"   # .../MIA-Spector/service
REPO_ROOT="$(dirname "$SCRIPT_DIR")"          # .../MIA-Spector

cd "$REPO_ROOT"                               # Switch to project root
export PYTHONPATH="$REPO_ROOT"                # Add root directory to sys.path (to find attacks/ and src/)
python -m uvicorn app.main:app --app-dir "$SCRIPT_DIR" --host 0.0.0.0 --port 8080 --log-level info


# bash service/uvicorn.run.sh

