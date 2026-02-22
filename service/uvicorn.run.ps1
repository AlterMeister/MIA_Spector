$ErrorActionPreference = "Stop"

# Get script directory
$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Path
# Get repo root (parent of script dir)
$REPO_ROOT = Split-Path -Parent $SCRIPT_DIR

# Set PYTHONPATH to include repo root
$env:PYTHONPATH = "$REPO_ROOT"

# Run uvicorn
# We execute python module uvicorn, pointing to app.main:app
# --app-dir tells uvicorn where to look for the app module relative to PYTHONPATH (or current dir)

# Change location to repo root just in case, similar to the sh script
Set-Location -Path $REPO_ROOT

Write-Host "Starting MIA-Spector Backend on port 8080..."
python -m uvicorn app.main:app --app-dir "$SCRIPT_DIR" --host 0.0.0.0 --port 8080 --log-level info
