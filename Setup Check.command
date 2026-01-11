#!/bin/bash
# Double-click to run GCP readiness checks (no API charges).

cd "$(dirname "$0")"

# Add common gcloud locations to PATH if present (Finder launches non-login shells)
if [ -f "/usr/local/Caskroom/google-cloud-sdk/latest/google-cloud-sdk/path.zsh.inc" ]; then
  source "/usr/local/Caskroom/google-cloud-sdk/latest/google-cloud-sdk/path.zsh.inc"
elif [ -f "/opt/homebrew/Caskroom/google-cloud-sdk/latest/google-cloud-sdk/path.zsh.inc" ]; then
  source "/opt/homebrew/Caskroom/google-cloud-sdk/latest/google-cloud-sdk/path.zsh.inc"
elif [ -f "$HOME/google-cloud-sdk/path.zsh.inc" ]; then
  source "$HOME/google-cloud-sdk/path.zsh.inc"
fi

# Use existing env if RESTORER_USE_SYSTEM_PY=1
if [ "${RESTORER_USE_SYSTEM_PY:-0}" = "1" ]; then
  PYTHON_BIN="${PYTHON_BIN:-python3}"
else
  if [ ! -d ".venv" ]; then
    python3 -m venv .venv
  fi
  source .venv/bin/activate
  PYTHON_BIN=python
  pip install -r requirements.txt >/dev/null
fi

$PYTHON_BIN restore_batch.py --setup-check
