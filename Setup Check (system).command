#!/bin/bash
# Double-click to run setup checks using system Python & existing deps (no venv/pip).

cd "$(dirname "$0")"

if [ -f "/usr/local/Caskroom/google-cloud-sdk/latest/google-cloud-sdk/path.zsh.inc" ]; then
  source "/usr/local/Caskroom/google-cloud-sdk/latest/google-cloud-sdk/path.zsh.inc"
elif [ -f "/opt/homebrew/Caskroom/google-cloud-sdk/latest/google-cloud-sdk/path.zsh.inc" ]; then
  source "/opt/homebrew/Caskroom/google-cloud-sdk/latest/google-cloud-sdk/path.zsh.inc"
elif [ -f "$HOME/google-cloud-sdk/path.zsh.inc" ]; then
  source "$HOME/google-cloud-sdk/path.zsh.inc"
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"

RESTORER_USE_SYSTEM_PY=1 PYTHON_BIN="$PYTHON_BIN" "$PYTHON_BIN" restore_batch.py --setup-check
