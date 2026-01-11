#!/bin/bash
# Double-click to launch the UI using your system Python & already-installed deps.
# No virtualenv creation or pip installs.

cd "$(dirname "$0")"

# Add common gcloud locations to PATH (Finder launches non-login shells)
if [ -f "/usr/local/Caskroom/google-cloud-sdk/latest/google-cloud-sdk/path.zsh.inc" ]; then
  source "/usr/local/Caskroom/google-cloud-sdk/latest/google-cloud-sdk/path.zsh.inc"
elif [ -f "/opt/homebrew/Caskroom/google-cloud-sdk/latest/google-cloud-sdk/path.zsh.inc" ]; then
  source "/opt/homebrew/Caskroom/google-cloud-sdk/latest/google-cloud-sdk/path.zsh.inc"
elif [ -f "$HOME/google-cloud-sdk/path.zsh.inc" ]; then
  source "$HOME/google-cloud-sdk/path.zsh.inc"
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"

RESTORER_USE_SYSTEM_PY=1 PYTHON_BIN="$PYTHON_BIN" "$PYTHON_BIN" restore_batch.py --ui
