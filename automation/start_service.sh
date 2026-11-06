#!/usr/bin/env bash
# Start the recording + clog-detection service (see automation/service.py).
# Uses the "ovad" conda env and reads OPENAI_API_KEY from ~/.bashrc if it isn't set.
set -e
cd "$(dirname "$0")/.."
source "$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")/etc/profile.d/conda.sh"
conda activate "${OVAD_ENV:-ovad}"
if [ -z "$OPENAI_API_KEY" ]; then
  eval "$(grep -m1 '^export OPENAI_API_KEY=' ~/.bashrc || true)"
fi
[ -n "$OPENAI_API_KEY" ] || echo "warning: OPENAI_API_KEY is not set; analysis will fail" >&2
exec python automation/service.py "$@"
