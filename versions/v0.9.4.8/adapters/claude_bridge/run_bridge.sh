#!/usr/bin/env bash
set -euo pipefail
root=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$root"
export PYTHONPATH="$root${PYTHONPATH:+:$PYTHONPATH}"
export FUSION_TERMINAL_COMPAT=1
export LITELLM_LOCAL_MODEL_COST_MAP=True
export LITELLM_TELEMETRY=False
exec "${LITELLM_BIN:-litellm}" --config "${LITELLM_CONFIG:-$root/litellm.yaml}" \
  --host 127.0.0.1 --port "${BRIDGE_PORT:-19037}" --num_workers 1
