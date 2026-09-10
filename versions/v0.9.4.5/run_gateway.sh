#!/usr/bin/env bash
set -euo pipefail

root=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$root"

export FUSION_PORT=${FUSION_PORT:-19025}
# The entry point owns V0945_* > V0944_* > older environment fallbacks.
# Its unchanged defaults are D16,D9,D13 and 3300/8192/512/120/768.

exec python fusion_terminal_server_v0945.py
