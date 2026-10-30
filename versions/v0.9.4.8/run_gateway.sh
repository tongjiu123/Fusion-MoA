#!/usr/bin/env bash
set -euo pipefail
root=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$root"
exec "${PYTHON:-python3}" gateway_server.py "$@"
