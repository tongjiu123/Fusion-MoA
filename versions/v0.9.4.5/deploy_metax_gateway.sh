#!/usr/bin/env bash
set -euo pipefail

source_root=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
image=${METAX_GATEWAY_IMAGE:?Set METAX_GATEWAY_IMAGE to a compatible MetaX image with the Gateway dependencies}
run_root=${FUSION_GATEWAY_RUN_ROOT:-/var/log/fusion-moa/v0.9.4.5/metax-qual}
container=${FUSION_GATEWAY_CONTAINER:-fusion-moa-v0945-gateway-qual}
port=${FUSION_PORT:-19026}
run_id=${FUSION_RUN_ID:-metax-v0945-shared-evidence-qual}
restart_policy=${FUSION_GATEWAY_RESTART_POLICY:-no}
d12_base_url=${D12_BASE_URL:-http://127.0.0.1:8001/v1}
d12_health_url=${D12_HEALTH_URL:-http://127.0.0.1:8001/health}
d12_model=${V0945_D12_MODEL:-${V0944_D12_MODEL:-Qwen/Qwen3.8-27B}}
executor_timeout_seconds=${V0945_EXECUTOR_TIMEOUT_SECONDS:-${V0944_EXECUTOR_TIMEOUT_SECONDS:-3300}}
executor_max_tokens=${V0945_EXECUTOR_MAX_TOKENS:-${V0944_EXECUTOR_MAX_TOKENS:-8192}}

curl -fsS --max-time 5 "$d12_health_url" >/dev/null
for cell_port in 8002 8004 8006; do
  curl -fsS --max-time 5 "http://127.0.0.1:${cell_port}/health" >/dev/null
done

mkdir -p "$run_root/traces" "$run_root/gateway"

docker run -d \
  --name "$container" \
  --restart "$restart_policy" \
  --network host \
  -e TZ=Asia/Shanghai \
  -e "FUSION_RUN_ID=$run_id" \
  -e "FUSION_PORT=$port" \
  -e FUSION_RUNS_DIR=/runs/gateway \
  -e FUSION_TRACE_PATH=/runs/traces/gateway.jsonl \
  -e V0945_ANALYST_CELLS=D16,D9,D13 \
  -e V0945_PANEL_INTEGRATOR_CELL=D13 \
  -e "D12_BASE_URL=$d12_base_url" \
  -e D16_BASE_URL=http://127.0.0.1:8002/v1 \
  -e D9_BASE_URL=http://127.0.0.1:8004/v1 \
  -e D13_BASE_URL=http://127.0.0.1:8006/v1 \
  -e "V0945_D12_MODEL=$d12_model" \
  -e V0945_D16_MODEL=fusion-moa-d16-kat-coder-v2.5-dev \
  -e V0945_D9_MODEL=TeichAI/Qwen3.6-27B-Fable-5-v1 \
  -e V0945_D13_MODEL=Jackrong/Qwopus3.6-35B-A3B-v1 \
  -e "V0945_EXECUTOR_TIMEOUT_SECONDS=$executor_timeout_seconds" \
  -e "V0945_EXECUTOR_MAX_TOKENS=$executor_max_tokens" \
  -e V0945_OUTPUT_RECOVERY_MAX_TOKENS=512 \
  -e V0945_ANALYST_DEADLINE_SECONDS=120 \
  -e V0945_ANALYST_MAX_TOKENS=768 \
  -v "$source_root:/workspace:ro" \
  -v "$run_root:/runs" \
  -w /workspace \
  "$image" \
  /opt/conda/bin/python fusion_terminal_server_v0945.py
