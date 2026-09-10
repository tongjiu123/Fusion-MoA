#!/usr/bin/env bash
set -euo pipefail

container=${D12_CONTAINER:-fusion-moa-d12-bf16-tp2-128k2-mtp5-prod}
image=${METAX_VLLM_IMAGE:?Set METAX_VLLM_IMAGE to a compatible custom MetaX vLLM image}
model_root=${FUSION_MODEL_ROOT:-/opt/fusion-moa/models}
revision=1eee80aef82e96900a3d3aa88c302e0dbf0e5e49

if docker inspect "$container" >/dev/null 2>&1; then
  echo "container already exists: $container" >&2
  exit 73
fi
if [ ! -f "$model_root/D12-BF16/$revision/config.json" ]; then
  echo "D12 snapshot is incomplete" >&2
  exit 66
fi

exec docker run -d \
  --name "$container" \
  --restart unless-stopped \
  --network host \
  --ipc host \
  --shm-size 100g \
  --device /dev/dri \
  --device /dev/mxcd \
  --group-add video \
  --security-opt label=disable \
  -e CUDA_VISIBLE_DEVICES=1,7 \
  -e TZ=Asia/Shanghai \
  -v "$model_root:/models:ro" \
  "$image" \
  /opt/conda/bin/vllm serve "/models/D12-BF16/$revision" \
    --host 127.0.0.1 \
    --port 8001 \
    --served-model-name Jackrong/Qwopus3.6-27B-Coder-MTP \
    --trust-remote-code \
    --dtype bfloat16 \
    --tensor-parallel-size 2 \
    --max-model-len 131072 \
    --max-num-seqs 2 \
    --max-num-batched-tokens 4096 \
    --gpu-memory-utilization 0.95 \
    --kv-cache-dtype auto \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --scheduler-reserve-full-isl \
    --enforce-eager \
    --language-model-only \
    --generation-config vllm \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --speculative-config '{"method":"mtp","num_speculative_tokens":5}'
