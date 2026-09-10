#!/usr/bin/env bash
set -euo pipefail

container=${D12_CONTAINER:-fusion-moa-d12-qwen38-bf16-tp2-256k2-mtp5-prod}
image=${METAX_VLLM_IMAGE:?Set METAX_VLLM_IMAGE to a compatible custom MetaX vLLM image}
model_root=${FUSION_MODEL_ROOT:-/opt/fusion-moa/models}
model_dir=${D12_MODEL_DIR:-D12-Qwen3.8-27B-BF16}
served_model=${D12_SERVED_MODEL:-Qwen/Qwen3.8-27B}
visible_devices=${D12_VISIBLE_DEVICES:-1,7}
port=${D12_PORT:-8001}
restart_policy=${D12_RESTART_POLICY:-unless-stopped}
max_model_len=${D12_MAX_MODEL_LEN:-262144}
max_num_seqs=${D12_MAX_NUM_SEQS:-2}

if docker inspect "$container" >/dev/null 2>&1; then
  echo "container already exists: $container" >&2
  exit 73
fi
if [ ! -f "$model_root/$model_dir/config.json" ] || \
   [ ! -f "$model_root/$model_dir/model.safetensors.index.json" ]; then
  echo "Qwen3.8 D12 snapshot is incomplete" >&2
  exit 66
fi

exec docker run -d \
  --name "$container" \
  --restart "$restart_policy" \
  --network host \
  --ipc host \
  --shm-size 100g \
  --device /dev/dri \
  --device /dev/mxcd \
  --group-add video \
  --security-opt label=disable \
  -e "CUDA_VISIBLE_DEVICES=$visible_devices" \
  -e TZ=Asia/Shanghai \
  -v "$model_root:/models:ro" \
  "$image" \
  /opt/conda/bin/vllm serve "/models/$model_dir" \
    --host 127.0.0.1 \
    --port "$port" \
    --served-model-name "$served_model" \
    --trust-remote-code \
    --dtype bfloat16 \
    --tensor-parallel-size 2 \
    --max-model-len "$max_model_len" \
    --max-num-seqs "$max_num_seqs" \
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
    --tool-call-parser qwen3_xml \
    --speculative-config '{"method":"mtp","num_speculative_tokens":5}'
