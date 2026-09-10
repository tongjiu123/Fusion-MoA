#!/usr/bin/env bash
# Recreate only the requested missing MT2 model container; never stop another cell.
# Shared serving resources retain v0944 names; a gateway version change does not run this script.
set -euo pipefail

cell="${1:?Usage: bash start_d12_d9.sh d12|d9}"
image="${AMD_VLLM_IMAGE:?Set AMD_VLLM_IMAGE to a compatible custom AMD vLLM image}"
model_root="${FUSION_MODEL_ROOT:-/opt/fusion-moa/models}"
extra=()
case "$cell" in
  d12)
    container="fusion-moa-v0944-mt2-d12-qual"
    model_dir="${D12_MODEL_PATH:-$model_root/Qwen3.8-27B}"
    model_name="Qwen/Qwen3.8-27B"
    devices="0,1,2,3"
    port=8001
    tensor_parallel=4
    context=262144
    extra=(--long-prefill-token-threshold 2048
           --enable-auto-tool-choice --tool-call-parser qwen3_xml
           --reasoning-parser qwen3
           --speculative-config '{"method":"mtp","num_speculative_tokens":5}')
    ;;
  d9)
    container="fusion-moa-v0944-mt2-d9-qual"
    model_dir="${D9_MODEL_PATH:-$model_root/D9-Fable5-W4A16/modelscope-master-20260902}"
    model_name="TeichAI/Qwen3.6-27B-Fable-5-v1"
    devices="5"
    port=8004
    tensor_parallel=1
    context=131072
    ;;
  *)
    printf 'Unknown cell: %s\n' "$cell" >&2
    exit 2
    ;;
esac

exec docker run -d --name "$container" \
  --restart unless-stopped --network host --ipc host --shm-size 64g \
  --device /dev/kfd --device /dev/dri --group-add video \
  --security-opt label=disable \
  -e "HIP_VISIBLE_DEVICES=$devices" \
  -v "$model_dir:/model:ro" \
  --entrypoint vllm "$image" serve /model \
  --host 127.0.0.1 --port "$port" --served-model-name "$model_name" \
  --trust-remote-code --dtype bfloat16 --tensor-parallel-size "$tensor_parallel" \
  --max-model-len "$context" --max-num-seqs 2 --max-num-batched-tokens 4096 \
  --gpu-memory-utilization 0.95 --kv-cache-dtype auto \
  --enable-prefix-caching --enable-chunked-prefill --enforce-eager \
  --language-model-only --generation-config vllm "${extra[@]}"
