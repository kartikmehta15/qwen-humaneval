#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_ID:=Qwen/Qwen2.5-Coder-0.5B-Instruct}"
: "${PORT:=8001}"
: "${VLLM_API_KEY:=<RANDOM_PASSWORD>}"

: "${GPU_MEMORY_UTILIZATION:=0.90}"
: "${MAX_MODEL_LEN:=4096}"
: "${TENSOR_PARALLEL_SIZE:=1}"
: "${HOST:=0.0.0.0}"

echo "[vLLM] Serving $MODEL_ID on :$PORT"
exec python -m vllm.entrypoints.openai.api_server \
  --host "$HOST" \
  --port "$PORT" \
  --model "$MODEL_ID" \
  --max-model-len "$MAX_MODEL_LEN" \
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --api-key "$VLLM_API_KEY"
