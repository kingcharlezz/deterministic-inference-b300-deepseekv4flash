#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-deepseek-ai/deepseek-v4-flash}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HOME="${HF_HOME:-$PWD/hf-cache}"
export VLLM_BATCH_INVARIANT="${VLLM_BATCH_INVARIANT:-1}"
export VLLM_ENABLE_V1_MULTIPROCESSING="${VLLM_ENABLE_V1_MULTIPROCESSING:-0}"

python - <<'PY'
import importlib.metadata as md
import os
import vllm.envs as envs

print(f"vllm={md.version('vllm')}")
assert hasattr(envs, "VLLM_BATCH_INVARIANT"), (
    "This vLLM build does not expose VLLM_BATCH_INVARIANT."
)
assert os.environ.get("VLLM_BATCH_INVARIANT") == "1", (
    "VLLM_BATCH_INVARIANT=1 is required for this deterministic run."
)
print("batch-invariant deterministic mode requested")
PY

exec vllm serve "$MODEL" \
  --host "$HOST" \
  --port "$PORT" \
  --seed 0 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.94 \
  --max-model-len 131072 \
  --max-num-seqs 50 \
  --max-num-batched-tokens 32768 \
  --enable-prefix-caching \
  --kv-cache-dtype fp8 \
  --moe-backend triton_unfused \
  --attention-backend TRITON_MLA \
  --served-model-name "$MODEL" \
  --generation-config vllm \
  --async-scheduling \
  --cudagraph-capture-sizes 1 2 4 8 16 24 32 40 48 50 \
  --max-cudagraph-capture-size 50
