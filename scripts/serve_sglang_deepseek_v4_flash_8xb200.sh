#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-deepseek-ai/DeepSeek-V4-Flash}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-30000}"
TP="${TP:-8}"
DP_SIZE="${DP_SIZE:-}"
EP_SIZE="${EP_SIZE:-}"
MOE_DP_SIZE="${MOE_DP_SIZE:-}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-dsv4}"
MOE_RUNNER_BACKEND="${MOE_RUNNER_BACKEND-flashinfer_mxfp4}"
MOE_A2A_BACKEND="${MOE_A2A_BACKEND:-}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.90}"
DTYPE="${DTYPE:-auto}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
HF_HOME="${HF_HOME:-$PWD/hf-cache}"
TMPDIR="${TMPDIR:-$PWD/.tmp-exec}"
if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x "$PWD/.venv/bin/python" ]]; then
    PYTHON="$PWD/.venv/bin/python"
  else
    PYTHON="python3"
  fi
fi

EXTRA_ARGS=()
if [[ "${DISABLE_RADIX_CACHE:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--disable-radix-cache)
fi
if [[ -n "${TRITON_ATTENTION_SPLIT_TILE_SIZE:-}" ]]; then
  EXTRA_ARGS+=(--triton-attention-split-tile-size "$TRITON_ATTENTION_SPLIT_TILE_SIZE")
fi
if [[ -n "${CHUNKED_PREFILL_SIZE:-}" ]]; then
  EXTRA_ARGS+=(--chunked-prefill-size "$CHUNKED_PREFILL_SIZE")
fi
if [[ -n "$MOE_RUNNER_BACKEND" ]]; then
  EXTRA_ARGS+=(--moe-runner-backend "$MOE_RUNNER_BACKEND")
fi
if [[ -n "$MOE_A2A_BACKEND" ]]; then
  EXTRA_ARGS+=(--moe-a2a-backend "$MOE_A2A_BACKEND")
fi
if [[ -n "$DP_SIZE" ]]; then
  EXTRA_ARGS+=(--data-parallel-size "$DP_SIZE")
fi
if [[ -n "$EP_SIZE" ]]; then
  EXTRA_ARGS+=(--expert-parallel-size "$EP_SIZE")
fi
if [[ -n "$MOE_DP_SIZE" ]]; then
  EXTRA_ARGS+=(--moe-data-parallel-size "$MOE_DP_SIZE")
fi
if [[ "${ENABLE_DP_ATTENTION:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--enable-dp-attention)
fi
if [[ -n "${MAX_RUNNING_REQUESTS:-}" ]]; then
  EXTRA_ARGS+=(--max-running-requests "$MAX_RUNNING_REQUESTS")
fi
if [[ -n "${CUDA_GRAPH_MAX_BS:-}" ]]; then
  EXTRA_ARGS+=(--cuda-graph-max-bs "$CUDA_GRAPH_MAX_BS")
fi
if [[ "${DISABLE_CUDA_GRAPH:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--disable-cuda-graph)
fi
if [[ "${DISABLE_FLASHINFER_AUTOTUNE:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--disable-flashinfer-autotune)
fi
if [[ -n "${FLASHINFER_MXFP4_MOE_PRECISION:-}" ]]; then
  EXTRA_ARGS+=(--flashinfer-mxfp4-moe-precision "$FLASHINFER_MXFP4_MOE_PRECISION")
fi
if [[ -n "${FP4_GEMM_BACKEND:-}" ]]; then
  EXTRA_ARGS+=(--fp4-gemm-backend "$FP4_GEMM_BACKEND")
fi
if [[ -n "${FP8_GEMM_BACKEND:-}" ]]; then
  EXTRA_ARGS+=(--fp8-gemm-backend "$FP8_GEMM_BACKEND")
fi
if [[ -n "${QUANTIZATION:-}" ]]; then
  EXTRA_ARGS+=(--quantization "$QUANTIZATION")
fi

export CUDA_VISIBLE_DEVICES
export HF_HOME
export TMPDIR
if [[ -d "${CUDA_HOME:-/usr/local/cuda}" ]]; then
  export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
  export CUDA_PATH="${CUDA_PATH:-$CUDA_HOME}"
  export PATH="$CUDA_HOME/bin:$PATH"
fi
mkdir -p "$TMPDIR"

"$PYTHON" scripts/preflight_8xb200_deepseek_v4_flash.py \
  --engine sglang \
  --python "$PYTHON"

exec "$PYTHON" -m sglang.launch_server \
  --model-path "$MODEL" \
  --host "$HOST" \
  --port "$PORT" \
  --tp "$TP" \
  --attention-backend "$ATTENTION_BACKEND" \
  --dtype "$DTYPE" \
  --enable-deterministic-inference \
  --mem-fraction-static "$MEM_FRACTION_STATIC" \
  "${EXTRA_ARGS[@]}"
