#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-deepseek-ai/DeepSeek-V4-Flash}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
TP="${TP:-8}"
PP="${PP:-}"
DP="${DP:-}"
DP_LOCAL="${DP_LOCAL:-}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
GENERATION_CONFIG="${GENERATION_CONFIG:-}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-}"
MOE_BACKEND="${MOE_BACKEND:-}"
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-}"
ASYNC_SCHEDULING="${ASYNC_SCHEDULING:-}"
ENFORCE_EAGER="${ENFORCE_EAGER:-}"
PERFORMANCE_MODE="${PERFORMANCE_MODE:-}"
ENABLE_CHUNKED_PREFILL="${ENABLE_CHUNKED_PREFILL:-}"
ENABLE_EXPERT_PARALLEL="${ENABLE_EXPERT_PARALLEL:-}"
ENABLE_EP_WEIGHT_FILTER="${ENABLE_EP_WEIGHT_FILTER:-}"
MAX_CUDAGRAPH_CAPTURE_SIZE="${MAX_CUDAGRAPH_CAPTURE_SIZE:-}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
HF_HOME="${HF_HOME:-$PWD/hf-cache}"
TMPDIR="${TMPDIR:-$PWD/.tmp-exec}"
if [[ -z "${CUDA_HOME:-}" && -x /usr/local/cuda/bin/nvcc ]]; then
  CUDA_HOME="/usr/local/cuda"
fi
CUDA_PATH="${CUDA_PATH:-${CUDA_HOME:-}}"
if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x "$PWD/.venv-vllm/bin/python" ]]; then
    PYTHON="$PWD/.venv-vllm/bin/python"
  else
    PYTHON="python3"
  fi
fi

export CUDA_VISIBLE_DEVICES
export HF_HOME
export TMPDIR
if [[ -n "${CUDA_HOME:-}" ]]; then
  export CUDA_HOME
  export PATH="$CUDA_HOME/bin:$PATH"
fi
if [[ -n "${CUDA_PATH:-}" ]]; then
  export CUDA_PATH
fi
export VLLM_BATCH_INVARIANT="${VLLM_BATCH_INVARIANT:-1}"
export VLLM_DETERMINISTIC_LOGIT_QUANTUM="${VLLM_DETERMINISTIC_LOGIT_QUANTUM:-0}"
export VLLM_DETERMINISTIC_LOGIT_BAND="${VLLM_DETERMINISTIC_LOGIT_BAND:-0}"
export VLLM_DETERMINISTIC_LOGIT_TIEBREAK="${VLLM_DETERMINISTIC_LOGIT_TIEBREAK:-high}"
export VLLM_ENABLE_V1_MULTIPROCESSING="${VLLM_ENABLE_V1_MULTIPROCESSING:-0}"
mkdir -p "$TMPDIR"

VLLM_CMD=("${VLLM_BIN:-vllm}" serve)
if ! command -v "${VLLM_CMD[0]}" >/dev/null 2>&1; then
  VLLM_CMD=("$PYTHON" -m vllm.entrypoints.cli.main serve)
fi

EXTRA_ARGS=()
if [[ -n "$GENERATION_CONFIG" ]]; then
  EXTRA_ARGS+=(--generation-config "$GENERATION_CONFIG")
fi
if [[ -n "$ATTENTION_BACKEND" ]]; then
  EXTRA_ARGS+=(--attention-backend "$ATTENTION_BACKEND")
fi
if [[ -n "$PP" ]]; then
  EXTRA_ARGS+=(--pipeline-parallel-size "$PP")
fi
if [[ -n "$DP" ]]; then
  EXTRA_ARGS+=(--data-parallel-size "$DP")
fi
if [[ -n "$DP_LOCAL" ]]; then
  EXTRA_ARGS+=(--data-parallel-size-local "$DP_LOCAL")
fi
if [[ -n "$MOE_BACKEND" ]]; then
  EXTRA_ARGS+=(--moe-backend "$MOE_BACKEND")
fi
if [[ "$ENABLE_PREFIX_CACHING" == "0" ]]; then
  EXTRA_ARGS+=(--no-enable-prefix-caching)
elif [[ "$ENABLE_PREFIX_CACHING" == "1" ]]; then
  EXTRA_ARGS+=(--enable-prefix-caching)
fi
if [[ "$ENABLE_CHUNKED_PREFILL" == "0" ]]; then
  EXTRA_ARGS+=(--no-enable-chunked-prefill)
elif [[ "$ENABLE_CHUNKED_PREFILL" == "1" ]]; then
  EXTRA_ARGS+=(--enable-chunked-prefill)
fi
if [[ "$ENABLE_EXPERT_PARALLEL" == "1" ]]; then
  EXTRA_ARGS+=(--enable-expert-parallel)
fi
if [[ "$ENABLE_EP_WEIGHT_FILTER" == "1" ]]; then
  EXTRA_ARGS+=(--enable-ep-weight-filter)
fi
if [[ "$ASYNC_SCHEDULING" == "0" ]]; then
  EXTRA_ARGS+=(--no-async-scheduling)
elif [[ "$ASYNC_SCHEDULING" == "1" ]]; then
  EXTRA_ARGS+=(--async-scheduling)
fi
if [[ "$ENFORCE_EAGER" == "1" ]]; then
  EXTRA_ARGS+=(--enforce-eager)
fi
if [[ -n "$PERFORMANCE_MODE" ]]; then
  EXTRA_ARGS+=(--performance-mode "$PERFORMANCE_MODE")
fi
if [[ -n "$MAX_CUDAGRAPH_CAPTURE_SIZE" ]]; then
  EXTRA_ARGS+=(--max-cudagraph-capture-size "$MAX_CUDAGRAPH_CAPTURE_SIZE")
fi

"$PYTHON" scripts/preflight_8xb200_deepseek_v4_flash.py \
  --engine vllm \
  --python "$PYTHON"

exec "${VLLM_CMD[@]}" "$MODEL" \
  --host "$HOST" \
  --port "$PORT" \
  --seed 0 \
  --tensor-parallel-size "$TP" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
  --kv-cache-dtype "$KV_CACHE_DTYPE" \
  --served-model-name "$MODEL" \
  "${EXTRA_ARGS[@]}"
