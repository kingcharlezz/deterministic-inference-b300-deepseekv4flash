#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-deepseek-ai/DeepSeek-V4-Pro}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
TP="${TP:-8}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
HF_HOME="${HF_HOME:-$PWD/hf-cache}"
PYTHON="${PYTHON:-python3}"

export CUDA_VISIBLE_DEVICES
export HF_HOME
export VLLM_BATCH_INVARIANT="${VLLM_BATCH_INVARIANT:-1}"
export VLLM_ENABLE_V1_MULTIPROCESSING="${VLLM_ENABLE_V1_MULTIPROCESSING:-0}"

"$PYTHON" - <<'PY'
import importlib.metadata as md
import os
import shutil
import subprocess

try:
    print(f"vllm={md.version('vllm')}")
except md.PackageNotFoundError as exc:
    raise SystemExit("vllm is not installed. Install requirements or your vLLM wheel first.") from exc

if os.environ.get("VLLM_BATCH_INVARIANT") != "1":
    raise SystemExit("VLLM_BATCH_INVARIANT=1 is required for this deterministic fallback.")

if shutil.which("nvidia-smi") is None:
    raise SystemExit("nvidia-smi is not on PATH; cannot validate the 8x B200 target host.")
try:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        text=True,
        stderr=subprocess.STDOUT,
    )
except subprocess.CalledProcessError as exc:
    raise SystemExit(f"nvidia-smi failed before launch:\n{exc.output}") from exc

gpus = [line.strip() for line in out.splitlines() if line.strip()]
print("visible GPUs:")
for idx, name in enumerate(gpus):
    print(f"  {idx}: {name}")
if len(gpus) != 8 or any("B200" not in name for name in gpus):
    raise SystemExit(f"expected exactly 8 visible NVIDIA B200 GPUs, found {len(gpus)}: {gpus}")
PY

exec vllm serve "$MODEL" \
  --host "$HOST" \
  --port "$PORT" \
  --seed 0 \
  --tensor-parallel-size "$TP" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
  --served-model-name "$MODEL"
