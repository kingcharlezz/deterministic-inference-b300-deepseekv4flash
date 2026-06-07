#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-deepseek-ai/DeepSeek-V4-Pro}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-30000}"
TP="${TP:-8}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-fa3}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.90}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
HF_HOME="${HF_HOME:-$PWD/hf-cache}"
PYTHON="${PYTHON:-python3}"

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

export CUDA_VISIBLE_DEVICES
export HF_HOME

"$PYTHON" - <<'PY'
import importlib.metadata as md
import shutil
import subprocess

try:
    print(f"sglang={md.version('sglang')}")
except md.PackageNotFoundError as exc:
    raise SystemExit("sglang is not installed. Install requirements or your SGLang wheel first.") from exc

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

exec "$PYTHON" -m sglang.launch_server \
  --model-path "$MODEL" \
  --host "$HOST" \
  --port "$PORT" \
  --tp "$TP" \
  --attention-backend "$ATTENTION_BACKEND" \
  --enable-deterministic-inference \
  --mem-fraction-static "$MEM_FRACTION_STATIC" \
  "${EXTRA_ARGS[@]}"
