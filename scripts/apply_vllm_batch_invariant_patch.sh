#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PATCH_FILE="$ROOT/patches/vllm-0.22.1-batch-invariant.patch"
if [[ -z "${VLLM_PYTHON:-}" ]]; then
  if [[ -x "$ROOT/.venv-vllm/bin/python" ]]; then
    VLLM_PYTHON="$ROOT/.venv-vllm/bin/python"
  else
    VLLM_PYTHON="${PYTHON:-python3}"
  fi
fi

if [[ ! -f "$PATCH_FILE" ]]; then
  echo "missing patch: $PATCH_FILE" >&2
  exit 1
fi

SITE_PACKAGES="$(
"$VLLM_PYTHON" - <<'PY'
import site

paths = site.getsitepackages()
if not paths:
    raise SystemExit("could not find site-packages")
print(paths[0])
PY
)"

"$VLLM_PYTHON" - <<'PY'
import importlib.metadata as md

version = md.version("vllm")
if not version.startswith("0.22.1"):
    raise SystemExit(f"expected vllm 0.22.1* before patching, found {version}")
print(f"patching vllm {version}")
PY

patch --forward --batch -p1 -d "$SITE_PACKAGES" < "$PATCH_FILE"

# The unified patch above predates the deterministic token-dim padding work.
# The real fix (env-controlled fixed-M padding via _deterministic_model_pad_target,
# _pad_token_dim / _slice_token_dim, the per-decoder-layer MHC/FFN/MoE padding, and
# the mixed prefill+decode step using max(prefill,decode) rather than the sum) lives
# in full-file snapshots. Overlay them on top of the base patch so the installed
# DeepSeek-V4 model files exactly match the verified deterministic build.
DSV4_DST="$SITE_PACKAGES/vllm/models/deepseek_v4"
if [[ -d "$DSV4_DST" ]]; then
  cp "$ROOT/patches/dsv4-deterministic/attention.py" "$DSV4_DST/attention.py"
  cp "$ROOT/patches/dsv4-deterministic/nvidia/model.py" "$DSV4_DST/nvidia/model.py"
  echo "overlaid deterministic deepseek_v4 attention.py + nvidia/model.py"
else
  echo "WARNING: $DSV4_DST not found; skipped deterministic overlay" >&2
fi

"$VLLM_PYTHON" - <<'PY'
import vllm.envs as envs

assert hasattr(envs, "VLLM_BATCH_INVARIANT")
assert hasattr(envs, "VLLM_DETERMINISTIC_LOGIT_BAND")
assert hasattr(envs, "VLLM_DETERMINISTIC_LOGIT_QUANTUM")
print("vLLM batch-invariant patch is installed")
PY
