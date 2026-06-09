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

# V1 scheduler no-mix patch: when VLLM_DETERMINISTIC_NO_MIX=1, never co-schedule
# a new prefill with an in-flight decode. A decode token computed in a mixed
# prefill+decode step runs at a different (variable) attention shape than in a
# pure-decode step and drifts; this gate forces every step to be pure-prefill or
# pure-decode -> byte-identical decode across concurrency (validated to conc 100).
# Env-gated, so it is inert unless VLLM_DETERMINISTIC_NO_MIX=1.
SCHED_DST="$SITE_PACKAGES/vllm/v1/core/sched/scheduler.py"
if [[ -f "$SCHED_DST" ]]; then
  cp "$ROOT/patches/dsv4-deterministic/v1/core/sched/scheduler.py" "$SCHED_DST"
  echo "overlaid deterministic no-mix (prefill-priority) v1 scheduler.py"
else
  echo "WARNING: $SCHED_DST not found; skipped scheduler overlay" >&2
fi

# batch_invariant.py: adds the VLLM_DETERMINISTIC_FAST_NCCL escape hatch (skip
# single-channel NCCL determinism settings for throughput). Inert unless that
# env is set; the byte-exact serving config does not set it.
BI_DST="$SITE_PACKAGES/vllm/model_executor/layers/batch_invariant.py"
if [[ -f "$BI_DST" ]]; then
  cp "$ROOT/patches/dsv4-deterministic/model_executor/layers/batch_invariant.py" "$BI_DST"
  echo "overlaid batch_invariant.py (FAST_NCCL knob)"
else
  echo "WARNING: $BI_DST not found; skipped batch_invariant overlay" >&2
fi

# parallel.py: adds VLLM_DETERMINISTIC_ALLOW_CUSTOM_AR — keeps vLLM's custom
# one-shot all-reduce (deterministic fixed rank-order sum) enabled under batch
# invariance instead of force-falling-back to slow single-channel NCCL. This is
# the key throughput unlock for the deterministic decode path on TP8.
PAR_DST="$SITE_PACKAGES/vllm/config/parallel.py"
if [[ -f "$PAR_DST" ]]; then
  cp "$ROOT/patches/dsv4-deterministic/config/parallel.py" "$PAR_DST"
  echo "overlaid parallel.py (ALLOW_CUSTOM_AR knob)"
else
  echo "WARNING: $PAR_DST not found; skipped parallel.py overlay" >&2
fi

"$VLLM_PYTHON" - <<'PY'
import vllm.envs as envs

assert hasattr(envs, "VLLM_BATCH_INVARIANT")
assert hasattr(envs, "VLLM_DETERMINISTIC_LOGIT_BAND")
assert hasattr(envs, "VLLM_DETERMINISTIC_LOGIT_QUANTUM")
print("vLLM batch-invariant patch is installed")
PY
