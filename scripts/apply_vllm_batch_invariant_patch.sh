#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PATCH_FILE="$ROOT/patches/vllm-0.22.1-batch-invariant.patch"

if [[ ! -f "$PATCH_FILE" ]]; then
  echo "missing patch: $PATCH_FILE" >&2
  exit 1
fi

SITE_PACKAGES="$(
python - <<'PY'
import site

paths = site.getsitepackages()
if not paths:
    raise SystemExit("could not find site-packages")
print(paths[0])
PY
)"

python - <<'PY'
import importlib.metadata as md

version = md.version("vllm")
if version != "0.22.1":
    raise SystemExit(f"expected vllm==0.22.1 before patching, found {version}")
print(f"patching vllm {version}")
PY

patch --forward --batch -p1 -d "$SITE_PACKAGES" < "$PATCH_FILE"

python - <<'PY'
import vllm.envs as envs

assert hasattr(envs, "VLLM_BATCH_INVARIANT")
assert hasattr(envs, "VLLM_DETERMINISTIC_LOGIT_BAND")
assert hasattr(envs, "VLLM_DETERMINISTIC_LOGIT_QUANTUM")
print("vLLM batch-invariant patch is installed")
PY
