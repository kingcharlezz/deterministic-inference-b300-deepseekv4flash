#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -z "${SGLANG_PYTHON:-}" ]]; then
  if [[ -x "$ROOT/.venv/bin/python" ]]; then
    SGLANG_PYTHON="$ROOT/.venv/bin/python"
  else
    SGLANG_PYTHON="${PYTHON:-python3}"
  fi
fi

SITE_PACKAGES="$(
"$SGLANG_PYTHON" - <<'PY'
import site

paths = site.getsitepackages()
if not paths:
    raise SystemExit("could not find site-packages")
print(paths[0])
PY
)"

"$SGLANG_PYTHON" - <<'PY'
import importlib.metadata as md

version = md.version("sglang")
if not version.startswith("0.5.12"):
    raise SystemExit(f"expected sglang 0.5.12* before patching, found {version}")
print(f"patching sglang {version}")
PY

SITE_PACKAGES="$SITE_PACKAGES" "$SGLANG_PYTHON" - <<'PY'
from __future__ import annotations

import os
from pathlib import Path

site_packages = Path(os.environ["SITE_PACKAGES"])


def patch_file(rel: str, replacements: list[tuple[str, str]]) -> None:
    path = site_packages / rel
    text = path.read_text()
    changed = False
    for old, new in replacements:
        if new in text:
            continue
        if old not in text:
            raise SystemExit(f"{path}: expected patch anchor not found")
        text = text.replace(old, new, 1)
        changed = True
    if changed:
        path.write_text(text)
        print(f"patched {rel}")
    else:
        print(f"already patched {rel}")


patch_file(
    "sglang/srt/layers/attention/deepseek_v4_backend.py",
    [
        (
            "from flash_mla import FlashMLASchedMeta",
            "from sgl_kernel.flash_mla import FlashMLASchedMeta",
        ),
        ("from flash_mla import flash_mla", "from sgl_kernel import flash_mla"),
        ("import flash_mla", "from sgl_kernel import flash_mla"),
    ],
)

patch_file(
    "sglang/srt/layers/moe/hash_topk.py",
    [
        (
            "        assert not apply_routed_scaling_factor_on_output, \"not implemented\"\n"
            "        self.tid2eid = nn.Parameter(",
            "        self.apply_routed_scaling_factor_on_output = (\n"
            "            apply_routed_scaling_factor_on_output\n"
            "        )\n"
            "        self.tid2eid = nn.Parameter(",
        ),
        (
            "                router_logits=router_logits,\n",
            "                router_logits=router_logits.float(),\n",
        ),
        (
            "        if is_hip():\n"
            "            topk_weights = topk_weights.to(torch.float32)\n"
            "\n"
            "        topk_ids = topk_ids_logical_to_physical(topk_ids, expert_location_dispatch_info)\n",
            "        if is_hip():\n"
            "            topk_weights = topk_weights.to(torch.float32)\n"
            "        if self.apply_routed_scaling_factor_on_output:\n"
            "            topk_weights = topk_weights * self.routed_scaling_factor\n"
            "\n"
            "        topk_ids = topk_ids_logical_to_physical(topk_ids, expert_location_dispatch_info)\n",
        ),
    ],
)

patch_file(
    "sglang/srt/layers/moe/topk.py",
    [
        (
            "    topk_weights, topk_ids = moe_fused_gate(\n"
            "        gating_output,\n",
            "    topk_weights, topk_ids = moe_fused_gate(\n"
            "        gating_output.float(),\n",
        ),
    ],
)

patch_file(
    "sglang/srt/layers/attention/flashinfer_backend.py",
    [
        (
            "        save_kv_cache=True,\n"
            "    ):\n",
            "        save_kv_cache=True,\n"
            "        **_,\n"
            "    ):\n",
        ),
    ],
)

patch_file(
    "sglang/srt/layers/quantization/fp8.py",
    [
        (
            "            if self.is_fp4_experts and get_moe_runner_backend().is_flashinfer_mxfp4():\n"
            "                # SM100 (Blackwell) -> trtllm-gen path.\n"
            "                # SM90  (Hopper)    -> cutlass mixed-input path (FlashInfer #3084).\n"
            "                if is_sm90_supported() and not is_sm100_supported():\n",
            "            if self.is_fp4_experts and (\n"
            "                get_moe_runner_backend().is_flashinfer_mxfp4()\n"
            "                or get_moe_runner_backend().is_flashinfer_cutlass()\n"
            "            ):\n"
            "                # SM100 (Blackwell) -> trtllm-gen path.\n"
            "                # SM90  (Hopper)    -> cutlass mixed-input path (FlashInfer #3084).\n"
            "                if (\n"
            "                    get_moe_runner_backend().is_flashinfer_cutlass()\n"
            "                    or (is_sm90_supported() and not is_sm100_supported())\n"
            "                ):\n",
        ),
    ],
)

patch_file(
    "sglang/srt/layers/quantization/mxfp4_flashinfer_cutlass_moe.py",
    [
        (
            "        output_dtype = torch.bfloat16\n",
            "        output_dtype = x.dtype\n",
        ),
        (
            "            use_w4_group_scaling=True,\n"
            "            activation_type=ActivationType.Swiglu,\n",
            "            use_w4_group_scaling=True,\n"
            "            use_packed_weights=True,\n"
            "            activation_type=ActivationType.Swiglu,\n",
        ),
    ],
)

patch_file(
    "sglang/srt/layers/quantization/mxfp4_marlin_moe.py",
    [
        (
            "from sglang.srt.utils.common import is_sm90_supported\n",
            "from sglang.srt.utils.common import is_blackwell_supported, is_sm90_supported\n",
        ),
        (
            "        if not is_sm90_supported():\n"
            "            raise RuntimeError(\n"
            "                \"DeepSeekV4 MXFP4 Marlin fallback requires Hopper/SM90 or above.\"\n"
            "            )\n",
            "        if not (is_sm90_supported() or is_blackwell_supported()):\n"
            "            raise RuntimeError(\n"
            "                \"DeepSeekV4 MXFP4 Marlin fallback requires Hopper/SM90 or Blackwell/SM100+.\"\n"
            "            )\n",
        ),
    ],
)
PY

"$SGLANG_PYTHON" - <<'PY'
import py_compile
import site
from pathlib import Path

root = Path(site.getsitepackages()[0])
for rel in [
    "sglang/srt/layers/attention/deepseek_v4_backend.py",
    "sglang/srt/layers/moe/hash_topk.py",
    "sglang/srt/layers/moe/topk.py",
    "sglang/srt/layers/attention/flashinfer_backend.py",
    "sglang/srt/layers/quantization/fp8.py",
    "sglang/srt/layers/quantization/mxfp4_flashinfer_cutlass_moe.py",
    "sglang/srt/layers/quantization/mxfp4_marlin_moe.py",
]:
    py_compile.compile(str(root / rel), doraise=True)
print("SGLang B200 DeepSeek-V4-Flash patch is installed")
PY
