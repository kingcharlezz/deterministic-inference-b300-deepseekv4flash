#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fnmatch
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = "deepseek-ai/DeepSeek-V4-Flash"


def default_python_for_engine(engine: str) -> str:
    env_name = "SGLANG_PYTHON" if engine == "sglang" else "VLLM_PYTHON"
    if os.environ.get(env_name):
        return os.environ[env_name]
    venv_name = ".venv" if engine == "sglang" else ".venv-vllm"
    candidate = REPO_ROOT / venv_name / "bin" / "python"
    if candidate.exists():
        return str(candidate)
    return sys.executable


@dataclass(frozen=True)
class Variant:
    name: str
    engine: str
    port: int
    env: dict[str, str]
    backend: str
    base_url: str

    @property
    def script(self) -> Path:
        if self.engine == "sglang":
            return REPO_ROOT / "scripts" / "serve_sglang_deepseek_v4_flash_8xb200.sh"
        if self.engine == "vllm":
            return REPO_ROOT / "scripts" / "serve_vllm_deepseek_v4_flash_8xb200.sh"
        raise AssertionError(f"unsupported engine: {self.engine}")


def sglang_variants() -> list[Variant]:
    base = {"PORT": "30000", "TP": "8", "MODEL": DEFAULT_MODEL}
    return [
        Variant("sglang-dsv4-mxfp4-mem090", "sglang", 30000, {**base, "ATTENTION_BACKEND": "dsv4", "MOE_RUNNER_BACKEND": "flashinfer_mxfp4", "MEM_FRACTION_STATIC": "0.90"}, "sglang-native", "http://127.0.0.1:30000"),
        Variant("sglang-dsv4-mxfp4-bf16-mem090", "sglang", 30000, {**base, "ATTENTION_BACKEND": "dsv4", "MOE_RUNNER_BACKEND": "flashinfer_mxfp4", "FLASHINFER_MXFP4_MOE_PRECISION": "bf16", "MEM_FRACTION_STATIC": "0.90"}, "sglang-native", "http://127.0.0.1:30000"),
        Variant("sglang-dsv4-tp4-dp2-mxfp4-mem090", "sglang", 30000, {**base, "TP": "4", "DP_SIZE": "2", "ATTENTION_BACKEND": "dsv4", "MOE_RUNNER_BACKEND": "flashinfer_mxfp4", "MEM_FRACTION_STATIC": "0.90"}, "sglang-native", "http://127.0.0.1:30000"),
        Variant("sglang-dsv4-tp4-dp2-megamoe-w4a8-mem086", "sglang", 30000, {**base, "TP": "4", "DP_SIZE": "2", "ATTENTION_BACKEND": "dsv4", "MOE_RUNNER_BACKEND": "", "MOE_A2A_BACKEND": "megamoe", "MEM_FRACTION_STATIC": "0.86", "SGLANG_OPT_USE_DEEPGEMM_MEGA_MOE": "1", "SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK": "8320"}, "sglang-native", "http://127.0.0.1:30000"),
        Variant("sglang-dsv4-tp4-dp2-megamoe-w4a4-mem086", "sglang", 30000, {**base, "TP": "4", "DP_SIZE": "2", "ATTENTION_BACKEND": "dsv4", "MOE_RUNNER_BACKEND": "", "MOE_A2A_BACKEND": "megamoe", "MEM_FRACTION_STATIC": "0.86", "SGLANG_OPT_USE_DEEPGEMM_MEGA_MOE": "1", "SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK": "8320", "SGLANG_OPT_DEEPGEMM_MEGA_MOE_USE_FP4_ACTS": "1", "SGLANG_OPT_DEEPGEMM_MEGA_MOE_USE_MXF4_KIND": "1"}, "sglang-native", "http://127.0.0.1:30000"),
        Variant("sglang-dsv4-marlin-noradix-mem090", "sglang", 30000, {**base, "ATTENTION_BACKEND": "dsv4", "MOE_RUNNER_BACKEND": "marlin", "MEM_FRACTION_STATIC": "0.90", "DISABLE_RADIX_CACHE": "1"}, "sglang-native", "http://127.0.0.1:30000"),
        Variant("sglang-dsv4-cutlass-noradix-mem090", "sglang", 30000, {**base, "ATTENTION_BACKEND": "dsv4", "MOE_RUNNER_BACKEND": "flashinfer_cutlass", "MEM_FRACTION_STATIC": "0.90", "DISABLE_RADIX_CACHE": "1"}, "sglang-native", "http://127.0.0.1:30000"),
        Variant("sglang-dsv4-cutlass-fp16-noradix-mem090", "sglang", 30000, {**base, "ATTENTION_BACKEND": "dsv4", "MOE_RUNNER_BACKEND": "flashinfer_cutlass", "DTYPE": "float16", "MEM_FRACTION_STATIC": "0.90", "DISABLE_RADIX_CACHE": "1"}, "sglang-native", "http://127.0.0.1:30000"),
        Variant("sglang-dsv4-trtllm-routed-mem090", "sglang", 30000, {**base, "ATTENTION_BACKEND": "dsv4", "MOE_RUNNER_BACKEND": "flashinfer_trtllm_routed", "MEM_FRACTION_STATIC": "0.90"}, "sglang-native", "http://127.0.0.1:30000"),
        Variant("sglang-dsv4-trtllm-routed-mem086-cg128", "sglang", 30000, {**base, "ATTENTION_BACKEND": "dsv4", "MOE_RUNNER_BACKEND": "flashinfer_trtllm_routed", "MEM_FRACTION_STATIC": "0.86", "CUDA_GRAPH_MAX_BS": "128"}, "sglang-native", "http://127.0.0.1:30000"),
        Variant("sglang-dsv4-trtllm-routed-no-autotune", "sglang", 30000, {**base, "ATTENTION_BACKEND": "dsv4", "MOE_RUNNER_BACKEND": "flashinfer_trtllm_routed", "MEM_FRACTION_STATIC": "0.90", "DISABLE_FLASHINFER_AUTOTUNE": "1"}, "sglang-native", "http://127.0.0.1:30000"),
        Variant("sglang-fa3-mem090", "sglang", 30000, {**base, "ATTENTION_BACKEND": "fa3", "MEM_FRACTION_STATIC": "0.90"}, "sglang-native", "http://127.0.0.1:30000"),
        Variant("sglang-flashinfer-mem090", "sglang", 30000, {**base, "ATTENTION_BACKEND": "flashinfer", "MEM_FRACTION_STATIC": "0.90"}, "sglang-native", "http://127.0.0.1:30000"),
        Variant("sglang-triton-mem090", "sglang", 30000, {**base, "ATTENTION_BACKEND": "triton", "MEM_FRACTION_STATIC": "0.90"}, "sglang-native", "http://127.0.0.1:30000"),
        Variant("sglang-dsv4-triton-unquant-noradix-mem082", "sglang", 30000, {**base, "ATTENTION_BACKEND": "dsv4", "MOE_RUNNER_BACKEND": "triton", "QUANTIZATION": "unquant", "MEM_FRACTION_STATIC": "0.82", "DISABLE_RADIX_CACHE": "1"}, "sglang-native", "http://127.0.0.1:30000"),
        Variant("sglang-fa3-no-radix", "sglang", 30000, {**base, "ATTENTION_BACKEND": "fa3", "MEM_FRACTION_STATIC": "0.90", "DISABLE_RADIX_CACHE": "1"}, "sglang-native", "http://127.0.0.1:30000"),
        Variant("sglang-triton-no-radix-tile64", "sglang", 30000, {**base, "ATTENTION_BACKEND": "triton", "MEM_FRACTION_STATIC": "0.90", "DISABLE_RADIX_CACHE": "1", "TRITON_ATTENTION_SPLIT_TILE_SIZE": "64"}, "sglang-native", "http://127.0.0.1:30000"),
        Variant("sglang-triton-no-radix-tile128", "sglang", 30000, {**base, "ATTENTION_BACKEND": "triton", "MEM_FRACTION_STATIC": "0.90", "DISABLE_RADIX_CACHE": "1", "TRITON_ATTENTION_SPLIT_TILE_SIZE": "128"}, "sglang-native", "http://127.0.0.1:30000"),
        Variant("sglang-triton-no-radix-tile256", "sglang", 30000, {**base, "ATTENTION_BACKEND": "triton", "MEM_FRACTION_STATIC": "0.90", "DISABLE_RADIX_CACHE": "1", "TRITON_ATTENTION_SPLIT_TILE_SIZE": "256"}, "sglang-native", "http://127.0.0.1:30000"),
        Variant("sglang-fa3-chunk4096-mem086", "sglang", 30000, {**base, "ATTENTION_BACKEND": "fa3", "MEM_FRACTION_STATIC": "0.86", "CHUNKED_PREFILL_SIZE": "4096"}, "sglang-native", "http://127.0.0.1:30000"),
    ]


def vllm_variants() -> list[Variant]:
    base = {
        "PORT": "8000",
        "TP": "8",
        "MODEL": DEFAULT_MODEL,
        "VLLM_BATCH_INVARIANT": "1",
        "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
        "MAX_MODEL_LEN": "8192",
        "KV_CACHE_DTYPE": "fp8",
    }
    return [
        Variant("vllm-seq256-tok8192-mem090", "vllm", 8000, {**base, "MAX_NUM_SEQS": "256", "MAX_NUM_BATCHED_TOKENS": "8192", "GPU_MEMORY_UTILIZATION": "0.90"}, "openai-completions", "http://127.0.0.1:8000"),
        Variant(
            "vllm-flashmla-tritonunfused-eager-mem084",
            "vllm",
            8000,
            {
                **base,
                "MAX_NUM_SEQS": "256",
                "MAX_NUM_BATCHED_TOKENS": "8192",
                "GPU_MEMORY_UTILIZATION": "0.84",
                "GENERATION_CONFIG": "vllm",
                "ATTENTION_BACKEND": "FLASHMLA_SPARSE_DSV4",
                "MOE_BACKEND": "triton_unfused",
                "ENABLE_PREFIX_CACHING": "0",
                "ASYNC_SCHEDULING": "0",
                "ENFORCE_EAGER": "1",
            },
            "openai-completions",
            "http://127.0.0.1:8000",
        ),
        Variant(
            "vllm-flashmla-tritonunfused-throughput-mem093-cg512",
            "vllm",
            8000,
            {
                **base,
                "MAX_NUM_SEQS": "256",
                "MAX_NUM_BATCHED_TOKENS": "16384",
                "GPU_MEMORY_UTILIZATION": "0.93",
                "GENERATION_CONFIG": "vllm",
                "ATTENTION_BACKEND": "FLASHMLA_SPARSE_DSV4",
                "MOE_BACKEND": "triton_unfused",
                "ENABLE_PREFIX_CACHING": "0",
                "ASYNC_SCHEDULING": "0",
                "PERFORMANCE_MODE": "throughput",
                "MAX_CUDAGRAPH_CAPTURE_SIZE": "512",
                "VLLM_USE_BREAKABLE_CUDAGRAPH": "0",
            },
            "openai-completions",
            "http://127.0.0.1:8000",
        ),
        Variant(
            "vllm-flashmla-tritonunfused-throughput-mem093-cg512-band05",
            "vllm",
            8000,
            {
                **base,
                "MAX_NUM_SEQS": "256",
                "MAX_NUM_BATCHED_TOKENS": "16384",
                "GPU_MEMORY_UTILIZATION": "0.93",
                "GENERATION_CONFIG": "vllm",
                "ATTENTION_BACKEND": "FLASHMLA_SPARSE_DSV4",
                "MOE_BACKEND": "triton_unfused",
                "ENABLE_PREFIX_CACHING": "0",
                "ASYNC_SCHEDULING": "0",
                "PERFORMANCE_MODE": "throughput",
                "MAX_CUDAGRAPH_CAPTURE_SIZE": "512",
                "VLLM_USE_BREAKABLE_CUDAGRAPH": "0",
                "VLLM_DETERMINISTIC_LOGIT_BAND": "0.5",
                "VLLM_DETERMINISTIC_LOGIT_TIEBREAK": "high",
            },
            "openai-completions",
            "http://127.0.0.1:8000",
        ),
        Variant(
            "vllm-flashmla-trtllm-mem084",
            "vllm",
            8000,
            {
                **base,
                "MAX_NUM_SEQS": "256",
                "MAX_NUM_BATCHED_TOKENS": "8192",
                "GPU_MEMORY_UTILIZATION": "0.84",
                "GENERATION_CONFIG": "vllm",
                "ATTENTION_BACKEND": "FLASHMLA_SPARSE_DSV4",
                "MOE_BACKEND": "flashinfer_trtllm",
                "ENABLE_PREFIX_CACHING": "0",
                "ASYNC_SCHEDULING": "0",
            },
            "openai-completions",
            "http://127.0.0.1:8000",
        ),
        Variant(
            "vllm-flashmla-trtllm-throughput-mem093-cg512",
            "vllm",
            8000,
            {
                **base,
                "MAX_NUM_SEQS": "256",
                "MAX_NUM_BATCHED_TOKENS": "16384",
                "GPU_MEMORY_UTILIZATION": "0.93",
                "GENERATION_CONFIG": "vllm",
                "ATTENTION_BACKEND": "FLASHMLA_SPARSE_DSV4",
                "MOE_BACKEND": "flashinfer_trtllm",
                "ENABLE_PREFIX_CACHING": "0",
                "ASYNC_SCHEDULING": "0",
                "PERFORMANCE_MODE": "throughput",
                "MAX_CUDAGRAPH_CAPTURE_SIZE": "512",
                "VLLM_USE_BREAKABLE_CUDAGRAPH": "0",
            },
            "openai-completions",
            "http://127.0.0.1:8000",
        ),
        Variant(
            "vllm-flashinfer-dsv4-trtllm-throughput-mem093-cg512",
            "vllm",
            8000,
            {
                **base,
                "MAX_NUM_SEQS": "256",
                "MAX_NUM_BATCHED_TOKENS": "16384",
                "GPU_MEMORY_UTILIZATION": "0.93",
                "GENERATION_CONFIG": "vllm",
                "ATTENTION_BACKEND": "FLASHINFER_MLA_SPARSE_DSV4",
                "MOE_BACKEND": "flashinfer_trtllm",
                "ENABLE_PREFIX_CACHING": "0",
                "ASYNC_SCHEDULING": "0",
                "PERFORMANCE_MODE": "throughput",
                "MAX_CUDAGRAPH_CAPTURE_SIZE": "512",
                "VLLM_USE_BREAKABLE_CUDAGRAPH": "0",
            },
            "openai-completions",
            "http://127.0.0.1:8000",
        ),
        Variant(
            "vllm-flashinfer-dsv4-trtllm-eager-mem084",
            "vllm",
            8000,
            {
                **base,
                "MAX_NUM_SEQS": "256",
                "MAX_NUM_BATCHED_TOKENS": "8192",
                "GPU_MEMORY_UTILIZATION": "0.84",
                "GENERATION_CONFIG": "vllm",
                "ATTENTION_BACKEND": "FLASHINFER_MLA_SPARSE_DSV4",
                "MOE_BACKEND": "flashinfer_trtllm",
                "ENABLE_PREFIX_CACHING": "0",
                "ASYNC_SCHEDULING": "0",
                "ENFORCE_EAGER": "1",
            },
            "openai-completions",
            "http://127.0.0.1:8000",
        ),
        Variant(
            "vllm-flashmla-humming-throughput-mem093-cg512",
            "vllm",
            8000,
            {
                **base,
                "MAX_NUM_SEQS": "256",
                "MAX_NUM_BATCHED_TOKENS": "16384",
                "GPU_MEMORY_UTILIZATION": "0.93",
                "GENERATION_CONFIG": "vllm",
                "ATTENTION_BACKEND": "FLASHMLA_SPARSE_DSV4",
                "MOE_BACKEND": "humming",
                "ENABLE_PREFIX_CACHING": "0",
                "ASYNC_SCHEDULING": "0",
                "PERFORMANCE_MODE": "throughput",
                "MAX_CUDAGRAPH_CAPTURE_SIZE": "512",
                "VLLM_USE_BREAKABLE_CUDAGRAPH": "0",
            },
            "openai-completions",
            "http://127.0.0.1:8000",
        ),
        Variant(
            "vllm-flashmla-humming-eager-mem084",
            "vllm",
            8000,
            {
                **base,
                "MAX_NUM_SEQS": "256",
                "MAX_NUM_BATCHED_TOKENS": "8192",
                "GPU_MEMORY_UTILIZATION": "0.84",
                "GENERATION_CONFIG": "vllm",
                "ATTENTION_BACKEND": "FLASHMLA_SPARSE_DSV4",
                "MOE_BACKEND": "humming",
                "ENABLE_PREFIX_CACHING": "0",
                "ASYNC_SCHEDULING": "0",
                "ENFORCE_EAGER": "1",
            },
            "openai-completions",
            "http://127.0.0.1:8000",
        ),
        Variant(
            "vllm-flashinfer-dsv4-humming-throughput-mem093-cg512",
            "vllm",
            8000,
            {
                **base,
                "MAX_NUM_SEQS": "256",
                "MAX_NUM_BATCHED_TOKENS": "16384",
                "GPU_MEMORY_UTILIZATION": "0.93",
                "GENERATION_CONFIG": "vllm",
                "ATTENTION_BACKEND": "FLASHINFER_MLA_SPARSE_DSV4",
                "MOE_BACKEND": "humming",
                "ENABLE_PREFIX_CACHING": "0",
                "ASYNC_SCHEDULING": "0",
                "PERFORMANCE_MODE": "throughput",
                "MAX_CUDAGRAPH_CAPTURE_SIZE": "512",
                "VLLM_USE_BREAKABLE_CUDAGRAPH": "0",
            },
            "openai-completions",
            "http://127.0.0.1:8000",
        ),
        Variant(
            "vllm-flashmla-trtllm-throughput-mem093-cg512-band05",
            "vllm",
            8000,
            {
                **base,
                "MAX_NUM_SEQS": "256",
                "MAX_NUM_BATCHED_TOKENS": "16384",
                "GPU_MEMORY_UTILIZATION": "0.93",
                "GENERATION_CONFIG": "vllm",
                "ATTENTION_BACKEND": "FLASHMLA_SPARSE_DSV4",
                "MOE_BACKEND": "flashinfer_trtllm",
                "ENABLE_PREFIX_CACHING": "0",
                "ASYNC_SCHEDULING": "0",
                "PERFORMANCE_MODE": "throughput",
                "MAX_CUDAGRAPH_CAPTURE_SIZE": "512",
                "VLLM_USE_BREAKABLE_CUDAGRAPH": "0",
                "VLLM_DETERMINISTIC_LOGIT_BAND": "0.5",
                "VLLM_DETERMINISTIC_LOGIT_TIEBREAK": "high",
            },
            "openai-completions",
            "http://127.0.0.1:8000",
        ),
        Variant(
            "vllm-flashmla-trtllm-throughput-mem093-cg512-band2",
            "vllm",
            8000,
            {
                **base,
                "MAX_NUM_SEQS": "256",
                "MAX_NUM_BATCHED_TOKENS": "16384",
                "GPU_MEMORY_UTILIZATION": "0.93",
                "GENERATION_CONFIG": "vllm",
                "ATTENTION_BACKEND": "FLASHMLA_SPARSE_DSV4",
                "MOE_BACKEND": "flashinfer_trtllm",
                "ENABLE_PREFIX_CACHING": "0",
                "ASYNC_SCHEDULING": "0",
                "PERFORMANCE_MODE": "throughput",
                "MAX_CUDAGRAPH_CAPTURE_SIZE": "512",
                "VLLM_USE_BREAKABLE_CUDAGRAPH": "0",
                "VLLM_DETERMINISTIC_LOGIT_BAND": "2.0",
                "VLLM_DETERMINISTIC_LOGIT_TIEBREAK": "high",
            },
            "openai-completions",
            "http://127.0.0.1:8000",
        ),
        Variant(
            "vllm-flashmla-trtllm-throughput-mem093-cg512-q05",
            "vllm",
            8000,
            {
                **base,
                "MAX_NUM_SEQS": "256",
                "MAX_NUM_BATCHED_TOKENS": "16384",
                "GPU_MEMORY_UTILIZATION": "0.93",
                "GENERATION_CONFIG": "vllm",
                "ATTENTION_BACKEND": "FLASHMLA_SPARSE_DSV4",
                "MOE_BACKEND": "flashinfer_trtllm",
                "ENABLE_PREFIX_CACHING": "0",
                "ASYNC_SCHEDULING": "0",
                "PERFORMANCE_MODE": "throughput",
                "MAX_CUDAGRAPH_CAPTURE_SIZE": "512",
                "VLLM_USE_BREAKABLE_CUDAGRAPH": "0",
                "VLLM_DETERMINISTIC_LOGIT_QUANTUM": "0.5",
            },
            "openai-completions",
            "http://127.0.0.1:8000",
        ),
        Variant(
            "vllm-tp4dp2-flashmla-trtllm-throughput-mem093-cg512",
            "vllm",
            8000,
            {
                **base,
                "TP": "4",
                "DP": "2",
                "DP_LOCAL": "2",
                "MAX_NUM_SEQS": "256",
                "MAX_NUM_BATCHED_TOKENS": "16384",
                "GPU_MEMORY_UTILIZATION": "0.93",
                "GENERATION_CONFIG": "vllm",
                "ATTENTION_BACKEND": "FLASHMLA_SPARSE_DSV4",
                "MOE_BACKEND": "flashinfer_trtllm",
                "ENABLE_PREFIX_CACHING": "0",
                "ENABLE_CHUNKED_PREFILL": "0",
                "ENABLE_EXPERT_PARALLEL": "1",
                "ENABLE_EP_WEIGHT_FILTER": "1",
                "ASYNC_SCHEDULING": "0",
                "PERFORMANCE_MODE": "throughput",
                "MAX_CUDAGRAPH_CAPTURE_SIZE": "512",
                "VLLM_USE_BREAKABLE_CUDAGRAPH": "0",
            },
            "openai-completions",
            "http://127.0.0.1:8000",
        ),
        Variant("vllm-seq256-tok16384-mem090", "vllm", 8000, {**base, "MAX_NUM_SEQS": "256", "MAX_NUM_BATCHED_TOKENS": "16384", "GPU_MEMORY_UTILIZATION": "0.90"}, "openai-completions", "http://127.0.0.1:8000"),
        Variant("vllm-seq128-tok8192-mem090", "vllm", 8000, {**base, "MAX_NUM_SEQS": "128", "MAX_NUM_BATCHED_TOKENS": "8192", "GPU_MEMORY_UTILIZATION": "0.90"}, "openai-completions", "http://127.0.0.1:8000"),
        Variant("vllm-seq192-tok8192-mem090", "vllm", 8000, {**base, "MAX_NUM_SEQS": "192", "MAX_NUM_BATCHED_TOKENS": "8192", "GPU_MEMORY_UTILIZATION": "0.90"}, "openai-completions", "http://127.0.0.1:8000"),
        Variant("vllm-seq384-tok16384-mem090", "vllm", 8000, {**base, "MAX_NUM_SEQS": "384", "MAX_NUM_BATCHED_TOKENS": "16384", "GPU_MEMORY_UTILIZATION": "0.90"}, "openai-completions", "http://127.0.0.1:8000"),
        Variant("vllm-seq256-tok8192-mem086", "vllm", 8000, {**base, "MAX_NUM_SEQS": "256", "MAX_NUM_BATCHED_TOKENS": "8192", "GPU_MEMORY_UTILIZATION": "0.86"}, "openai-completions", "http://127.0.0.1:8000"),
        Variant("vllm-seq256-tok8192-mem094", "vllm", 8000, {**base, "MAX_NUM_SEQS": "256", "MAX_NUM_BATCHED_TOKENS": "8192", "GPU_MEMORY_UTILIZATION": "0.94"}, "openai-completions", "http://127.0.0.1:8000"),
    ]


def selected_variants(engine_csv: str) -> list[Variant]:
    engines = {item.strip() for item in engine_csv.split(",") if item.strip()}
    variants: list[Variant] = []
    if "sglang" in engines:
        variants.extend(sglang_variants())
    if "vllm" in engines:
        variants.extend(vllm_variants())
    unknown = engines - {"sglang", "vllm"}
    if unknown:
        raise SystemExit(f"unknown engine selection: {', '.join(sorted(unknown))}")
    return variants


def filter_variants(variants: list[Variant], pattern_csv: str | None) -> list[Variant]:
    if not pattern_csv:
        return variants
    patterns = [item.strip() for item in pattern_csv.split(",") if item.strip()]
    selected = [
        variant
        for variant in variants
        if any(fnmatch.fnmatchcase(variant.name, pattern) for pattern in patterns)
    ]
    if not selected:
        raise SystemExit(f"no variants matched --variants={pattern_csv!r}")
    return selected


def wait_for_server(base_url: str, proc: subprocess.Popen[Any], timeout_s: float) -> bool:
    deadline = time.monotonic() + timeout_s
    paths = ["/health", "/v1/models"]
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return False
        for path in paths:
            if proc.poll() is not None:
                return False
            try:
                with urllib.request.urlopen(base_url.rstrip("/") + path, timeout=1) as response:
                    if 200 <= response.status < 500:
                        return True
            except (urllib.error.URLError, TimeoutError):
                pass
        time.sleep(2)
    return False


def stop_process(proc: subprocess.Popen[Any], timeout_s: float = 60) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(timeout=30)


def best_output_tok_s(result_path: Path) -> float:
    if not result_path.exists():
        return 0.0
    data = json.loads(result_path.read_text(encoding="utf-8"))
    rows = data.get("benchmark") or []
    if not rows:
        return 0.0
    return max(float(row.get("output_tok_s") or 0.0) for row in rows)


def run_capture(cmd: list[str], timeout_s: float = 60) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout_s,
        )
        return {
            "command": cmd,
            "exit_code": proc.returncode,
            "output": proc.stdout,
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "command": cmd,
            "exit_code": None,
            "output": str(exc),
        }


def write_environment_snapshot(
    run_dir: Path,
    python: str,
    sglang_python: str,
    vllm_python: str,
) -> None:
    snapshot = {
        "created_at_unix_s": time.time(),
        "python_version": run_capture([python, "--version"]),
        "gpu": run_capture(
            [
                "nvidia-smi",
                "--query-gpu=index,name,driver_version,memory.total",
                "--format=csv",
            ]
        ),
        "packages": run_capture([python, "-m", "pip", "freeze"]),
        "packages_by_engine": {
            "sglang": run_capture([sglang_python, "-m", "pip", "freeze"]),
            "vllm": run_capture([vllm_python, "-m", "pip", "freeze"]),
        },
    }
    (run_dir / "environment.json").write_text(
        json.dumps(snapshot, indent=2) + "\n",
        encoding="utf-8",
    )


def query_gpu_inventory() -> dict[str, Any]:
    return run_capture(
        [
            "nvidia-smi",
            "--query-gpu=index,name,driver_version,memory.total",
            "--format=csv,noheader",
        ]
    )


def gpu_inventory_is_8xb200(snapshot: dict[str, Any]) -> bool:
    if snapshot.get("exit_code") != 0:
        return False
    rows = [line.strip() for line in str(snapshot.get("output", "")).splitlines() if line.strip()]
    return len(rows) == 8 and all("B200" in row for row in rows)


def run_variant(variant: Variant, args: argparse.Namespace, run_dir: Path) -> dict[str, Any]:
    variant_dir = run_dir / variant.name
    variant_dir.mkdir(parents=True, exist_ok=True)
    server_log_path = variant_dir / "server.log"
    bench_log_path = variant_dir / "benchmark.log"
    result_json_path = variant_dir / "result.json"
    result_md_path = variant_dir / "benchmark.md"
    meta: dict[str, Any] = {
        "name": variant.name,
        "engine": variant.engine,
        "backend": variant.backend,
        "base_url": variant.base_url,
        "env": variant.env,
        "server_log": str(server_log_path),
        "benchmark_log": str(bench_log_path),
        "result_json": str(result_json_path),
        "result_markdown": str(result_md_path),
    }

    launch_python = args.sglang_python if variant.engine == "sglang" else args.vllm_python
    benchmark_python = args.benchmark_python or launch_python

    env = os.environ.copy()
    env.update(variant.env)
    env["PYTHON"] = launch_python
    meta["launch_python"] = launch_python
    meta["benchmark_python"] = benchmark_python

    with server_log_path.open("wb") as server_log:
        proc = subprocess.Popen(
            ["bash", str(variant.script)],
            cwd=REPO_ROOT,
            env=env,
            stdout=server_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        meta["server_pid"] = proc.pid
        try:
            if not wait_for_server(variant.base_url, proc, args.startup_timeout_s):
                meta["status"] = "server_failed_to_start"
                meta["server_exit_code"] = proc.poll()
                return meta

            bench_cmd = [
                benchmark_python,
                "benchmark/bench_deterministic_inference.py",
                "--backend",
                variant.backend,
                "--base-url",
                variant.base_url,
                "--model",
                DEFAULT_MODEL,
                "--hardware-label",
                "8xB200",
                "--target-output-tok-s",
                str(args.target_output_tok_s),
                "--misconfig-output-tok-s",
                str(args.misconfig_output_tok_s),
                "--stretch-output-tok-s",
                str(args.stretch_output_tok_s),
                "--json-output",
                str(result_json_path),
                "--markdown-output",
                str(result_md_path),
            ]
            if args.min_requests:
                bench_cmd.extend(["--min-requests", str(args.min_requests)])
            if args.concurrencies:
                bench_cmd.extend(["--concurrencies", args.concurrencies])
            if args.determinism_concurrencies:
                bench_cmd.extend(["--determinism-concurrencies", args.determinism_concurrencies])

            meta["benchmark_command"] = bench_cmd
            with bench_log_path.open("wb") as bench_log:
                bench_proc = subprocess.run(
                    bench_cmd,
                    cwd=REPO_ROOT,
                    env=env,
                    stdout=bench_log,
                    stderr=subprocess.STDOUT,
                    timeout=args.benchmark_timeout_s,
                )
            meta["benchmark_exit_code"] = bench_proc.returncode
            meta["best_output_tok_s"] = best_output_tok_s(result_json_path)
            if bench_proc.returncode == 0:
                meta["status"] = "passed"
            elif meta["best_output_tok_s"] < args.misconfig_output_tok_s:
                meta["status"] = "misconfiguration_or_failed_benchmark"
            else:
                meta["status"] = "below_target"
            return meta
        except subprocess.TimeoutExpired:
            meta["status"] = "benchmark_timeout"
            return meta
        finally:
            stop_process(proc)
            time.sleep(args.cooldown_s)


def write_summary(run_dir: Path, attempts: list[dict[str, Any]]) -> None:
    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps({"attempts": attempts}, indent=2) + "\n", encoding="utf-8")

    rows = [
        "| attempt | engine | status | best output tok/s | result |",
        "|---|---|---|---:|---|",
    ]
    for attempt in attempts:
        best = float(attempt.get("best_output_tok_s") or 0.0)
        rows.append(
            "| "
            + " | ".join(
                [
                    attempt["name"],
                    attempt["engine"],
                    attempt["status"],
                    f"{best:.1f}",
                    attempt["result_json"],
                ]
            )
            + " |"
        )
    (run_dir / "summary.md").write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the deterministic DeepSeek-V4-Flash 8x B200 tuning ladder."
    )
    parser.add_argument("--engines", default="sglang,vllm", help="Comma-separated: sglang,vllm")
    parser.add_argument(
        "--variants",
        default=None,
        help="Optional comma-separated variant name or glob list, for example 'sglang-fa3-*'.",
    )
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--python", default=sys.executable, help="Fallback Python for local utilities.")
    parser.add_argument("--sglang-python", default=default_python_for_engine("sglang"))
    parser.add_argument("--vllm-python", default=default_python_for_engine("vllm"))
    parser.add_argument("--benchmark-python", default=None)
    parser.add_argument("--startup-timeout-s", type=float, default=1800)
    parser.add_argument("--benchmark-timeout-s", type=float, default=3600)
    parser.add_argument("--cooldown-s", type=float, default=20)
    parser.add_argument("--min-requests", type=int, default=256)
    parser.add_argument("--concurrencies", default="1,4,8,16,32,64,128,256")
    parser.add_argument("--determinism-concurrencies", default="1,8,32,128")
    parser.add_argument("--target-output-tok-s", type=float, default=5000)
    parser.add_argument("--misconfig-output-tok-s", type=float, default=2500)
    parser.add_argument("--stretch-output-tok-s", type=float, default=8000)
    parser.add_argument(
        "--skip-gpu-check",
        action="store_true",
        help="Do not fail fast when nvidia-smi cannot verify 8x B200.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    variants = filter_variants(selected_variants(args.engines), args.variants)
    if not variants:
        raise SystemExit("no variants selected")

    run_dir = Path(args.run_dir) if args.run_dir else REPO_ROOT / "runs" / time.strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        print(json.dumps([{"name": item.name, "engine": item.engine, "env": item.env} for item in variants], indent=2))
        return 0

    write_environment_snapshot(run_dir, args.python, args.sglang_python, args.vllm_python)

    if not args.skip_gpu_check:
        gpu_snapshot = query_gpu_inventory()
        if not gpu_inventory_is_8xb200(gpu_snapshot):
            attempts = [
                {
                    "name": "preflight-gpu-inventory",
                    "engine": "host",
                    "status": "gpu_unavailable",
                    "best_output_tok_s": 0.0,
                    "result_json": str(run_dir / "environment.json"),
                    "gpu": gpu_snapshot,
                }
            ]
            write_summary(run_dir, attempts)
            print(json.dumps(attempts[0], indent=2), flush=True)
            print(
                "No variant launched because nvidia-smi did not verify exactly 8 B200 GPUs; "
                f"results in {run_dir}",
                file=sys.stderr,
            )
            return 2

    attempts: list[dict[str, Any]] = []
    for variant in variants:
        print(f"=== running {variant.name} ===", flush=True)
        attempt = run_variant(variant, args, run_dir)
        attempts.append(attempt)
        write_summary(run_dir, attempts)
        print(json.dumps(attempt, indent=2), flush=True)
        if attempt.get("status") == "passed":
            print(f"PASS: {variant.name} reached target; results in {run_dir}", flush=True)
            return 0

    print(f"No variant reached target; results in {run_dir}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
