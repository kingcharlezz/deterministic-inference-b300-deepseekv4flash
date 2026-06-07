#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.metadata as md
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


REQUIRED_SGLANG_FLAGS = [
    "--model-path",
    "--tp",
    "--attention-backend",
    "--enable-deterministic-inference",
    "--mem-fraction-static",
]
OPTIONAL_SGLANG_FLAGS = [
    "--disable-radix-cache",
    "--triton-attention-split-tile-size",
    "--chunked-prefill-size",
]
REQUIRED_VLLM_FLAGS = [
    "--host",
    "--port",
    "--seed",
    "--tensor-parallel-size",
    "--gpu-memory-utilization",
    "--max-model-len",
    "--max-num-seqs",
    "--max-num-batched-tokens",
]


def run_capture(cmd: list[str], timeout_s: float = 60) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout_s,
        )
        return {"command": cmd, "exit_code": proc.returncode, "output": proc.stdout}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"command": cmd, "exit_code": None, "output": str(exc)}


def package_version(name: str) -> str | None:
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return None


def query_gpus() -> dict[str, Any]:
    if shutil.which("nvidia-smi") is None:
        return {"ok": False, "error": "nvidia-smi is not on PATH", "rows": []}
    result = run_capture(
        [
            "nvidia-smi",
            "--query-gpu=index,name,driver_version,memory.total",
            "--format=csv,noheader",
        ]
    )
    rows = [line.strip() for line in result["output"].splitlines() if line.strip()]
    ok = result["exit_code"] == 0 and len(rows) == 8 and all("B200" in row for row in rows)
    return {"ok": ok, "command": result["command"], "exit_code": result["exit_code"], "rows": rows}


def missing_flags(help_text: str, flags: list[str]) -> list[str]:
    return [flag for flag in flags if flag not in help_text]


def check_sglang(python: str) -> dict[str, Any]:
    version = package_version("sglang")
    help_result = run_capture([python, "-m", "sglang.launch_server", "--help"])
    missing_required = missing_flags(help_result["output"], REQUIRED_SGLANG_FLAGS)
    missing_optional = missing_flags(help_result["output"], OPTIONAL_SGLANG_FLAGS)
    return {
        "package": "sglang",
        "version": version,
        "help": help_result,
        "missing_required_flags": missing_required,
        "missing_optional_tuning_flags": missing_optional,
        "ok": version is not None and help_result["exit_code"] == 0 and not missing_required,
    }


def check_vllm(python: str) -> dict[str, Any]:
    version = package_version("vllm")
    command = ["vllm", "serve", "--help"] if shutil.which("vllm") else [python, "-m", "vllm.entrypoints.cli.main", "serve", "--help"]
    help_result = run_capture(command)
    missing_required = missing_flags(help_result["output"], REQUIRED_VLLM_FLAGS)
    batch_invariant = os.environ.get("VLLM_BATCH_INVARIANT") == "1"
    return {
        "package": "vllm",
        "version": version,
        "help": help_result,
        "missing_required_flags": missing_required,
        "vllm_batch_invariant": os.environ.get("VLLM_BATCH_INVARIANT"),
        "ok": (
            version is not None
            and help_result["exit_code"] == 0
            and not missing_required
            and batch_invariant
        ),
    }


def write_report(path: str | None, data: dict[str, Any]) -> None:
    if not path:
        return
    Path(path).write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Preflight 8x B200 DeepSeek-V4-Pro inference host.")
    parser.add_argument("--engine", choices=["sglang", "vllm", "both"], default="both")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--json-output", default=None)
    parser.add_argument("--skip-gpu-check", action="store_true")
    args = parser.parse_args()

    report: dict[str, Any] = {
        "engine": args.engine,
        "python": args.python,
        "checks": {},
    }
    if not args.skip_gpu_check:
        report["checks"]["gpu"] = query_gpus()

    if args.engine in {"sglang", "both"}:
        report["checks"]["sglang"] = check_sglang(args.python)
    if args.engine in {"vllm", "both"}:
        report["checks"]["vllm"] = check_vllm(args.python)

    write_report(args.json_output, report)
    print(json.dumps(report, indent=2))

    failed = [
        name
        for name, check in report["checks"].items()
        if isinstance(check, dict) and check.get("ok") is False
    ]
    if failed:
        print(f"preflight failed: {', '.join(failed)}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
