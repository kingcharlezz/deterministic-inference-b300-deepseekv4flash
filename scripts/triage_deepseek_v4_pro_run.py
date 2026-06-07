#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Pattern:
    code: str
    severity: str
    regex: str
    signal: str
    next_action: str


PATTERNS = [
    Pattern(
        "nvidia_driver_unavailable",
        "blocker",
        r"NVIDIA-SMI has failed|couldn't communicate with the NVIDIA driver",
        "NVIDIA driver is unavailable.",
        "Fix driver/container GPU visibility before inference tuning.",
    ),
    Pattern(
        "wrong_gpu_inventory",
        "blocker",
        r"expected exactly 8 visible NVIDIA B200 GPUs|environment has 8 B200 GPUs: `False`",
        "Host does not expose exactly 8 B200 GPUs.",
        "Fix CUDA_VISIBLE_DEVICES, container device mapping, or target host selection.",
    ),
    Pattern(
        "missing_package",
        "blocker",
        r"ModuleNotFoundError: No module named '(sglang|vllm)'|(sglang|vllm) is not installed",
        "Inference engine package is missing.",
        "Install the matching requirements-sglang.txt or requirements-vllm.txt stack.",
    ),
    Pattern(
        "missing_cli",
        "blocker",
        r"command not found|No such file or directory: 'vllm'",
        "Engine CLI is missing from PATH.",
        "Activate the intended virtual environment or install the engine CLI.",
    ),
    Pattern(
        "unsupported_flag",
        "blocker",
        r"unrecognized arguments?:",
        "Installed engine does not expose required launch flags.",
        "Pin or upgrade the engine version; rerun preflight before loading the model.",
    ),
    Pattern(
        "hf_auth_or_gated_model",
        "blocker",
        r"401 Client Error|403 Client Error|gated repo|Repository Not Found|HF_TOKEN|authentication",
        "Model download/authentication failed.",
        "Set HF_TOKEN with access to deepseek-ai/DeepSeek-V4-Pro and retry.",
    ),
    Pattern(
        "cuda_oom",
        "tuning",
        r"CUDA out of memory|OutOfMemoryError|CUDACachingAllocator",
        "GPU memory pressure or OOM.",
        "Lower memory utilization/static fraction, reduce max model length, or reduce batched tokens.",
    ),
    Pattern(
        "cuda_kernel_or_arch",
        "blocker",
        r"no kernel image is available|invalid device function|illegal memory access",
        "CUDA kernel or architecture incompatibility.",
        "Check CUDA, torch, flashinfer/FA3/Triton versions for B200 support; try another backend.",
    ),
    Pattern(
        "nccl_or_distributed",
        "blocker",
        r"NCCL|torch.distributed|ProcessGroup|connection refused",
        "Tensor-parallel distributed setup failed.",
        "Check all 8 GPUs, NCCL env, shared memory, networking, and TP=8 launch.",
    ),
    Pattern(
        "attention_backend",
        "tuning",
        r"(flashinfer|fa3|Triton attention|attention backend).*(failed|error|exception|unsupported)",
        "Attention backend failed or emitted errors.",
        "Try the next deterministic backend: fa3, flashinfer, then triton.",
    ),
    Pattern(
        "determinism_failure",
        "blocker",
        r"same-prompt determinism failed|order invariance failed|mismatches",
        "Exact text determinism failed.",
        "Keep deterministic mode enabled; try disabling radix cache and Triton split tile sizes.",
    ),
    Pattern(
        "throughput_below_misconfig",
        "tuning",
        r"below 2500|below 2,500|treat as misconfiguration",
        "Throughput is below the misconfiguration threshold.",
        "Verify TP=8, all GPUs active, no CPU fallback, correct model/config, and batching limits.",
    ),
    Pattern(
        "throughput_below_target",
        "tuning",
        r"below target|below 5000|below 5,000",
        "Throughput is deterministic but below target.",
        "Increase concurrency/batched tokens if memory allows; compare SGLang and vLLM variants.",
    ),
]


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return str(exc)


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def add_issue(issues: list[dict[str, Any]], code: str, severity: str, signal: str, next_action: str, source: str) -> None:
    issues.append(
        {
            "code": code,
            "severity": severity,
            "signal": signal,
            "next_action": next_action,
            "source": source,
        }
    )


def scan_text_file(path: Path, issues: list[dict[str, Any]]) -> None:
    text = read_text(path)
    for pattern in PATTERNS:
        if re.search(pattern.regex, text, flags=re.IGNORECASE):
            add_issue(
                issues,
                pattern.code,
                pattern.severity,
                pattern.signal,
                pattern.next_action,
                str(path),
            )


def scan_result_json(path: Path, issues: list[dict[str, Any]]) -> None:
    data = load_json(path)
    if not data:
        return
    determinism = data.get("determinism") or {}
    same_prompt = determinism.get("same_prompt") or []
    order = determinism.get("order") or []
    mismatches = sum(int(row.get("mismatches") or 0) for row in same_prompt + order)
    if mismatches:
        add_issue(
            issues,
            "determinism_failure",
            "blocker",
            f"Determinism result contains {mismatches} mismatches.",
            "Stop performance tuning; debug deterministic engine flags/cache/backend first.",
            str(path),
        )
    rows = data.get("benchmark") or []
    if rows:
        best = max(float(row.get("output_tok_s") or 0.0) for row in rows)
        targets = data.get("targets") or {}
        misconfig = float(targets.get("misconfiguration_below_output_tok_s") or 2500)
        target = float(targets.get("pass_output_tok_s") or 5000)
        if best < misconfig:
            add_issue(
                issues,
                "throughput_below_misconfig",
                "tuning",
                f"Best output throughput {best:.1f} tok/s is below {misconfig:.1f}.",
                "Treat as misconfiguration: verify TP=8, all GPUs active, batching, and backend.",
                str(path),
            )
        elif best < target:
            add_issue(
                issues,
                "throughput_below_target",
                "tuning",
                f"Best output throughput {best:.1f} tok/s is below {target:.1f}.",
                "Tune batching/concurrency/memory and compare SGLang/vLLM variants.",
                str(path),
            )


def scan_summary_json(path: Path, issues: list[dict[str, Any]]) -> None:
    data = load_json(path)
    if not data:
        return
    for attempt in data.get("attempts") or []:
        status = attempt.get("status")
        if status and status != "passed":
            add_issue(
                issues,
                f"attempt_{status}",
                "info",
                f"Variant {attempt.get('name')} ended with status {status}.",
                "Open that variant's server.log and benchmark.log; use pattern findings for the next action.",
                str(path),
            )


def scan_pipeline_json(path: Path, issues: list[dict[str, Any]]) -> None:
    data = load_json(path)
    if not data:
        return
    selected = data.get("selected_engines")
    if selected == []:
        add_issue(
            issues,
            "no_engine_passed_preflight",
            "blocker",
            "No selected engine passed preflight.",
            "Inspect preflight-*.json/log, then fix GPU visibility or engine installation before tuning.",
            str(path),
        )
    for item in data.get("preflight") or []:
        if not item.get("ok"):
            add_issue(
                issues,
                f"preflight_failed_{item.get('engine')}",
                "blocker",
                f"Preflight failed for {item.get('engine')}.",
                "Inspect the matching preflight JSON/log before loading the model.",
                str(path),
            )


def scan_preflight_json(path: Path, issues: list[dict[str, Any]]) -> None:
    data = load_json(path)
    if not data:
        return
    gpu = (data.get("checks") or {}).get("gpu") or {}
    if gpu.get("ok") is False:
        rows = "\n".join(gpu.get("rows") or [])
        if "NVIDIA-SMI has failed" in rows or "communicate with the NVIDIA driver" in rows:
            add_issue(
                issues,
                "nvidia_driver_unavailable",
                "blocker",
                "NVIDIA driver is unavailable.",
                "Fix driver/container GPU visibility before inference tuning.",
                str(path),
            )
        else:
            add_issue(
                issues,
                "wrong_gpu_inventory",
                "blocker",
                "Host does not expose exactly 8 B200 GPUs.",
                "Fix CUDA_VISIBLE_DEVICES, container device mapping, or target host selection.",
                str(path),
            )
    for engine in ("sglang", "vllm"):
        check = (data.get("checks") or {}).get(engine) or {}
        if not check:
            continue
        if check.get("version") is None:
            add_issue(
                issues,
                "missing_package",
                "blocker",
                "Inference engine package is missing.",
                "Install the matching requirements-sglang.txt or requirements-vllm.txt stack.",
                str(path),
            )
        elif check.get("missing_required_flags"):
            add_issue(
                issues,
                "unsupported_flag",
                "blocker",
                "Installed engine does not expose required launch flags.",
                "Pin or upgrade the engine version; rerun preflight before loading the model.",
                str(path),
            )
        if engine == "vllm" and check.get("vllm_batch_invariant") != "1":
            add_issue(
                issues,
                "missing_vllm_batch_invariant",
                "blocker",
                "VLLM_BATCH_INVARIANT is not set to 1.",
                "Export VLLM_BATCH_INVARIANT=1 before vLLM preflight and serving.",
                str(path),
            )


def dedupe_issues(issues: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for issue in issues:
        key = issue["code"]
        if key in grouped:
            sources = grouped[key].setdefault("sources", [grouped[key]["source"]])
            if issue["source"] not in sources:
                sources.append(issue["source"])
            continue
        grouped[key] = dict(issue)
        grouped[key]["sources"] = [issue["source"]]
    deduped = []
    for issue in grouped.values():
        sources = issue.pop("sources")
        issue["source"] = ", ".join(sources[:4])
        if len(sources) > 4:
            issue["source"] += f", +{len(sources) - 4} more"
        deduped.append(issue)
    severity_order = {"blocker": 0, "tuning": 1, "info": 2}
    return sorted(deduped, key=lambda item: (severity_order.get(item["severity"], 9), item["code"], item["source"]))


def triage_run(run_dir: Path) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix in {".log", ".md", ".txt"}:
            scan_text_file(path, issues)
        elif path.name == "result.json":
            scan_result_json(path, issues)
        elif path.name == "summary.json":
            scan_summary_json(path, issues)
        elif path.name == "pipeline.json":
            scan_pipeline_json(path, issues)
        elif path.name.startswith("preflight-") and path.suffix == ".json":
            scan_preflight_json(path, issues)
    issues = dedupe_issues(issues)
    return {
        "run_dir": str(run_dir),
        "issue_count": len(issues),
        "blocker_count": sum(1 for issue in issues if issue["severity"] == "blocker"),
        "tuning_count": sum(1 for issue in issues if issue["severity"] == "tuning"),
        "issues": issues,
    }


def format_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# DeepSeek-V4-Pro Run Triage",
        "",
        f"run_dir: `{report['run_dir']}`",
        f"issues: `{report['issue_count']}`",
        f"blockers: `{report['blocker_count']}`",
        f"tuning: `{report['tuning_count']}`",
        "",
        "| severity | code | signal | next action | source |",
        "|---|---|---|---|---|",
    ]
    for issue in report["issues"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    issue["severity"],
                    issue["code"],
                    issue["signal"].replace("|", "\\|"),
                    issue["next_action"].replace("|", "\\|"),
                    issue["source"],
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Triage DeepSeek-V4-Pro 8x B200 run logs.")
    parser.add_argument("run_dir")
    parser.add_argument("--json-output", default=None)
    parser.add_argument("--markdown-output", default=None)
    args = parser.parse_args()

    report = triage_run(Path(args.run_dir))
    output_json = json.dumps(report, indent=2) + "\n"
    output_markdown = format_markdown(report)
    if args.json_output:
        Path(args.json_output).write_text(output_json, encoding="utf-8")
    if args.markdown_output:
        Path(args.markdown_output).write_text(output_markdown, encoding="utf-8")
    print(output_json)
    print(output_markdown)
    return 1 if report["blocker_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
