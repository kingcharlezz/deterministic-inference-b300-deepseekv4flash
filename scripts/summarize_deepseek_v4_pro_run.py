#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_TARGET_OUTPUT_TOK_S = 5000.0
DEFAULT_MODEL = "deepseek-ai/DeepSeek-V4-Pro"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def package_lines(environment: dict[str, Any]) -> list[str]:
    output = ((environment.get("packages") or {}).get("output") or "").splitlines()
    interesting = ("sglang", "vllm", "torch", "triton", "flashinfer", "flash-attn")
    return [line for line in output if line.split("==", 1)[0].lower() in interesting]


def gpu_lines(environment: dict[str, Any]) -> list[str]:
    return ((environment.get("gpu") or {}).get("output") or "").splitlines()


def has_8_b200_gpus(lines: list[str]) -> bool:
    data_rows = [line for line in lines if line.strip() and not line.lower().startswith("index,")]
    return len(data_rows) == 8 and all("B200" in line for line in data_rows)


def best_row(result: dict[str, Any]) -> dict[str, Any]:
    rows = result.get("benchmark") or []
    if not rows:
        return {}
    return max(rows, key=lambda row: float(row.get("output_tok_s") or 0.0))


def deterministic_passed(result: dict[str, Any]) -> bool:
    determinism = result.get("determinism") or {}
    same_prompt = determinism.get("same_prompt") or []
    mixed_batch = determinism.get("mixed_batch") or []
    order = determinism.get("order") or []
    return (
        bool(same_prompt)
        and bool(mixed_batch)
        and bool(order)
        and all(int(row.get("mismatches") or 0) == 0 for row in same_prompt)
        and all(int(row.get("mismatches") or 0) == 0 for row in mixed_batch)
        and all(int(row.get("mismatches") or 0) == 0 for row in order)
    )


def launch_command(attempt: dict[str, Any]) -> str:
    env = attempt.get("env") or {}
    engine = attempt.get("engine")
    model = env.get("MODEL", DEFAULT_MODEL)
    if engine == "sglang":
        parts = [
            "python -m sglang.launch_server",
            f"  --model-path {model}",
            "  --host 0.0.0.0",
            f"  --port {env.get('PORT', '30000')}",
            f"  --tp {env.get('TP', '8')}",
            f"  --attention-backend {env.get('ATTENTION_BACKEND', 'fa3')}",
            "  --enable-deterministic-inference",
            f"  --mem-fraction-static {env.get('MEM_FRACTION_STATIC', '0.90')}",
        ]
        if env.get("DISABLE_RADIX_CACHE") == "1":
            parts.append("  --disable-radix-cache")
        if env.get("TRITON_ATTENTION_SPLIT_TILE_SIZE"):
            parts.append(
                "  --triton-attention-split-tile-size "
                + env["TRITON_ATTENTION_SPLIT_TILE_SIZE"]
            )
        if env.get("CHUNKED_PREFILL_SIZE"):
            parts.append("  --chunked-prefill-size " + env["CHUNKED_PREFILL_SIZE"])
        return " \\\n".join(parts)
    if engine == "vllm":
        parts = [
            f"VLLM_BATCH_INVARIANT={env.get('VLLM_BATCH_INVARIANT', '1')} vllm serve {model}",
            "  --host 0.0.0.0",
            f"  --port {env.get('PORT', '8000')}",
            "  --seed 0",
            f"  --tensor-parallel-size {env.get('TP', '8')}",
            f"  --gpu-memory-utilization {env.get('GPU_MEMORY_UTILIZATION', '0.90')}",
            f"  --max-model-len {env.get('MAX_MODEL_LEN', '8192')}",
            f"  --max-num-seqs {env.get('MAX_NUM_SEQS', '256')}",
            f"  --max-num-batched-tokens {env.get('MAX_NUM_BATCHED_TOKENS', '8192')}",
        ]
        return " \\\n".join(parts)
    return "<unknown launch command>"


def request_params(result: dict[str, Any]) -> str:
    return json.dumps(result.get("request_params") or {}, separators=(",", ":"))


def deterministic_flags(attempt: dict[str, Any]) -> str:
    env = attempt.get("env") or {}
    if attempt.get("engine") == "sglang":
        flags = ["--enable-deterministic-inference"]
        if env.get("DISABLE_RADIX_CACHE") == "1":
            flags.append("--disable-radix-cache")
        if env.get("TRITON_ATTENTION_SPLIT_TILE_SIZE"):
            flags.append("--triton-attention-split-tile-size " + env["TRITON_ATTENTION_SPLIT_TILE_SIZE"])
        return ", ".join(flags)
    if attempt.get("engine") == "vllm":
        return "VLLM_BATCH_INVARIANT=1, --seed 0, request seed=42"
    return "<unknown>"


def benchmark_table(result_dir: Path, result: dict[str, Any]) -> str:
    markdown_path = result_dir / "benchmark.md"
    if markdown_path.exists():
        return markdown_path.read_text(encoding="utf-8").strip()

    rows = result.get("benchmark") or []
    lines = [
        "| conc | reqs | out tok/s | prompt tok/s | total tok/s | req/s | TTFT p50 | TTFT p95 | TTFT p99 | lat p50 | lat p95 | lat p99 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("concurrency")),
                    str(row.get("requests")),
                    f"{float(row.get('output_tok_s') or 0.0):.1f}",
                    f"{float(row.get('prompt_tok_s') or 0.0):.1f}",
                    f"{float(row.get('total_tok_s') or 0.0):.1f}",
                    f"{float(row.get('req_s') or 0.0):.2f}",
                    f"{float(row.get('ttft_p50_s') or 0.0):.3f}s",
                    f"{float(row.get('ttft_p95_s') or 0.0):.3f}s",
                    f"{float(row.get('ttft_p99_s') or 0.0):.3f}s",
                    f"{float(row.get('lat_p50_s') or 0.0):.3f}s",
                    f"{float(row.get('lat_p95_s') or 0.0):.3f}s",
                    f"{float(row.get('lat_p99_s') or 0.0):.3f}s",
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def find_attempt(run_dir: Path, summary: dict[str, Any]) -> dict[str, Any]:
    attempts = summary.get("attempts") or []
    passed = [attempt for attempt in attempts if attempt.get("status") == "passed"]
    if passed:
        return passed[-1]
    if attempts:
        return max(attempts, key=lambda item: float(item.get("best_output_tok_s") or 0.0))
    raise SystemExit(f"{run_dir} summary has no attempts")


def build_report(run_dir: Path, target_output_tok_s: float) -> tuple[str, bool]:
    summary = load_json(run_dir / "summary.json")
    environment_path = run_dir / "environment.json"
    environment = load_json(environment_path) if environment_path.exists() else {}
    attempt = find_attempt(run_dir, summary)
    result_path = Path(attempt.get("result_json") or "")
    if not result_path.is_absolute():
        result_path = run_dir / result_path
    if not result_path.exists():
        raise SystemExit(f"missing result JSON: {result_path}")

    result = load_json(result_path)
    result_dir = result_path.parent
    best = best_row(result)
    best_output = float(best.get("output_tok_s") or 0.0)
    deterministic_ok = deterministic_passed(result)
    target_ok = best_output >= target_output_tok_s
    model_ok = result.get("model") == DEFAULT_MODEL
    hardware_ok = result.get("hardware_label") == "8xB200"

    gpu = gpu_lines(environment)
    packages = package_lines(environment)
    gpu_ok = has_8_b200_gpus(gpu)
    packages_ok = bool(packages)
    passed = deterministic_ok and target_ok and model_ok and hardware_ok and gpu_ok and packages_ok
    same_prompt = (result.get("determinism") or {}).get("same_prompt") or []
    mixed_batch = (result.get("determinism") or {}).get("mixed_batch") or []
    order = (result.get("determinism") or {}).get("order") or []

    lines = [
        "# DeepSeek-V4-Pro 8x B200 Deterministic Inference Proof",
        "",
        f"run_dir: `{run_dir}`",
        f"engine: `{attempt.get('engine')}`",
        f"variant: `{attempt.get('name')}`",
        f"status: `{attempt.get('status')}`",
        "",
        "## Launch Command",
        "",
        "```bash",
        launch_command(attempt),
        "```",
        "",
        "## Deterministic Controls",
        "",
        f"engine flags: `{deterministic_flags(attempt)}`",
        f"request params: `{request_params(result)}`",
        "",
        "## Versions And GPU",
        "",
        "```text",
        *(packages or ["<not recorded>"]),
        "```",
        "",
        "```text",
        *(gpu or ["<not recorded>"]),
        "```",
        "",
        "## Determinism",
        "",
        f"same-prompt rows: `{len(same_prompt)}`",
        f"same-prompt mismatches: `{sum(int(row.get('mismatches') or 0) for row in same_prompt)}`",
        f"mixed-batch rows: `{len(mixed_batch)}`",
        f"mixed-batch mismatches: `{sum(int(row.get('mismatches') or 0) for row in mixed_batch)}`",
        f"order rows: `{len(order)}`",
        f"order mismatches: `{sum(int(row.get('mismatches') or 0) for row in order)}`",
        "",
        "## Throughput",
        "",
        f"best concurrency: `{best.get('concurrency')}`",
        f"best output tok/s: `{best_output:.1f}`",
        f"target output tok/s: `{target_output_tok_s:.1f}`",
        "",
        benchmark_table(result_dir, result),
        "",
        "## Verdict",
        "",
        f"deterministic: `{deterministic_ok}`",
        f"model: `{result.get('model')}`",
        f"hardware label: `{result.get('hardware_label')}`",
        f"environment has 8 B200 GPUs: `{gpu_ok}`",
        f"package versions recorded: `{packages_ok}`",
        f"proof >= target: `{passed}`",
        "",
    ]
    return "\n".join(lines), passed


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate and summarize a completed DeepSeek-V4-Pro 8x B200 tuning run."
    )
    parser.add_argument("run_dir")
    parser.add_argument("--target-output-tok-s", type=float, default=DEFAULT_TARGET_OUTPUT_TOK_S)
    parser.add_argument("--output", default=None, help="Defaults to <run_dir>/final_report.md")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    report, passed = build_report(run_dir, args.target_output_tok_s)
    output_path = Path(args.output) if args.output else run_dir / "final_report.md"
    output_path.write_text(report, encoding="utf-8")
    print(report)
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
