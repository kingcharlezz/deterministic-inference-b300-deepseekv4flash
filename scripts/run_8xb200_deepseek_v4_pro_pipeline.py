#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_engines(value: str) -> list[str]:
    engines = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(engines) - {"sglang", "vllm"})
    if unknown:
        raise SystemExit(f"unknown engine(s): {', '.join(unknown)}")
    if not engines:
        raise SystemExit("at least one engine is required")
    return engines


def run_command(
    cmd: list[str],
    log_path: Path,
    env: dict[str, str] | None = None,
    timeout_s: float | None = None,
) -> dict[str, Any]:
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    with log_path.open("wb") as log:
        proc = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            env=merged_env,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=timeout_s,
        )
    return {"command": cmd, "exit_code": proc.returncode, "log": str(log_path)}


def preflight_engine(engine: str, python: str, run_dir: Path) -> dict[str, Any]:
    json_path = run_dir / f"preflight-{engine}.json"
    log_path = run_dir / f"preflight-{engine}.log"
    env = {"VLLM_BATCH_INVARIANT": "1"} if engine == "vllm" else {}
    cmd = [
        python,
        "scripts/preflight_8xb200_deepseek_v4_pro.py",
        "--engine",
        engine,
        "--python",
        python,
        "--json-output",
        str(json_path),
    ]
    result = run_command(cmd, log_path, env=env)
    result["engine"] = engine
    result["json"] = str(json_path)
    result["ok"] = result["exit_code"] == 0
    return result


def write_pipeline_state(run_dir: Path, state: dict[str, Any]) -> None:
    (run_dir / "pipeline.json").write_text(
        json.dumps(state, indent=2) + "\n",
        encoding="utf-8",
    )


def run_triage(run_dir: Path, python: str) -> dict[str, Any]:
    return run_command(
        [
            python,
            "scripts/triage_deepseek_v4_pro_run.py",
            str(run_dir),
            "--json-output",
            str(run_dir / "triage.json"),
            "--markdown-output",
            str(run_dir / "triage.md"),
        ],
        run_dir / "triage.log",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Preflight, tune, and summarize deterministic DeepSeek-V4-Pro "
            "inference on 8x B200."
        )
    )
    parser.add_argument("--engines", default="sglang,vllm")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--variants", default=None)
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
        "--skip-preflight",
        action="store_true",
        help="Run tuner directly for selected engines. Launchers still perform preflight.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    requested_engines = parse_engines(args.engines)
    run_dir = Path(args.run_dir) if args.run_dir else REPO_ROOT / "runs" / time.strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    state: dict[str, Any] = {
        "requested_engines": requested_engines,
        "run_dir": str(run_dir),
        "preflight": [],
        "selected_engines": requested_engines,
    }

    if not args.skip_preflight:
        selected: list[str] = []
        for engine in requested_engines:
            result = preflight_engine(engine, args.python, run_dir)
            state["preflight"].append(result)
            if result["ok"]:
                selected.append(engine)
        state["selected_engines"] = selected
        write_pipeline_state(run_dir, state)
        if not selected:
            state["triage"] = run_triage(run_dir, args.python)
            write_pipeline_state(run_dir, state)
            print(json.dumps(state, indent=2))
            print(f"No selected engine passed preflight; logs in {run_dir}", file=sys.stderr)
            return 2

    if args.dry_run:
        write_pipeline_state(run_dir, state)
        print(json.dumps(state, indent=2))
        return 0

    tune_cmd = [
        args.python,
        "scripts/tune_deepseek_v4_pro_8xb200.py",
        "--engines",
        ",".join(state["selected_engines"]),
        "--run-dir",
        str(run_dir),
        "--python",
        args.python,
        "--startup-timeout-s",
        str(args.startup_timeout_s),
        "--benchmark-timeout-s",
        str(args.benchmark_timeout_s),
        "--cooldown-s",
        str(args.cooldown_s),
        "--min-requests",
        str(args.min_requests),
        "--concurrencies",
        args.concurrencies,
        "--determinism-concurrencies",
        args.determinism_concurrencies,
        "--target-output-tok-s",
        str(args.target_output_tok_s),
        "--misconfig-output-tok-s",
        str(args.misconfig_output_tok_s),
        "--stretch-output-tok-s",
        str(args.stretch_output_tok_s),
    ]
    if args.variants:
        tune_cmd.extend(["--variants", args.variants])
    state["tune"] = run_command(tune_cmd, run_dir / "tune.log")
    write_pipeline_state(run_dir, state)
    if state["tune"]["exit_code"] != 0:
        state["triage"] = run_triage(run_dir, args.python)
        write_pipeline_state(run_dir, state)
        print(json.dumps(state, indent=2))
        print(f"Tuning did not reach target; logs in {run_dir}", file=sys.stderr)
        return int(state["tune"]["exit_code"] or 2)

    summarize_cmd = [
        args.python,
        "scripts/summarize_deepseek_v4_pro_run.py",
        str(run_dir),
        "--target-output-tok-s",
        str(args.target_output_tok_s),
    ]
    state["summary"] = run_command(summarize_cmd, run_dir / "summary-report.log")
    write_pipeline_state(run_dir, state)
    print(json.dumps(state, indent=2))
    return int(state["summary"]["exit_code"] or 0)


if __name__ == "__main__":
    raise SystemExit(main())
