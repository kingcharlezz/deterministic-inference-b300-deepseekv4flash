from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import triage_deepseek_v4_flash_run as triage  # noqa: E402


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


class TriageTests(unittest.TestCase):
    def test_triage_detects_preflight_blockers(self) -> None:
        run_dir = Path(tempfile.mkdtemp())
        write_json(
            run_dir / "pipeline.json",
            {
                "selected_engines": [],
                "preflight": [
                    {"engine": "sglang", "ok": False},
                    {"engine": "vllm", "ok": False},
                ],
            },
        )
        (run_dir / "preflight-sglang.log").write_text(
            "NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver.\n"
            "ModuleNotFoundError: No module named 'sglang'\n",
            encoding="utf-8",
        )

        report = triage.triage_run(run_dir)
        codes = {issue["code"] for issue in report["issues"]}

        self.assertGreaterEqual(report["blocker_count"], 3)
        self.assertIn("no_engine_passed_preflight", codes)
        self.assertIn("nvidia_driver_unavailable", codes)
        self.assertIn("missing_package", codes)

    def test_preflight_json_missing_package_does_not_emit_unsupported_flag(self) -> None:
        run_dir = Path(tempfile.mkdtemp())
        write_json(
            run_dir / "preflight-sglang.json",
            {
                "checks": {
                    "gpu": {"ok": True, "rows": [f"{idx}, NVIDIA B200" for idx in range(8)]},
                    "sglang": {
                        "version": None,
                        "missing_required_flags": ["--enable-deterministic-inference"],
                    },
                }
            },
        )

        report = triage.triage_run(run_dir)
        codes = {issue["code"] for issue in report["issues"]}

        self.assertIn("missing_package", codes)
        self.assertNotIn("unsupported_flag", codes)

    def test_triage_detects_determinism_and_throughput_issues(self) -> None:
        run_dir = Path(tempfile.mkdtemp())
        write_json(
            run_dir / "sglang-fa3" / "result.json",
            {
                "targets": {
                    "misconfiguration_below_output_tok_s": 2500,
                    "pass_output_tok_s": 5000,
                },
                "determinism": {
                    "same_prompt": [{"mismatches": 1}],
                    "mixed_batch": [{"mismatches": 1}],
                    "order": [{"mismatches": 0}],
                },
                "benchmark": [{"output_tok_s": 1200.0}],
            },
        )

        report = triage.triage_run(run_dir)
        codes = {issue["code"] for issue in report["issues"]}

        self.assertIn("determinism_failure", codes)
        self.assertIn("throughput_below_misconfig", codes)

    def test_format_markdown_contains_next_action(self) -> None:
        report = {
            "run_dir": "/tmp/run",
            "issue_count": 1,
            "blocker_count": 1,
            "tuning_count": 0,
            "issues": [
                {
                    "severity": "blocker",
                    "code": "missing_package",
                    "signal": "Inference engine package is missing.",
                    "next_action": "Install the matching requirements file.",
                    "source": "/tmp/run/server.log",
                }
            ],
        }

        markdown = triage.format_markdown(report)

        self.assertIn("missing_package", markdown)
        self.assertIn("Install the matching requirements file.", markdown)


if __name__ == "__main__":
    unittest.main()
