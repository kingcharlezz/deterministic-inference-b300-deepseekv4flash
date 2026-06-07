from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import run_8xb200_deepseek_v4_pro_pipeline as pipeline  # noqa: E402


class PipelineTests(unittest.TestCase):
    def test_parse_engines_accepts_ordered_known_engines(self) -> None:
        self.assertEqual(pipeline.parse_engines("sglang,vllm"), ["sglang", "vllm"])
        self.assertEqual(pipeline.parse_engines(" vllm "), ["vllm"])

    def test_parse_engines_rejects_unknown_engine(self) -> None:
        with self.assertRaises(SystemExit):
            pipeline.parse_engines("sglang,unknown")

    def test_pipeline_returns_nonzero_when_preflight_selects_no_engines(self) -> None:
        run_dir = Path(tempfile.mkdtemp())

        def fake_preflight(engine: str, python: str, selected_run_dir: Path):
            return {
                "engine": engine,
                "command": [python],
                "exit_code": 2,
                "log": str(selected_run_dir / f"preflight-{engine}.log"),
                "json": str(selected_run_dir / f"preflight-{engine}.json"),
                "ok": False,
            }

        argv = [
            "pipeline",
            "--engines",
            "sglang,vllm",
            "--run-dir",
            str(run_dir),
            "--dry-run",
        ]
        with patch.object(sys, "argv", argv):
            with patch.object(pipeline, "preflight_engine", side_effect=fake_preflight):
                with contextlib.redirect_stdout(io.StringIO()):
                    with contextlib.redirect_stderr(io.StringIO()):
                        exit_code = pipeline.main()

        self.assertEqual(exit_code, 2)
        state = json.loads((run_dir / "pipeline.json").read_text(encoding="utf-8"))
        self.assertEqual(state["selected_engines"], [])
        self.assertIn("triage", state)
        self.assertTrue((run_dir / "triage.json").exists())

    def test_run_command_records_exit_code_and_log(self) -> None:
        run_dir = Path(tempfile.mkdtemp())
        result = pipeline.run_command(
            [sys.executable, "-c", "print('ok')"],
            run_dir / "cmd.log",
        )

        self.assertEqual(result["exit_code"], 0)
        self.assertEqual((run_dir / "cmd.log").read_text(encoding="utf-8").strip(), "ok")


if __name__ == "__main__":
    unittest.main()
