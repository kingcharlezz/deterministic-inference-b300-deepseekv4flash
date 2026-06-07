from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import preflight_8xb200_deepseek_v4_pro as preflight  # noqa: E402


class PreflightTests(unittest.TestCase):
    def test_missing_flags_reports_absent_entries(self) -> None:
        help_text = "--model-path PATH --tp TP --enable-deterministic-inference"

        missing = preflight.missing_flags(
            help_text,
            ["--model-path", "--tp", "--attention-backend", "--enable-deterministic-inference"],
        )

        self.assertEqual(missing, ["--attention-backend"])

    def test_query_gpus_requires_exactly_eight_b200_rows(self) -> None:
        rows = "\n".join(
            f"{idx}, NVIDIA B200, 555.55.55, 192000 MiB" for idx in range(8)
        )

        with patch.object(preflight.shutil, "which", return_value="/usr/bin/nvidia-smi"):
            with patch.object(
                preflight,
                "run_capture",
                return_value={"exit_code": 0, "output": rows, "command": ["nvidia-smi"]},
            ):
                result = preflight.query_gpus()

        self.assertTrue(result["ok"])
        self.assertEqual(len(result["rows"]), 8)

    def test_query_gpus_rejects_non_b200_or_wrong_count(self) -> None:
        rows = "\n".join(
            f"{idx}, NVIDIA H200, 555.55.55, 141000 MiB" for idx in range(8)
        )

        with patch.object(preflight.shutil, "which", return_value="/usr/bin/nvidia-smi"):
            with patch.object(
                preflight,
                "run_capture",
                return_value={"exit_code": 0, "output": rows, "command": ["nvidia-smi"]},
            ):
                result = preflight.query_gpus()

        self.assertFalse(result["ok"])

    def test_sglang_check_requires_deterministic_flag(self) -> None:
        help_text = " ".join(flag for flag in preflight.REQUIRED_SGLANG_FLAGS if flag != "--enable-deterministic-inference")

        with patch.object(preflight, "package_version", return_value="0.5.12.post1"):
            with patch.object(
                preflight,
                "run_capture",
                return_value={"exit_code": 0, "output": help_text, "command": ["python"]},
            ):
                result = preflight.check_sglang("python3")

        self.assertFalse(result["ok"])
        self.assertIn("--enable-deterministic-inference", result["missing_required_flags"])


if __name__ == "__main__":
    unittest.main()
