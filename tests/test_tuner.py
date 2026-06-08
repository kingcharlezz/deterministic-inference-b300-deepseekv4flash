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

import tune_deepseek_v4_flash_8xb200 as tuner  # noqa: E402


class TunerTests(unittest.TestCase):
    def test_tp4_dp2_megamoe_variants_set_explicit_deepgemm_envs(self) -> None:
        variants = {
            variant.name: variant
            for variant in tuner.filter_variants(
                tuner.sglang_variants(),
                "sglang-dsv4-tp4-dp2-megamoe-*",
            )
        }

        self.assertEqual(
            set(variants),
            {
                "sglang-dsv4-tp4-dp2-megamoe-w4a8-mem086",
                "sglang-dsv4-tp4-dp2-megamoe-w4a4-mem086",
            },
        )
        for variant in variants.values():
            self.assertEqual(variant.env["TP"], "4")
            self.assertEqual(variant.env["DP_SIZE"], "2")
            self.assertEqual(variant.env["MOE_RUNNER_BACKEND"], "")
            self.assertEqual(variant.env["MOE_A2A_BACKEND"], "megamoe")
            self.assertEqual(variant.env["SGLANG_OPT_USE_DEEPGEMM_MEGA_MOE"], "1")
            self.assertEqual(
                variant.env["SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK"],
                "8320",
            )

        w4a4 = variants["sglang-dsv4-tp4-dp2-megamoe-w4a4-mem086"].env
        self.assertEqual(w4a4["SGLANG_OPT_DEEPGEMM_MEGA_MOE_USE_FP4_ACTS"], "1")
        self.assertEqual(w4a4["SGLANG_OPT_DEEPGEMM_MEGA_MOE_USE_MXF4_KIND"], "1")

    def test_gpu_inventory_requires_exactly_eight_b200s(self) -> None:
        rows = "\n".join(
            f"{idx}, NVIDIA B200, 555.55.55, 192000 MiB" for idx in range(8)
        )

        self.assertTrue(
            tuner.gpu_inventory_is_8xb200({"exit_code": 0, "output": rows})
        )
        self.assertFalse(
            tuner.gpu_inventory_is_8xb200({"exit_code": 9, "output": rows})
        )
        self.assertFalse(
            tuner.gpu_inventory_is_8xb200({"exit_code": 0, "output": rows.replace("B200", "H200", 1)})
        )

    def test_main_records_gpu_unavailable_before_launching_variants(self) -> None:
        run_dir = Path(tempfile.mkdtemp())
        argv = [
            "tuner",
            "--engines",
            "sglang",
            "--variants",
            "sglang-dsv4-tp4-dp2-*",
            "--run-dir",
            str(run_dir),
        ]

        with patch.object(sys, "argv", argv):
            with patch.object(tuner, "write_environment_snapshot") as write_env:
                with patch.object(
                    tuner,
                    "query_gpu_inventory",
                    return_value={
                        "exit_code": 9,
                        "output": "NVIDIA-SMI has failed\n",
                        "command": ["nvidia-smi"],
                    },
                ):
                    with patch.object(tuner, "run_variant") as run_variant:
                        with contextlib.redirect_stdout(io.StringIO()):
                            with contextlib.redirect_stderr(io.StringIO()):
                                exit_code = tuner.main()

        self.assertEqual(exit_code, 2)
        write_env.assert_called_once()
        run_variant.assert_not_called()

        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["attempts"][0]["status"], "gpu_unavailable")
        self.assertEqual(summary["attempts"][0]["name"], "preflight-gpu-inventory")


if __name__ == "__main__":
    unittest.main()
