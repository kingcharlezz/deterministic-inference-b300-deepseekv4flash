from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import summarize_deepseek_v4_pro_run as summary  # noqa: E402


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


class RunSummaryTests(unittest.TestCase):
    def make_run(self, output_tok_s: float, status: str = "passed") -> Path:
        root = Path(tempfile.mkdtemp())
        variant_dir = root / "sglang-fa3-mem090"
        variant_dir.mkdir()
        result_path = variant_dir / "result.json"
        write_json(
            root / "summary.json",
            {
                "attempts": [
                    {
                        "name": "sglang-fa3-mem090",
                        "engine": "sglang",
                        "status": status,
                        "best_output_tok_s": output_tok_s,
                        "result_json": str(result_path),
                        "env": {
                            "MODEL": "deepseek-ai/DeepSeek-V4-Pro",
                            "PORT": "30000",
                            "TP": "8",
                            "ATTENTION_BACKEND": "fa3",
                            "MEM_FRACTION_STATIC": "0.90",
                        },
                    }
                ]
            },
        )
        write_json(
            root / "environment.json",
            {
                "packages": {
                    "output": "sglang==0.5.12.post1\ntorch==2.9.0\nunrelated==1.0\n"
                },
                "gpu": {
                    "output": (
                        "index, name, driver_version, memory.total\n"
                        "0, NVIDIA B200, 555.55.55, 192000 MiB\n"
                        "1, NVIDIA B200, 555.55.55, 192000 MiB\n"
                        "2, NVIDIA B200, 555.55.55, 192000 MiB\n"
                        "3, NVIDIA B200, 555.55.55, 192000 MiB\n"
                        "4, NVIDIA B200, 555.55.55, 192000 MiB\n"
                        "5, NVIDIA B200, 555.55.55, 192000 MiB\n"
                        "6, NVIDIA B200, 555.55.55, 192000 MiB\n"
                        "7, NVIDIA B200, 555.55.55, 192000 MiB\n"
                    )
                },
            },
        )
        write_json(
            result_path,
            {
                "backend": "sglang-native",
                "base_url": "http://127.0.0.1:30000",
                "hardware_label": "8xB200",
                "model": "deepseek-ai/DeepSeek-V4-Pro",
                "request_params": {
                    "temperature": 0,
                    "top_p": 1,
                    "top_k": -1,
                    "max_new_tokens": 256,
                },
                "targets": {
                    "pass_output_tok_s": 5000,
                },
                "determinism": {
                    "same_prompt": [
                        {"concurrency": 1, "repeat": 1, "requests": 1, "mismatches": 0},
                        {"concurrency": 8, "repeat": 1, "requests": 8, "mismatches": 0},
                    ],
                    "order": [{"requests": 16, "orders": ["forward", "reverse"], "mismatches": 0}],
                },
                "benchmark": [
                    {
                        "concurrency": 128,
                        "requests": 256,
                        "wall_s": 12.0,
                        "output_tokens": 65536,
                        "prompt_tokens": 8192,
                        "total_tokens": 73728,
                        "output_tok_s": output_tok_s,
                        "prompt_tok_s": 682.7,
                        "total_tok_s": output_tok_s + 682.7,
                        "req_s": 21.3,
                        "ttft_p50_s": 0.11,
                        "ttft_p95_s": 0.22,
                        "ttft_p99_s": 0.33,
                        "lat_p50_s": 3.0,
                        "lat_p95_s": 4.0,
                        "lat_p99_s": 5.0,
                    }
                ],
            },
        )
        (variant_dir / "benchmark.md").write_text("| conc | out tok/s |\n|---|---|\n", encoding="utf-8")
        return root

    def test_build_report_accepts_passing_result(self) -> None:
        report, passed = summary.build_report(self.make_run(5123.4), 5000)

        self.assertTrue(passed)
        self.assertIn("proof >= target: `True`", report)
        self.assertIn("--enable-deterministic-inference", report)
        self.assertIn("sglang==0.5.12.post1", report)
        self.assertIn("NVIDIA B200", report)
        self.assertIn("environment has 8 B200 GPUs: `True`", report)

    def test_build_report_rejects_below_target_result(self) -> None:
        report, passed = summary.build_report(self.make_run(4999.9), 5000)

        self.assertFalse(passed)
        self.assertIn("best output tok/s: `4999.9`", report)
        self.assertIn("proof >= target: `False`", report)


if __name__ == "__main__":
    unittest.main()
