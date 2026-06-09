from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "benchmark"))

import bench_deterministic_inference as bench  # noqa: E402


class BenchmarkHarnessTests(unittest.TestCase):
    def test_determinism_passes_for_same_prompt_and_reordered_prompts(self) -> None:
        async def fake_generate_once(
            client,
            backend,
            base_url,
            model,
            prompt,
            request_params,
            stream,
        ):
            return bench.ResponseStats(
                text=f"stable::{prompt}",
                prompt_tokens=11,
                completion_tokens=256,
                ttft_s=0.01,
                latency_s=0.02,
            )

        with patch.object(bench, "generate_once", fake_generate_once):
            result = asyncio.run(
                bench.determinism_test(
                    client=object(),
                    backend="sglang-native",
                    base_url="http://test",
                    model=bench.DEFAULT_MODEL,
                    request_params=bench.DEFAULT_SGLANG_REQUEST_PARAMS,
                    repeats=2,
                    concurrencies=[1, 4],
                    order_checks=True,
                )
            )

        self.assertEqual(len(result["same_prompt"]), 4)
        self.assertTrue(all(row["mismatches"] == 0 for row in result["same_prompt"]))
        self.assertEqual(len(result["mixed_batch"]), 2)
        self.assertTrue(all(row["mismatches"] == 0 for row in result["mixed_batch"]))
        self.assertEqual(result["order"][0]["mismatches"], 0)

    def test_determinism_fails_on_same_prompt_text_mismatch(self) -> None:
        calls = 0

        async def fake_generate_once(
            client,
            backend,
            base_url,
            model,
            prompt,
            request_params,
            stream,
        ):
            nonlocal calls
            calls += 1
            text = "baseline" if calls == 1 else "changed"
            return bench.ResponseStats(
                text=text,
                prompt_tokens=11,
                completion_tokens=256,
                ttft_s=0.01,
                latency_s=0.02,
            )

        with patch.object(bench, "generate_once", fake_generate_once):
            with self.assertRaisesRegex(AssertionError, "same-prompt determinism failed"):
                asyncio.run(
                    bench.determinism_test(
                        client=object(),
                        backend="sglang-native",
                        base_url="http://test",
                        model=bench.DEFAULT_MODEL,
                        request_params=bench.DEFAULT_SGLANG_REQUEST_PARAMS,
                        repeats=1,
                        concurrencies=[2],
                        order_checks=True,
                    )
                )

    def test_determinism_fails_on_mixed_prompt_batch_mismatch(self) -> None:
        calls_by_prompt: dict[str, int] = {}

        async def fake_generate_once(
            client,
            backend,
            base_url,
            model,
            prompt,
            request_params,
            stream,
        ):
            calls_by_prompt[prompt] = calls_by_prompt.get(prompt, 0) + 1
            text = f"stable::{prompt}"
            if prompt == bench.ORDER_PROMPTS[0] and calls_by_prompt[prompt] > 1:
                text = f"changed::{prompt}"
            return bench.ResponseStats(
                text=text,
                prompt_tokens=11,
                completion_tokens=256,
                ttft_s=0.01,
                latency_s=0.02,
            )

        with patch.object(bench, "generate_once", fake_generate_once):
            with self.assertRaisesRegex(AssertionError, "mixed-prompt batch invariance failed"):
                asyncio.run(
                    bench.determinism_test(
                        client=object(),
                        backend="sglang-native",
                        base_url="http://test",
                        model=bench.DEFAULT_MODEL,
                        request_params=bench.DEFAULT_SGLANG_REQUEST_PARAMS,
                        repeats=1,
                        concurrencies=[1],
                        order_checks=True,
                    )
                )

    def test_benchmark_aggregates_token_and_latency_metrics(self) -> None:
        async def fake_generate_once(
            client,
            backend,
            base_url,
            model,
            prompt,
            request_params,
            stream,
        ):
            return bench.ResponseStats(
                text="x" * 16,
                prompt_tokens=10,
                completion_tokens=256,
                ttft_s=0.10,
                latency_s=0.25,
            )

        with patch.object(bench, "generate_once", fake_generate_once):
            rows = asyncio.run(
                bench.benchmark(
                    client=object(),
                    backend="openai-completions",
                    base_url="http://test",
                    model=bench.DEFAULT_MODEL,
                    request_params=bench.DEFAULT_VLLM_REQUEST_PARAMS,
                    concurrencies=[1, 4],
                    min_requests=4,
                )
            )

        self.assertEqual([row["concurrency"] for row in rows], [1, 4])
        self.assertEqual([row["requests"] for row in rows], [4, 4])
        self.assertEqual([row["output_tokens"] for row in rows], [1024, 1024])
        self.assertEqual([row["prompt_tokens"] for row in rows], [40, 40])
        self.assertTrue(all(row["output_tok_s"] > 0 for row in rows))
        self.assertEqual(rows[0]["ttft_p50_s"], 0.10)
        self.assertEqual(rows[1]["lat_p99_s"], 0.25)

    def test_markdown_table_contains_required_metrics(self) -> None:
        table = bench.format_markdown_table(
            [
                {
                    "concurrency": 128,
                    "requests": 256,
                    "output_tok_s": 5123.4,
                    "prompt_tok_s": 111.1,
                    "total_tok_s": 5234.5,
                    "req_s": 20.0,
                    "ttft_p50_s": 0.12,
                    "ttft_p95_s": 0.23,
                    "ttft_p99_s": 0.34,
                    "lat_p50_s": 1.2,
                    "lat_p95_s": 2.3,
                    "lat_p99_s": 3.4,
                }
            ]
        )

        self.assertIn("out tok/s", table)
        self.assertIn("TTFT p99", table)
        self.assertIn("5123.4", table)
        self.assertIn("3.400s", table)


if __name__ == "__main__":
    unittest.main()
