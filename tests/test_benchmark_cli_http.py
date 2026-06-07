from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


class FakeInferenceHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    requests: list[dict[str, Any]] = []

    def log_message(self, format: str, *args: Any) -> None:
        return

    def write_json(self, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def write_sse(self, chunks: list[dict[str, Any]]) -> None:
        body = "".join(
            f"data: {json.dumps(chunk)}\n\n" for chunk in chunks
        ) + "data: [DONE]\n\n"
        encoded = body.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self) -> None:
        if self.path in {"/health", "/v1/models"}:
            self.write_json({"ok": True})
            return
        self.send_error(404)

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        data = json.loads(self.rfile.read(length).decode("utf-8"))
        FakeInferenceHandler.requests.append({"path": self.path, "body": data})
        if self.path == "/generate":
            prompt = data["text"]
            sampling = data["sampling_params"]
            if sampling.get("temperature") != 0 or sampling.get("top_k") != -1:
                self.send_error(400, "bad sglang sampling params")
                return
            self.write_json(
                {
                    "text": f"stable::{prompt}",
                    "meta_info": {
                        "prompt_tokens": 8,
                        "completion_tokens": 2048,
                        "ttft": 0.001,
                        "e2e_latency": 0.002,
                    },
                }
            )
            return
        if self.path == "/v1/completions":
            prompt = data["prompt"]
            if data.get("temperature") != 0 or data.get("seed") != 42:
                self.send_error(400, "bad vllm request params")
                return
            if data.get("stream"):
                self.write_sse(
                    [
                        {"choices": [{"text": f"stable::{prompt}"}]},
                        {
                            "choices": [],
                            "usage": {
                                "prompt_tokens": 8,
                                "completion_tokens": 2048,
                            },
                        },
                    ]
                )
                return
            self.write_json(
                {
                    "choices": [{"text": f"stable::{prompt}"}],
                    "usage": {
                        "prompt_tokens": 8,
                        "completion_tokens": 2048,
                    },
                }
            )
            return
        self.send_error(404)


class FakeServer:
    def __enter__(self) -> str:
        FakeInferenceHandler.requests = []
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), FakeInferenceHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        return f"http://127.0.0.1:{self.server.server_port}"

    def __exit__(self, exc_type, exc, tb) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


class BenchmarkCliHttpTests(unittest.TestCase):
    def run_benchmark(self, backend: str, base_url: str, tmp: Path) -> subprocess.CompletedProcess[str]:
        json_path = tmp / "result.json"
        md_path = tmp / "benchmark.md"
        return subprocess.run(
            [
                sys.executable,
                "benchmark/bench_deterministic_inference.py",
                "--backend",
                backend,
                "--base-url",
                base_url,
                "--model",
                "deepseek-ai/DeepSeek-V4-Pro",
                "--hardware-label",
                "8xB200",
                "--determinism-repeats",
                "1",
                "--determinism-concurrencies",
                "1",
                "--skip-order-check",
                "--concurrencies",
                "1",
                "--min-requests",
                "1",
                "--target-output-tok-s",
                "1",
                "--misconfig-output-tok-s",
                "1",
                "--json-output",
                str(json_path),
                "--markdown-output",
                str(md_path),
            ],
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=30,
        )

    def test_sglang_native_cli_writes_result_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            with FakeServer() as base_url:
                proc = self.run_benchmark("sglang-native", base_url, tmp)

            self.assertEqual(proc.returncode, 0, proc.stdout)
            result = json.loads((tmp / "result.json").read_text(encoding="utf-8"))
            self.assertEqual(result["backend"], "sglang-native")
            self.assertEqual(result["request_params"]["top_k"], -1)
            self.assertGreaterEqual(result["benchmark"][0]["output_tok_s"], 1)
            self.assertIn("out tok/s", (tmp / "benchmark.md").read_text(encoding="utf-8"))
            self.assertTrue(any(row["path"] == "/generate" for row in FakeInferenceHandler.requests))

    def test_openai_completions_cli_writes_result_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            with FakeServer() as base_url:
                proc = self.run_benchmark("openai-completions", base_url, tmp)

            self.assertEqual(proc.returncode, 0, proc.stdout)
            result = json.loads((tmp / "result.json").read_text(encoding="utf-8"))
            self.assertEqual(result["backend"], "openai-completions")
            self.assertEqual(result["request_params"]["seed"], 42)
            self.assertGreaterEqual(result["benchmark"][0]["output_tok_s"], 1)
            self.assertTrue(
                any(row["path"] == "/v1/completions" for row in FakeInferenceHandler.requests)
            )


if __name__ == "__main__":
    unittest.main()
