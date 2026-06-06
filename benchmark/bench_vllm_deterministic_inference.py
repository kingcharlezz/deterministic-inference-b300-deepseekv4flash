#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Optional

import httpx


DEFAULT_MODEL = "deepseek-ai/deepseek-v4-flash"
DEFAULT_REQUEST_PARAMS = {
    "temperature": 0,
    "top_p": 1,
    "max_tokens": 256,
    "seed": 42,
}

DETERMINISM_PROMPT = (
    "You are a deterministic inference probe. For this exact prompt, write a "
    "stable technical note about batch-invariant decoding. Use plain ASCII "
    "text, no lists, no markdown, and end with the token STABLE-END."
)

ORDER_PROMPTS = [
    "Order probe A: explain deterministic argmax decoding in 80 words.",
    "Order probe B: explain batch-invariant attention in 80 words.",
    "Order probe C: explain reproducible request seeding in 80 words.",
    "Order probe D: explain tensor parallel inference in 80 words.",
    "Order probe E: explain CUDA graph replay for inference in 80 words.",
    "Order probe F: explain KV cache effects on determinism in 80 words.",
    "Order probe G: explain exact output comparison in 80 words.",
    "Order probe H: explain throughput saturation in 80 words.",
]


@dataclass
class ResponseStats:
    text: str
    prompt_tokens: int
    completion_tokens: int
    ttft_s: float
    latency_s: float


def parse_int_csv(value: str) -> list[int]:
    try:
        parsed = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if not parsed:
        raise argparse.ArgumentTypeError("expected at least one integer")
    if any(item <= 0 for item in parsed):
        raise argparse.ArgumentTypeError("all values must be positive")
    return parsed


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * (pct / 100)
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (rank - lo)


def make_headers(api_key: Optional[str]) -> dict[str, str]:
    if not api_key:
        return {}
    return {"Authorization": f"Bearer {api_key}"}


def completion_payload(
    model: str,
    prompt: str,
    request_params: dict[str, Any],
    stream: bool,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        **request_params,
    }
    if stream:
        payload["stream"] = True
        payload["stream_options"] = {"include_usage": True}
    return payload


def build_benchmark_prompt(request_index: int) -> str:
    return (
        f"Benchmark request {request_index:04d}. Produce exactly 256 output "
        "tokens of plain ASCII text about deterministic high-throughput ML "
        "inference. Do not use markdown. Do not stop early. End only after a "
        "complete final sentence."
    )


async def generate_once(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    prompt: str,
    request_params: dict[str, Any],
    stream: bool,
) -> ResponseStats:
    start = time.perf_counter()
    url = f"{base_url.rstrip('/')}/v1/completions"
    payload = completion_payload(model, prompt, request_params, stream)
    if not stream:
        response = await client.post(url, json=payload, timeout=None)
        response.raise_for_status()
        latency = time.perf_counter() - start
        data = response.json()
        usage = data.get("usage") or {}
        choice = (data.get("choices") or [{}])[0]
        return ResponseStats(
            text=choice.get("text") or "",
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            ttft_s=latency,
            latency_s=latency,
        )

    first_token_time: Optional[float] = None
    text_parts: list[str] = []
    usage: dict[str, Any] = {}
    async with client.stream("POST", url, json=payload, timeout=None) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            if not line.startswith("data: "):
                continue
            chunk = line[6:]
            if chunk == "[DONE]":
                continue
            data = json.loads(chunk)
            if data.get("usage"):
                usage = data["usage"]
                continue
            choices = data.get("choices") or []
            if not choices:
                continue
            piece = choices[0].get("text") or ""
            if piece and first_token_time is None:
                first_token_time = time.perf_counter()
            text_parts.append(piece)

    latency = time.perf_counter() - start
    return ResponseStats(
        text="".join(text_parts),
        prompt_tokens=int(usage.get("prompt_tokens") or 0),
        completion_tokens=int(usage.get("completion_tokens") or 0),
        ttft_s=(first_token_time - start) if first_token_time is not None else latency,
        latency_s=latency,
    )


async def bounded_gather(coros: list[Any], concurrency: int) -> list[Any]:
    sem = asyncio.Semaphore(concurrency)

    async def run(coro: Any) -> Any:
        async with sem:
            return await coro

    return await asyncio.gather(*(run(coro) for coro in coros))


async def determinism_test(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    request_params: dict[str, Any],
    repeats: int,
    concurrencies: list[int],
) -> dict[str, Any]:
    results: dict[str, Any] = {"same_prompt": [], "order": []}
    baseline = await generate_once(
        client,
        base_url,
        model,
        DETERMINISM_PROMPT,
        request_params,
        stream=False,
    )
    baseline_text = baseline.text

    for concurrency in concurrencies:
        for repeat in range(1, repeats + 1):
            responses = await bounded_gather(
                [
                    generate_once(
                        client,
                        base_url,
                        model,
                        DETERMINISM_PROMPT,
                        request_params,
                        stream=False,
                    )
                    for _ in range(concurrency)
                ],
                concurrency,
            )
            mismatches = sum(response.text != baseline_text for response in responses)
            results["same_prompt"].append(
                {
                    "concurrency": concurrency,
                    "repeat": repeat,
                    "requests": concurrency,
                    "mismatches": mismatches,
                    "completion_tokens": responses[0].completion_tokens if responses else 0,
                }
            )
            if mismatches:
                raise AssertionError(
                    "same-prompt determinism failed at "
                    f"concurrency={concurrency}, repeat={repeat}, mismatches={mismatches}"
                )

    forward = await bounded_gather(
        [
            generate_once(client, base_url, model, prompt, request_params, stream=False)
            for prompt in ORDER_PROMPTS
        ],
        len(ORDER_PROMPTS),
    )
    reverse = await bounded_gather(
        [
            generate_once(client, base_url, model, prompt, request_params, stream=False)
            for prompt in reversed(ORDER_PROMPTS)
        ],
        len(ORDER_PROMPTS),
    )
    reverse_by_prompt = dict(zip(reversed(ORDER_PROMPTS), reverse))
    order_mismatches = [
        prompt
        for prompt, response in zip(ORDER_PROMPTS, forward)
        if response.text != reverse_by_prompt[prompt].text
    ]
    results["order"].append(
        {
            "requests": len(ORDER_PROMPTS) * 2,
            "orders": ["forward", "reverse"],
            "mismatches": len(order_mismatches),
        }
    )
    if order_mismatches:
        raise AssertionError(f"order invariance failed for {len(order_mismatches)} prompts")
    return results


async def benchmark(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    request_params: dict[str, Any],
    concurrencies: list[int],
    min_requests: int,
) -> list[dict[str, Any]]:
    rows = []
    for concurrency in concurrencies:
        request_count = max(min_requests, concurrency)
        prompts = [build_benchmark_prompt(i) for i in range(request_count)]
        start = time.perf_counter()
        responses = await bounded_gather(
            [
                generate_once(client, base_url, model, prompt, request_params, stream=True)
                for prompt in prompts
            ],
            concurrency,
        )
        wall_s = time.perf_counter() - start
        prompt_tokens = sum(response.prompt_tokens for response in responses)
        output_tokens = sum(response.completion_tokens for response in responses)
        total_tokens = prompt_tokens + output_tokens
        latencies = [response.latency_s for response in responses]
        ttfts = [response.ttft_s for response in responses]
        rows.append(
            {
                "concurrency": concurrency,
                "requests": request_count,
                "wall_s": wall_s,
                "output_tokens": output_tokens,
                "prompt_tokens": prompt_tokens,
                "total_tokens": total_tokens,
                "output_tok_s": output_tokens / wall_s if wall_s else 0,
                "prompt_tok_s": prompt_tokens / wall_s if wall_s else 0,
                "total_tok_s": total_tokens / wall_s if wall_s else 0,
                "req_s": request_count / wall_s if wall_s else 0,
                "ttft_p50_s": percentile(ttfts, 50),
                "ttft_p95_s": percentile(ttfts, 95),
                "ttft_p99_s": percentile(ttfts, 99),
                "lat_p50_s": percentile(latencies, 50),
                "lat_p95_s": percentile(latencies, 95),
                "lat_p99_s": percentile(latencies, 99),
            }
        )
    return rows


def print_markdown_table(rows: list[dict[str, Any]]) -> None:
    headers = [
        "conc",
        "reqs",
        "out tok/s",
        "prompt tok/s",
        "total tok/s",
        "req/s",
        "TTFT p50",
        "TTFT p95",
        "TTFT p99",
        "lat p50",
        "lat p95",
        "lat p99",
    ]
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        print(
            "| "
            + " | ".join(
                [
                    str(row["concurrency"]),
                    str(row["requests"]),
                    f'{row["output_tok_s"]:.1f}',
                    f'{row["prompt_tok_s"]:.1f}',
                    f'{row["total_tok_s"]:.1f}',
                    f'{row["req_s"]:.2f}',
                    f'{row["ttft_p50_s"]:.3f}s',
                    f'{row["ttft_p95_s"]:.3f}s',
                    f'{row["ttft_p99_s"]:.3f}s',
                    f'{row["lat_p50_s"]:.3f}s',
                    f'{row["lat_p95_s"]:.3f}s',
                    f'{row["lat_p99_s"]:.3f}s',
                ]
            )
            + " |"
        )


async def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate deterministic vLLM completions and benchmark throughput."
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY"))
    parser.add_argument("--hardware-label", default="B300")
    parser.add_argument("--skip-determinism", action="store_true")
    parser.add_argument("--skip-benchmark", action="store_true")
    parser.add_argument("--determinism-repeats", type=int, default=3)
    parser.add_argument("--min-requests", type=int, default=300)
    parser.add_argument("--min-output-tok-s", type=float, default=1000)
    parser.add_argument(
        "--concurrencies",
        type=parse_int_csv,
        default=parse_int_csv("1,4,8,16,32,64,128,256,300"),
        help="Comma-separated benchmark concurrency levels.",
    )
    parser.add_argument(
        "--determinism-concurrencies",
        type=parse_int_csv,
        default=parse_int_csv("1,8,32,128,300"),
        help="Comma-separated determinism concurrency levels.",
    )
    args = parser.parse_args()
    if args.determinism_repeats <= 0:
        parser.error("--determinism-repeats must be positive")
    if args.min_requests <= 0:
        parser.error("--min-requests must be positive")

    request_params = dict(DEFAULT_REQUEST_PARAMS)
    output: dict[str, Any] = {
        "base_url": args.base_url,
        "hardware_label": args.hardware_label,
        "model": args.model,
        "request_params": request_params,
    }
    async with httpx.AsyncClient(
        headers=make_headers(args.api_key),
        timeout=None,
    ) as client:
        if not args.skip_determinism:
            output["determinism"] = await determinism_test(
                client,
                args.base_url,
                args.model,
                request_params,
                args.determinism_repeats,
                args.determinism_concurrencies,
            )
        if not args.skip_benchmark:
            output["benchmark"] = await benchmark(
                client,
                args.base_url,
                args.model,
                request_params,
                args.concurrencies,
                args.min_requests,
            )

    print(json.dumps(output, indent=2))
    if "benchmark" in output:
        print()
        print_markdown_table(output["benchmark"])
        best = max(output["benchmark"], key=lambda row: row["output_tok_s"])
        if args.min_output_tok_s > 0 and best["output_tok_s"] < args.min_output_tok_s:
            print(
                "ERROR: best output throughput "
                f'{best["output_tok_s"]:.1f} tok/s is below '
                f"{args.min_output_tok_s:.1f} tok/s",
                file=sys.stderr,
            )
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
