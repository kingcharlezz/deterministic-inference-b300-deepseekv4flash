#!/usr/bin/env python3
"""Sustained-load generator: keep N requests in flight for a fixed duration,
measure client-side aggregate output tok/s. Robust to transient connection
errors (retries). Greedy (temp=0) so it doubles as a load + sanity tool.
"""
import argparse
import asyncio
import os
import time

import aiohttp


async def worker(session, base_url, model, prompt, max_tokens, stop_at, stats, wid):
    while time.monotonic() < stop_at:
        body = {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0,
            "top_p": 1,
            "seed": 42,
            "stream": False,
        }
        try:
            async with session.post(
                base_url.rstrip("/") + "/v1/completions", json=body
            ) as r:
                data = await r.json()
            n = data.get("usage", {}).get("completion_tokens", 0)
            stats["out"] += n
            stats["req"] += 1
        except Exception as e:  # noqa: BLE001 - transient conn errors, retry
            stats["err"] += 1
            await asyncio.sleep(0.05)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-V4-Flash")
    ap.add_argument("--concurrency", type=int, default=512)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--prompt-tokens", type=int, default=128)
    ap.add_argument("--api-key", default=os.environ.get("VLLM_API_KEY", ""))
    args = ap.parse_args()
    headers = {"Authorization": f"Bearer {args.api_key}"} if args.api_key else {}

    # Distinct prompt per worker (avoid prefix-cache-like effects); ~prompt-tokens words.
    base = (
        "Write a detailed technical explanation. Topic number {wid}: discuss "
        "distributed systems, consensus, replication, and fault tolerance. "
    )
    prompts = [
        (base.format(wid=w) + "background context. " * (args.prompt_tokens // 3))
        for w in range(args.concurrency)
    ]

    conn = aiohttp.TCPConnector(limit=0, limit_per_host=0)
    timeout = aiohttp.ClientTimeout(total=None, sock_connect=30, sock_read=600)
    stats = {"out": 0, "req": 0, "err": 0}
    t0 = time.monotonic()
    stop_at = t0 + args.duration
    async with aiohttp.ClientSession(
        connector=conn, timeout=timeout, headers=headers
    ) as session:
        tasks = [
            asyncio.create_task(
                worker(
                    session, args.base_url, args.model, prompts[w],
                    args.max_tokens, stop_at, stats, w,
                )
            )
            for w in range(args.concurrency)
        ]
        await asyncio.gather(*tasks)
    dt = time.monotonic() - t0
    print(
        f"concurrency={args.concurrency} duration={dt:.1f}s "
        f"completed_reqs={stats['req']} errors={stats['err']} "
        f"output_tokens={stats['out']} "
        f"CLIENT_OUTPUT_TOK_S={stats['out']/dt:.1f}",
        flush=True,
    )


if __name__ == "__main__":
    asyncio.run(main())
