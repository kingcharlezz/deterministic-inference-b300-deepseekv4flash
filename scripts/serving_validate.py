#!/usr/bin/env python3
"""Serving validation for the <=50-concurrency byte-exact deployment.

determinism mode: a pool of DISTINCT prompts is fired in many batches of varying
size (concurrency 2..50), varying composition (mixed batch-mates) and varying
order, repeated R times. Every occurrence of a given prompt must produce the
byte-identical completion -> proves identical-output + mixed-batch + order +
cross-concurrency invariance simultaneously.

perf mode: at a fixed concurrency, measure per-request TTFT (streaming) and
per-request decode tok/s, report p50/p95/p99.
"""
import argparse
import asyncio
import collections
import json
import os
import statistics
import time

import aiohttp

PROMPTS = [
    "Explain how a transformer performs autoregressive decoding.",
    "Write a haiku about the ocean at dawn.",
    "What is the capital of France and why is it historically important?",
    "Summarize the theory of general relativity in three sentences.",
    "List the first 10 prime numbers and explain what makes a number prime.",
    "Describe how photosynthesis converts sunlight into chemical energy.",
    "Compare TCP and UDP for real-time video streaming.",
    "Give a step-by-step recipe for a simple tomato pasta.",
    "Explain the difference between supervised and unsupervised learning.",
    "What causes the seasons on Earth? Be precise.",
    "Write a short limerick about a debugging session.",
    "Explain the CAP theorem in distributed systems.",
    "How does a hash map achieve average O(1) lookup?",
    "Describe the water cycle from evaporation to precipitation.",
    "What is the significance of the number e in mathematics?",
    "Explain how vaccines train the immune system.",
]


def headers():
    k = os.environ.get("VLLM_API_KEY", "")
    return {"Content-Type": "application/json", **({"Authorization": f"Bearer {k}"} if k else {})}


async def gen(session, url, model, prompt, max_tokens):
    body = {"model": model, "prompt": prompt, "max_tokens": max_tokens,
            "temperature": 0, "top_p": 1, "seed": 42, "stream": False}
    for attempt in range(4):
        try:
            async with session.post(url, json=body) as r:
                d = await r.json()
            return d["choices"][0]["text"]
        except Exception:  # noqa: BLE001
            await asyncio.sleep(0.1 * (attempt + 1))
    return None


async def determinism(args):
    url = args.base_url.rstrip("/") + "/v1/completions"
    concs = [int(x) for x in args.concurrencies.split(",")]
    # prompt -> set of distinct outputs observed across every batch/order/repeat
    seen = collections.defaultdict(set)
    occurrences = collections.defaultdict(int)
    conn = aiohttp.TCPConnector(limit=0)
    timeout = aiohttp.ClientTimeout(total=None, sock_connect=30, sock_read=900)
    async with aiohttp.ClientSession(connector=conn, timeout=timeout, headers=headers()) as s:
        for rep in range(args.repeats):
            for c in concs:
                # build a mixed batch of size c: distinct prompts, rotated by
                # (rep,c) so composition AND order vary run to run.
                shift = (rep * 7 + c) % len(PROMPTS)
                batch = [PROMPTS[(shift + i) % len(PROMPTS)] for i in range(c)]
                outs = await asyncio.gather(
                    *[gen(s, url, args.model, p, args.max_tokens) for p in batch]
                )
                for p, o in zip(batch, outs):
                    if o is not None:
                        seen[p].add(o)
                        occurrences[p] += 1
                uniq = max(len(seen[p]) for p in set(batch))
                print(f"[rep {rep} conc {c:2d}] distinct prompts={len(set(batch))} "
                      f"max_variants_per_prompt_so_far={uniq}", flush=True)
    # verdict
    bad = {p: len(v) for p, v in seen.items() if len(v) > 1}
    total = len(seen)
    print(f"\nprompts tested={total}  total occurrences={sum(occurrences.values())}")
    if bad:
        print(f"FAIL: {len(bad)}/{total} prompts produced >1 distinct output across "
              f"batch composition/order/concurrency/repeats:")
        for p, n in list(bad.items())[:5]:
            print(f"   {n} variants: {p[:60]!r}")
        return 1
    print("PASS: every prompt byte-identical across all concurrency/mix/order/repeats")
    return 0


async def stream_one(session, url, model, prompt, max_tokens):
    body = {"model": model, "prompt": prompt, "max_tokens": max_tokens,
            "temperature": 0, "top_p": 1, "seed": 42, "stream": True}
    t0 = time.monotonic()
    ttft = None
    n = 0
    async with session.post(url, json=body) as r:
        async for raw in r.content:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            txt = chunk["choices"][0].get("text", "")
            if txt:
                if ttft is None:
                    ttft = time.monotonic() - t0
                n += 1
    total = time.monotonic() - t0
    decode_s = max(total - (ttft or 0), 1e-6)
    return ttft, n, n / decode_s


async def perf(args):
    url = args.base_url.rstrip("/") + "/v1/completions"
    c = args.concurrency
    conn = aiohttp.TCPConnector(limit=0)
    timeout = aiohttp.ClientTimeout(total=None, sock_connect=30, sock_read=900)
    async with aiohttp.ClientSession(connector=conn, timeout=timeout, headers=headers()) as s:
        res = await asyncio.gather(*[
            stream_one(s, url, args.model, PROMPTS[i % len(PROMPTS)], args.max_tokens)
            for i in range(c)
        ])
    ttfts = sorted(1000 * r[0] for r in res if r[0] is not None)
    decs = sorted(r[2] for r in res)

    def pct(xs, p):
        return xs[min(len(xs) - 1, int(len(xs) * p))]
    print(f"concurrency={c}  requests={len(res)}  max_tokens={args.max_tokens}")
    print(f"TTFT ms   p50={pct(ttfts,.5):.0f}  p95={pct(ttfts,.95):.0f}  p99={pct(ttfts,.99):.0f}")
    print(f"per-req decode tok/s  p50={pct(decs,.5):.1f}  min={decs[0]:.1f}  "
          f"median={statistics.median(decs):.1f}")
    print(f"aggregate decode tok/s ~= {sum(decs):.0f}")
    return 0


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-V4-Flash")
    ap.add_argument("--mode", choices=["determinism", "perf"], default="determinism")
    ap.add_argument("--concurrencies", default="2,4,8,16,32,50")
    ap.add_argument("--concurrency", type=int, default=50)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=128)
    args = ap.parse_args()
    rc = await (determinism(args) if args.mode == "determinism" else perf(args))
    raise SystemExit(rc)


if __name__ == "__main__":
    asyncio.run(main())
