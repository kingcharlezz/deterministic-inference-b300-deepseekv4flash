#!/usr/bin/env python3
"""Determinism probe: fire the SAME prompt N times concurrently, check that
every completion is byte-identical. Reports unique_text count + mismatches
per concurrency level. temp=0, top_p=1, seed=42, greedy.
"""
import argparse
import asyncio
import collections
import json
import sys
import time
import urllib.request


def one_request(base_url, model, prompt, max_tokens, seed):
    body = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0,
            "top_p": 1,
            "seed": seed,
            "stream": False,
        }
    ).encode()
    req = urllib.request.Request(
        base_url.rstrip("/") + "/v1/completions",
        data=body,
        headers={"Content-Type": "application/json", "Authorization": "Bearer x"},
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        data = json.loads(r.read())
    return data["choices"][0]["text"]


async def run_concurrency(base_url, model, prompt, max_tokens, seed, conc):
    loop = asyncio.get_event_loop()
    tasks = [
        loop.run_in_executor(
            None, one_request, base_url, model, prompt, max_tokens, seed
        )
        for _ in range(conc)
    ]
    t0 = time.time()
    texts = await asyncio.gather(*tasks)
    dt = time.time() - t0
    counts = collections.Counter(texts)
    unique = len(counts)
    # mismatches = number of completions not equal to the modal text
    modal, modal_n = counts.most_common(1)[0]
    mismatches = conc - modal_n
    return unique, mismatches, counts, dt, modal


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-V4-Flash")
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--concurrencies", default="2,4,8,16,32,128,256,512")
    ap.add_argument(
        "--prompt",
        default="Explain in detail how a transformer language model performs "
        "autoregressive decoding, step by step, including attention and the "
        "key-value cache.",
    )
    args = ap.parse_args()
    concs = [int(x) for x in args.concurrencies.split(",") if x]
    overall_ok = True
    cross = {}
    for c in concs:
        unique, mismatches, counts, dt, modal = await run_concurrency(
            args.base_url, args.model, args.prompt, args.max_tokens, args.seed, c
        )
        cross[c] = modal
        ok = unique == 1
        overall_ok &= ok
        status = "PASS" if ok else "FAIL"
        print(
            f"[conc={c:4d}] unique_text={unique} mismatches={mismatches} "
            f"wall={dt:.2f}s {status}",
            flush=True,
        )
        if not ok:
            for i, (txt, n) in enumerate(counts.most_common()):
                print(f"    variant{i} (n={n}): {txt[:90]!r}", flush=True)
    # cross-concurrency check: do all concurrency levels agree on the text?
    modal_texts = set(cross.values())
    cross_ok = len(modal_texts) == 1
    print(
        f"\nCROSS-CONCURRENCY: {'PASS' if cross_ok else 'FAIL'} "
        f"({len(modal_texts)} distinct modal texts across levels)",
        flush=True,
    )
    if not cross_ok:
        for c, txt in cross.items():
            print(f"    conc={c}: {txt[:90]!r}", flush=True)
    print(f"\nOVERALL: {'PASS' if (overall_ok and cross_ok) else 'FAIL'}", flush=True)
    sys.exit(0 if (overall_ok and cross_ok) else 1)


if __name__ == "__main__":
    asyncio.run(main())
