#!/usr/bin/env python3
"""GSM8K 8-shot CoT eval over an OpenAI-compatible /v1/completions endpoint.

Greedy (temp=0). Fires `concurrency` requests in flight via a worker pool.
Records every problem's full output so results can be re-graded / compared
across server configs. Designed to compare a deterministic vLLM config against
a regular one and check that task accuracy is preserved.
"""
import argparse
import asyncio
import json
import re
import time

import aiohttp

# Canonical 8-shot chain-of-thought exemplars (Wei et al. 2022), the standard
# GSM8K prompt. Identical across all server configs so prompt sensitivity cancels.
FEWSHOT = [
    ("There are 15 trees in the grove. Grove workers will plant trees in the grove "
     "today. After they are done, there will be 21 trees. How many trees did the "
     "grove workers plant today?",
     "There are 15 trees originally. Then there were 21 trees after some more were "
     "planted. So there must have been 21 - 15 = 6. The answer is 6."),
    ("If there are 3 cars in the parking lot and 2 more cars arrive, how many cars "
     "are in the parking lot?",
     "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5."),
    ("Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces "
     "do they have left in total?",
     "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had "
     "32 + 42 = 74. After eating 35, they had 74 - 35 = 39. The answer is 39."),
    ("Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 "
     "lollipops. How many lollipops did Jason give to Denny?",
     "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So "
     "he gave Denny 20 - 12 = 8. The answer is 8."),
    ("Shawn has five toys. For Christmas, he got two toys each from his mom and dad. "
     "How many toys does he have now?",
     "Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then "
     "that is 4 more toys. 5 + 4 = 9. The answer is 9."),
    ("There were nine computers in the server room. Five more computers were "
     "installed each day, from monday to thursday. How many computers are now in the "
     "server room?",
     "There were originally 9 computers. For each of 4 days, 5 more computers were "
     "added. So 5 * 4 = 20 computers were added. 9 + 20 = 29. The answer is 29."),
    ("Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he "
     "lost 2 more. How many golf balls did he have at the end of wednesday?",
     "Michael started with 58 golf balls. After losing 23 on tuesday, he had "
     "58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls. The answer is 33."),
    ("Olivia has $23. She bought five bagels for $3 each. How much money does she "
     "have left?",
     "Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. "
     "So she has 23 - 15 = 8 dollars left. The answer is 8."),
]

PREAMBLE = "".join(
    f"Question: {q}\nAnswer: {a}\n\n" for q, a in FEWSHOT
)


def build_prompt(question: str) -> str:
    return PREAMBLE + f"Question: {question}\nAnswer:"


def extract_gold(answer_field: str) -> str:
    # GSM8K gold answers end with "#### N"
    m = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", answer_field)
    return _norm_num(m.group(1)) if m else ""


def _norm_num(s: str) -> str:
    s = s.replace(",", "").replace("$", "").strip()
    try:
        f = float(s)
        return str(int(f)) if f == int(f) else str(f)
    except ValueError:
        return s


def extract_pred(text: str) -> str:
    # Prefer the last "answer is X"; fallback to the last number in the text.
    matches = re.findall(r"answer is\s*\$?(-?[\d,]+(?:\.\d+)?)", text, flags=re.I)
    if matches:
        return _norm_num(matches[-1])
    nums = re.findall(r"-?[\d,]+(?:\.\d+)?", text)
    return _norm_num(nums[-1]) if nums else ""


async def worker(name, queue, session, url, model, max_tokens, results, stats):
    while True:
        item = await queue.get()
        if item is None:
            queue.task_done()
            return
        idx, q, gold = item
        body = {
            "model": model,
            "prompt": build_prompt(q),
            "max_tokens": max_tokens,
            "temperature": 0,
            "top_p": 1,
            "seed": 42,
            "stop": ["Question:"],
            "stream": False,
        }
        text = ""
        for attempt in range(4):
            try:
                async with session.post(url, json=body) as r:
                    data = await r.json()
                text = data["choices"][0]["text"]
                break
            except Exception:  # noqa: BLE001 - transient, retry
                await asyncio.sleep(0.1 * (attempt + 1))
        pred = extract_pred(text)
        correct = (pred == gold and gold != "")
        results[idx] = {
            "idx": idx,
            "question": q,
            "gold": gold,
            "pred": pred,
            "correct": correct,
            "output": text,
        }
        stats["done"] += 1
        if correct:
            stats["correct"] += 1
        queue.task_done()


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-V4-Flash")
    ap.add_argument("--data", default="benchmark/data/gsm8k_test.jsonl")
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--concurrency", type=int, default=256)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--out", required=True, help="path to write per-problem JSONL")
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    problems = []
    with open(args.data) as f:
        for i, line in enumerate(f):
            d = json.loads(line)
            problems.append((i, d["question"], extract_gold(d["answer"])))
    if args.limit:
        problems = problems[: args.limit]

    url = args.base_url.rstrip("/") + "/v1/completions"
    results = {}
    stats = {"done": 0, "correct": 0}
    queue = asyncio.Queue()
    for p in problems:
        queue.put_nowait(p)
    conn = aiohttp.TCPConnector(limit=0, limit_per_host=0)
    timeout = aiohttp.ClientTimeout(total=None, sock_connect=30, sock_read=900)
    t0 = time.monotonic()
    async with aiohttp.ClientSession(connector=conn, timeout=timeout) as session:
        workers = [
            asyncio.create_task(
                worker(w, queue, session, url, args.model, args.max_tokens, results, stats)
            )
            for w in range(args.concurrency)
        ]
        await queue.join()
        for _ in workers:
            queue.put_nowait(None)
        await asyncio.gather(*workers)
    dt = time.monotonic() - t0

    n = len(problems)
    acc = stats["correct"] / n if n else 0.0
    with open(args.out, "w") as f:
        for idx in sorted(results):
            f.write(json.dumps(results[idx]) + "\n")
    summary = {
        "label": args.label,
        "n": n,
        "correct": stats["correct"],
        "accuracy": acc,
        "concurrency": args.concurrency,
        "wall_s": round(dt, 1),
        "out": args.out,
    }
    print(json.dumps(summary), flush=True)
    print(
        f"[{args.label}] GSM8K {n} problems  acc={acc:.4f} "
        f"({stats['correct']}/{n})  conc={args.concurrency}  wall={dt:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    asyncio.run(main())
