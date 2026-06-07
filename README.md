# deterministic-inference-8xb200-deepseek-v4-pro

Deterministic high-throughput inference harness for
`deepseek-ai/DeepSeek-V4-Pro` on 8x NVIDIA B200.

This repository started from an older 1x B300 DeepSeek V4 Flash vLLM setup.
Treat those artifacts only as prior debugging notes. The current target is
8x B200, tensor parallel size 8, DeepSeek-V4-Pro, deterministic engine-level
batch-invariant execution, and at least 5,000 aggregate output tok/s.

`temperature=0` is not enough. The server must use deterministic or
batch-invariant execution, and the benchmark checks exact text equality across
batch size, request order, concurrency, and repeated runs.

## What Is Included

- `scripts/serve_sglang_deepseek_v4_pro_8xb200.sh`: primary SGLang launch
  command for 8x B200 with deterministic inference enabled.
- `scripts/serve_vllm_deepseek_v4_pro_8xb200.sh`: vLLM fallback with
  `VLLM_BATCH_INVARIANT=1`.
- `benchmark/bench_deterministic_inference.py`: backend-aware deterministic
  correctness and throughput probe.
- `scripts/tune_deepseek_v4_pro_8xb200.py`: host-side tuning loop that tries
  deterministic SGLang variants first, then vLLM fallback variants, recording
  server logs, benchmark JSON, and Markdown tables under `runs/`.
- `docs/20260607-8xb200-deepseek-v4-pro-deterministic-inference.md`: runbook,
  tuning ladder, and acceptance criteria.
- `patches/vllm-0.22.1-batch-invariant.patch`: historical patch from the
  original repo; use only if pinned to that old vLLM build.

## Fresh Machine Setup

These commands assume a Linux host with exactly 8 visible NVIDIA B200 GPUs, a
working driver, internet access, Hugging Face access to DeepSeek-V4-Pro, and
Python 3.12.

```bash
git clone https://github.com/kingcharlezz/deterministic-inference-b300-deepseekv4flash.git
cd deterministic-inference-b300-deepseekv4flash

python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-sglang.txt
```

Use `requirements-vllm.txt` instead when validating the vLLM fallback in a
separate environment. Avoid installing both engines into the same environment
unless the target host image is known to support that combination.

If the model requires Hugging Face auth in your environment:

```bash
export HF_TOKEN=hf_...
export HF_HOME="$PWD/hf-cache"
```

## SGLang Primary

Start with FA3:

```bash
source .venv/bin/activate
bash scripts/serve_sglang_deepseek_v4_pro_8xb200.sh
```

That expands to:

```bash
python -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V4-Pro \
  --host 0.0.0.0 \
  --port 30000 \
  --tp 8 \
  --attention-backend fa3 \
  --enable-deterministic-inference \
  --mem-fraction-static 0.90
```

Try deterministic attention backends in this order when debugging:

```bash
ATTENTION_BACKEND=fa3 bash scripts/serve_sglang_deepseek_v4_pro_8xb200.sh
ATTENTION_BACKEND=flashinfer bash scripts/serve_sglang_deepseek_v4_pro_8xb200.sh
ATTENTION_BACKEND=triton bash scripts/serve_sglang_deepseek_v4_pro_8xb200.sh
```

If exact output comparisons vary, keep deterministic inference enabled and test:

```bash
DISABLE_RADIX_CACHE=1 bash scripts/serve_sglang_deepseek_v4_pro_8xb200.sh
TRITON_ATTENTION_SPLIT_TILE_SIZE=128 ATTENTION_BACKEND=triton bash scripts/serve_sglang_deepseek_v4_pro_8xb200.sh
CHUNKED_PREFILL_SIZE=4096 bash scripts/serve_sglang_deepseek_v4_pro_8xb200.sh
```

SGLang benchmark requests use:

```json
{"temperature":0,"top_p":1,"top_k":-1,"max_new_tokens":256}
```

Run the probe:

```bash
python benchmark/bench_deterministic_inference.py \
  --backend sglang-native \
  --base-url http://127.0.0.1:30000 \
  --model deepseek-ai/DeepSeek-V4-Pro \
  --hardware-label 8xB200
```

To run the SGLang tuning ladder automatically on the target host:

```bash
python scripts/tune_deepseek_v4_pro_8xb200.py --engines sglang
```

Use `--variants 'sglang-fa3-*'` or an exact variant name to rerun a focused
subset after inspecting logs.

## vLLM Fallback

If SGLang cannot load or cannot stay deterministic at target throughput:

```bash
source .venv/bin/activate
bash scripts/serve_vllm_deepseek_v4_pro_8xb200.sh
```

That expands to:

```bash
VLLM_BATCH_INVARIANT=1 vllm serve deepseek-ai/DeepSeek-V4-Pro \
  --host 0.0.0.0 \
  --port 8000 \
  --seed 0 \
  --tensor-parallel-size 8 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 8192 \
  --max-num-seqs 256 \
  --max-num-batched-tokens 8192
```

vLLM benchmark requests use:

```json
{"temperature":0,"top_p":1,"max_tokens":256,"seed":42}
```

Run the fallback probe:

```bash
python benchmark/bench_deterministic_inference.py \
  --backend openai-completions \
  --base-url http://127.0.0.1:8000 \
  --model deepseek-ai/DeepSeek-V4-Pro \
  --hardware-label 8xB200
```

To try SGLang first and continue into vLLM fallback variants until one reaches
the deterministic throughput gate:

```bash
python scripts/tune_deepseek_v4_pro_8xb200.py --engines sglang,vllm
```

Each attempt writes `server.log`, `benchmark.log`, `result.json`, and
`benchmark.md` under `runs/<timestamp>/<variant>/`. The runner exits `0` on
the first deterministic result at or above 5,000 output tok/s.

## Acceptance Criteria

The benchmark repeats same-prompt checks at concurrency `1,8,32,128`, repeats
each level 3 times, verifies prompt order invariance, then benchmarks
concurrency `1,4,8,16,32,64,128,256`.

It reports output tok/s, prompt tok/s, total tok/s, req/s, TTFT p50/p95/p99,
and latency p50/p95/p99. It exits non-zero if deterministic text comparison
fails, if best output throughput is below 5,000 tok/s, or if throughput is
below 2,500 tok/s, which should be treated as a misconfiguration. The stretch
target is 8,000 output tok/s.
