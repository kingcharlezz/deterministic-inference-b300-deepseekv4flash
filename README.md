# deterministic-inference-8xb200-deepseek-v4-flash

Deterministic high-throughput inference harness for
`deepseek-ai/DeepSeek-V4-Flash` on 8x NVIDIA B200.

This repository started from an older 1x B300 DeepSeek V4 Flash vLLM setup.
Treat those artifacts only as prior debugging notes. The current target is
8x B200 aggregate serving, DeepSeek-V4-Flash, deterministic engine-level
batch-invariant execution, and at least 5,000 aggregate output tok/s. The
original TP=8 shape is still tested, but current Flash-specific guidance also
requires testing TP=4 with DP=2 across the 8 GPUs.

`temperature=0` is not enough. The server must use deterministic or
batch-invariant execution, and the benchmark checks exact text equality across
batch size, request order, concurrency, and repeated runs.

## What Is Included

- `scripts/serve_sglang_deepseek_v4_flash_8xb200.sh`: primary SGLang launch
  command for 8x B200 with deterministic inference enabled.
- `scripts/serve_vllm_deepseek_v4_flash_8xb200.sh`: vLLM fallback with
  `VLLM_BATCH_INVARIANT=1`.
- `scripts/preflight_8xb200_deepseek_v4_flash.py`: target-host checks for
  exactly 8 visible B200 GPUs, package presence, and required deterministic
  engine flags in SGLang/vLLM help output.
- `scripts/run_8xb200_deepseek_v4_flash_pipeline.py`: one-command target-host
  pipeline that preflights selected engines, tunes passing engines, and writes
  the final proof report.
- `scripts/triage_deepseek_v4_flash_run.py`: scans run logs/results and writes
  JSON/Markdown failure triage with concrete next actions.
- `benchmark/bench_deterministic_inference.py`: backend-aware deterministic
  correctness and throughput probe.
- `scripts/tune_deepseek_v4_flash_8xb200.py`: host-side tuning loop that tries
  deterministic SGLang variants first, then vLLM fallback variants, recording
  server logs, benchmark JSON, and Markdown tables under `runs/`.
- `scripts/summarize_deepseek_v4_flash_run.py`: validates a completed run
  directory and writes the final proof report with launch command, versions,
  GPU, determinism rows, benchmark table, and throughput verdict.
- `tests/test_benchmark_harness.py`: local tests for exact-output comparison,
  mismatch detection, metric aggregation, and benchmark table formatting.
- `tests/test_benchmark_cli_http.py`: local fake-server tests for the benchmark
  CLI against SGLang `/generate` and vLLM/OpenAI streaming completions.
- `docs/20260608-8xb200-deepseek-v4-flash-deterministic-inference.md`: runbook,
  tuning ladder, and acceptance criteria.
- `patches/vllm-0.22.1-batch-invariant.patch`: historical patch from the
  original repo; use only if pinned to that old vLLM build.

## Fresh Machine Setup

These commands assume a Linux host with exactly 8 visible NVIDIA B200 GPUs, a
working driver, internet access, Hugging Face access to DeepSeek-V4-Flash, and
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

Run local benchmark-harness tests before launching on the GPU host:

```bash
python -m unittest discover -s tests
```

If the model requires Hugging Face auth in your environment:

```bash
export HF_TOKEN=hf_...
export HF_HOME="$PWD/hf-cache"
```

Before launching a long model load, verify the target host and installed engine:

```bash
python scripts/preflight_8xb200_deepseek_v4_flash.py --engine sglang
VLLM_BATCH_INVARIANT=1 python scripts/preflight_8xb200_deepseek_v4_flash.py --engine vllm
```

To run preflight, tuning, and final report generation as one target-host flow:

```bash
python scripts/run_8xb200_deepseek_v4_flash_pipeline.py --engines sglang,vllm
```

## SGLang Primary

Install the B200 DeepSeek-V4-Flash compatibility patch for SGLang, then start
with the DeepSeek V4 attention backend:

```bash
source .venv/bin/activate
scripts/apply_sglang_b200_deepseek_v4_flash_patch.sh
bash scripts/serve_sglang_deepseek_v4_flash_8xb200.sh
```

That expands to:

```bash
python -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V4-Flash \
  --host 0.0.0.0 \
  --port 30000 \
  --tp 8 \
  --attention-backend dsv4 \
  --moe-runner-backend flashinfer_mxfp4 \
  --enable-deterministic-inference \
  --mem-fraction-static 0.90
```

Try deterministic attention backends in this order when debugging:

```bash
MOE_RUNNER_BACKEND=flashinfer_mxfp4 ATTENTION_BACKEND=dsv4 bash scripts/serve_sglang_deepseek_v4_flash_8xb200.sh
MOE_RUNNER_BACKEND=flashinfer_mxfp4 FLASHINFER_MXFP4_MOE_PRECISION=bf16 ATTENTION_BACKEND=dsv4 bash scripts/serve_sglang_deepseek_v4_flash_8xb200.sh
TP=4 DP_SIZE=2 MOE_RUNNER_BACKEND=flashinfer_mxfp4 ATTENTION_BACKEND=dsv4 bash scripts/serve_sglang_deepseek_v4_flash_8xb200.sh
TP=4 DP_SIZE=2 MOE_RUNNER_BACKEND= MOE_A2A_BACKEND=megamoe SGLANG_OPT_USE_DEEPGEMM_MEGA_MOE=1 ATTENTION_BACKEND=dsv4 bash scripts/serve_sglang_deepseek_v4_flash_8xb200.sh
MOE_RUNNER_BACKEND=marlin DISABLE_RADIX_CACHE=1 ATTENTION_BACKEND=dsv4 bash scripts/serve_sglang_deepseek_v4_flash_8xb200.sh
MOE_RUNNER_BACKEND=flashinfer_trtllm_routed ATTENTION_BACKEND=dsv4 bash scripts/serve_sglang_deepseek_v4_flash_8xb200.sh
ATTENTION_BACKEND=flashinfer bash scripts/serve_sglang_deepseek_v4_flash_8xb200.sh
ATTENTION_BACKEND=triton bash scripts/serve_sglang_deepseek_v4_flash_8xb200.sh
```

If exact output comparisons vary, keep deterministic inference enabled and test:

```bash
DISABLE_RADIX_CACHE=1 bash scripts/serve_sglang_deepseek_v4_flash_8xb200.sh
TRITON_ATTENTION_SPLIT_TILE_SIZE=128 ATTENTION_BACKEND=triton bash scripts/serve_sglang_deepseek_v4_flash_8xb200.sh
CHUNKED_PREFILL_SIZE=4096 bash scripts/serve_sglang_deepseek_v4_flash_8xb200.sh
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
  --model deepseek-ai/DeepSeek-V4-Flash \
  --hardware-label 8xB200
```

To run the SGLang tuning ladder automatically on the target host:

```bash
python scripts/tune_deepseek_v4_flash_8xb200.py --engines sglang
```

For determinism-first triage, skip throughput gates until exact text is stable:

```bash
python scripts/tune_deepseek_v4_flash_8xb200.py \
  --engines sglang \
  --variants 'sglang-dsv4-tp4-dp2-*' \
  --mode determinism
```

Use `--variants 'sglang-fa3-*'` or an exact variant name to rerun a focused
subset after inspecting logs.

Current local evidence says the missing piece is still the combined path:
attention, MoE, checkpoint quantization, and parallelism must all be
batch-invariant together. `dsv4 + flashinfer_mxfp4` and `dsv4 + marlin` can
serve after local patches, but exact text still changes under concurrent
batching. The `dsv4 + triton + QUANTIZATION=unquant` smoke test still follows
the fp8 checkpoint path and crashed in fused MoE with `Hidden size mismatch`.
The next candidates are `TP=4 DP_SIZE=2` and Blackwell MegaMoE, because current
DeepSeek-V4-Flash serving guidance treats Flash as a 4-GPU B200 shape rather
than the 8-GPU TP shape used for Pro.

## vLLM Fallback

If SGLang cannot load or cannot stay deterministic at target throughput:

```bash
source .venv/bin/activate
bash scripts/serve_vllm_deepseek_v4_flash_8xb200.sh
```

That expands to:

```bash
VLLM_BATCH_INVARIANT=1 vllm serve deepseek-ai/DeepSeek-V4-Flash \
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

For determinism-first debugging on 8x B200, use the serial vLLM baseline before
throughput tuning. It keeps TP=8 but sets `MAX_NUM_SEQS=1`, disables prefix
caching and async scheduling, uses Humming MXFP4 MoE, captures only batch size
1 CUDA graphs, and queues concurrent requests instead of letting them share a
decode batch:

```bash
TP=8 \
MAX_NUM_SEQS=1 \
MAX_NUM_BATCHED_TOKENS=8192 \
MAX_MODEL_LEN=8192 \
MAX_CUDAGRAPH_CAPTURE_SIZE=1 \
MOE_BACKEND=humming \
KV_CACHE_DTYPE=fp8 \
ENABLE_PREFIX_CACHING=0 \
ASYNC_SCHEDULING=0 \
GENERATION_CONFIG=vllm \
VLLM_BATCH_INVARIANT=1 \
VLLM_ENABLE_V1_MULTIPROCESSING=0 \
bash scripts/serve_vllm_deepseek_v4_flash_8xb200.sh
```

This is a deterministic baseline, not the throughput target. If it passes exact
text checks while `MAX_NUM_SEQS>1` drifts, the remaining work is finding or
patching a batch-invariant attention + MoE + quantization path.

## Two verified configs: byte-exact (≤100 conc) vs high-throughput

Determinism and throughput trade off here, because determinism forces every
decode step through a fixed `M=2048` MoE shape whose cost is independent of how
many tokens ride along. Pick the config by what you need.

> **Measuring determinism correctly:** use `scripts/det_probe.py`, which sizes
> its client thread pool to the requested concurrency. A naive probe (Python's
> default executor caps at ~32 threads) only ever exercises ~32-way batching and
> will *report* determinism that does not hold at real concurrency.

### Recommended deployment: byte-exact serving at ≤50 concurrency

For a deployment that serves **≤50 concurrent requests** and wants byte-exact
determinism with per-request responsiveness (NOT aggregate throughput), size the
decode CUDA-graph bucket to the load and use the prefill-priority no-mix
scheduler:

```bash
VLLM_DETERMINISTIC_NO_MIX=1 \
TP=8 \
MAX_NUM_SEQS=64 \
CUDAGRAPH_CAPTURE_SIZES=64 \
MAX_NUM_BATCHED_TOKENS=2048 \
MAX_MODEL_LEN=131072 \
MOE_BACKEND=humming \
KV_CACHE_DTYPE=fp8 \
ENABLE_PREFIX_CACHING=0 \
ASYNC_SCHEDULING=0 \
ENFORCE_EAGER=0 \
GENERATION_CONFIG=vllm \
VLLM_BATCH_INVARIANT=1 \
VLLM_DETERMINISTIC_MODEL_PAD_TOKENS=2048 \
VLLM_HUMMING_MOE_GEMM_TYPE=grouped \
VLLM_ENABLE_V1_MULTIPROCESSING=0 \
bash scripts/serve_vllm_deepseek_v4_flash_8xb200.sh
```

Validated (`scripts/serving_validate.py` + `scripts/det_probe.py`):
- **Determinism gate PASS** — byte-identical output across concurrency 2..50,
  3 repeats, **and** mixed-batch + submission-order invariance (16 distinct
  prompts, 324 occurrences, every prompt one unique completion).
- **Performance @ conc 50** — TTFT p95 ≈ 2.9 s (conc 1: 154 ms); per-request
  decode ≈ 7.3 tok/s. Decode rate is bounded by the byte-exact mechanism (fixed
  2048-row MoE pad + single-channel NCCL); the MoE pad must stay 2048 (1024
  drifts), so per-request decode cannot go faster without custom batch-invariant
  kernels (see `benchmark/det_kernel_poc.py`).

Note the **decode bucket (64) is sized to the load** while the **MoE pad stays
2048** — these are different knobs: the bucket fixes the attention row count
(smaller = lower latency); the pad fixes the MoE reduction shape (must be ≥2048
for determinism). `VLLM_DETERMINISTIC_NO_MIX=1` is prefill-priority: a burst is
admitted in a pure-prefill step, then served by pure-decode steps, so no step
mixes prefill+decode (the drift source) while new requests still prefill
promptly (the decode-priority variant serialized the batch → 35 s TTFT).

### A. Byte-exact deterministic, ≤100 concurrency (recommended for verification)

Same prompt → **byte-identical** continuation, validated at conc 1/32/64/100,
cross-concurrency, repeated sweeps, 200-token outputs, zero variants.

```bash
VLLM_DETERMINISTIC_NO_MIX=1 \
TP=8 \
MAX_NUM_SEQS=128 \
MAX_NUM_BATCHED_TOKENS=2048 \
MAX_MODEL_LEN=131072 \
MOE_BACKEND=humming \
KV_CACHE_DTYPE=fp8 \
ENABLE_PREFIX_CACHING=0 \
ASYNC_SCHEDULING=0 \
ENFORCE_EAGER=0 \
CUDAGRAPH_CAPTURE_SIZES=128 \
GENERATION_CONFIG=vllm \
VLLM_BATCH_INVARIANT=1 \
VLLM_DETERMINISTIC_MODEL_PAD_TOKENS=2048 \
VLLM_HUMMING_MOE_GEMM_TYPE=grouped \
VLLM_ENABLE_V1_MULTIPROCESSING=0 \
bash scripts/serve_vllm_deepseek_v4_flash_8xb200.sh
```

Throughput here is ~250–360 output tok/s at conc ≤100 — low on purpose: the
2048-row MoE pad is mostly idle for ≤100 real tokens, and that per-step cost
dominates. This is the price of byte-exactness on this model.

### B. High throughput, *approximate* determinism (conc 1024)

`MAX_NUM_SEQS=1024 CUDAGRAPH_CAPTURE_SIZES=1024` (drop `VLLM_DETERMINISTIC_NO_MIX`,
`MAX_MODEL_LEN=8192`) sustains **~4,810 output tok/s** at conc 1024
(~2,970 at 512) with short prompts. But it is **not byte-exact** under real
high concurrency: a minority of requests diverge on late tokens (≈78–98%
byte-agreement; **final answers far more stable** — GSM8K matched 1311/1319
between conc 512 and 1024, and accuracy is 95.45%). Use this when aggregate
throughput matters more than exact reproducibility.

### Why byte-exactness needs all four knobs

1. **Fixed MoE/GEMM shape** (`VLLM_DETERMINISTIC_MODEL_PAD_TOKENS=2048`).
   `VLLM_BATCH_INVARIANT=1` only overrides `aten` matmuls; the custom Humming
   MoE / tilelang MHC / FlashMLA kernels keep an `M`-dependent reduction. Padding
   pins it. `M=2048` is empirically the value that works — 1024 still drifts.
2. **Single decode CUDA-graph bucket** (`CUDAGRAPH_CAPTURE_SIZES=N`). Otherwise
   CUDA-graph bucketing makes a request's attention `M` depend on batch
   composition (stragglers land in small buckets) → timing-dependent drift.
3. **Fixed prefill `M`** (`MAX_NUM_BATCHED_TOKENS=2048`). Every prefill batch is
   ≤2048 → padded to exactly 2048, so a prompt's prefill output is independent of
   how many prompts co-batch. Letting it float (e.g. `bt=8192`) makes conc>68
   prefill at a larger `M` and produce a *different* output than low concurrency.
4. **No mixed prefill+decode steps** (`VLLM_DETERMINISTIC_NO_MIX=1`, a V1
   scheduler patch). The scheduler otherwise co-schedules a new prefill with
   in-flight decodes; the decode token computed in that variable-shape mixed step
   drifts vs a pure-decode step. The gate defers new prefills whenever a decode
   is already scheduled, so every step is pure-prefill or pure-decode.

Approaches that were tested and **rejected** for determinism: pad=1024 (drifts),
prefill-pad=8192 (drifts), MTP speculative decoding (clean at ≤64 but 30 variants
at conc 100, only ~+12% throughput).

Caveat: byte-exactness is validated for prompts that prefill within one
`MAX_NUM_BATCHED_TOKENS` step (≤2048 tokens). Longer prompts chunk across steps
and can re-mix with decodes; raise `MAX_NUM_BATCHED_TOKENS` above the longest
prompt (keeping it the single fixed pad value) if you need exactness on long
inputs.

Verify:

```bash
# byte-identical continuation at real concurrency (executor sized to --concurrencies)
python scripts/det_probe.py --concurrencies 1,32,64,100 --max-tokens 160
# throughput (read server-side "generation throughput")
python scripts/load_gen.py --concurrency 100 --max-tokens 256 --duration 60 --api-key "$VLLM_API_KEY"
```

Install the deterministic vLLM edits with
`scripts/apply_vllm_batch_invariant_patch.sh` (applies the base patch, overlays
`patches/dsv4-deterministic/{attention.py,nvidia/model.py}`, and overlays the
no-mix V1 `scheduler.py`).

### Task quality (GSM8K)

The deterministic config preserves task quality. `benchmark/gsm8k_eval.py` runs
the standard 8-shot CoT GSM8K test set (1319 problems, greedy) over the
`/v1/completions` endpoint:

```bash
curl -s -o benchmark/data/gsm8k_test.jsonl \
  https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl
python benchmark/gsm8k_eval.py --concurrency 1024 --max-tokens 512 \
  --out runs/det_gsm8k.jsonl --label DET
```

Deterministic conc-1024 config: **95.45%** (1259/1319). Note that GSM8K's
~950-token 8-shot prompts are prefill/KV-bound, so the server admits ~210-240
sequences simultaneously (not the full 1024 client offered-load) and runs at
~670 tok/s — the multi-thousand tok/s figures above need the short-prompt,
decode-bound regime.

Determinism caveat: byte-identical reproduction holds for prompts that fit in a
single prefill chunk (the synthetic probe). For long prompts, chunked-prefill
boundaries depend on batch composition, so full text can still drift across
concurrency levels (final answers matched on 1311/1319 between conc 512 and
1024); set `MAX_NUM_BATCHED_TOKENS` >= the longest prompt to remove that source.

Run the fallback probe:

```bash
python benchmark/bench_deterministic_inference.py \
  --backend openai-completions \
  --base-url http://127.0.0.1:8000 \
  --model deepseek-ai/DeepSeek-V4-Flash \
  --hardware-label 8xB200
```

To try SGLang first and continue into vLLM fallback variants until one reaches
the deterministic throughput gate:

```bash
python scripts/tune_deepseek_v4_flash_8xb200.py --engines sglang,vllm
```

The full pipeline wrapper runs this tuner after preflighting each selected
engine and skips engines that fail preflight.

When a run fails, the full pipeline writes `triage.json` and `triage.md`
automatically. You can also triage an existing run directory manually:

```bash
python scripts/triage_deepseek_v4_flash_run.py runs/<timestamp>
```

Each attempt writes `server.log`, `benchmark.log`, `result.json`, and
`benchmark.md` under `runs/<timestamp>/<variant>/`. The runner exits `0` on
the first deterministic result at or above 5,000 output tok/s.

After a passing run, generate the final proof report:

```bash
python scripts/summarize_deepseek_v4_flash_run.py runs/<timestamp>
```

This writes `runs/<timestamp>/final_report.md` and exits non-zero if the result
does not prove deterministic DeepSeek-V4-Flash inference on `8xB200` at or above
5,000 aggregate output tok/s.

## Acceptance Criteria

The benchmark repeats same-prompt checks at concurrency `1,8,32,128`, repeats
each level 3 times, verifies mixed-prompt batch-size invariance at the same
determinism concurrencies, verifies prompt order invariance, then benchmarks
concurrency `1,4,8,16,32,64,128,256`.

It reports output tok/s, prompt tok/s, total tok/s, req/s, TTFT p50/p95/p99,
and latency p50/p95/p99. It exits non-zero if deterministic text comparison
fails, if best output throughput is below 5,000 tok/s, or if throughput is
below 2,500 tok/s, which should be treated as a misconfiguration. The stretch
target is 8,000 output tok/s.
