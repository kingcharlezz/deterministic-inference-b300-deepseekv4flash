# 8x B200 DeepSeek-V4-Flash Deterministic Inference

This runbook is for deterministic high-throughput inference with
`deepseek-ai/DeepSeek-V4-Flash` on 8x NVIDIA B200. The old 1x B300 DeepSeek V4
Flash setup is only historical context; all launch defaults, tensor
parallelism, model names, request parameters, and throughput gates here target
the new 8x B200 machine.

## Baseline Inspection

Run these before installing or launching:

```bash
find . -maxdepth 3 -type f | sort
nvidia-smi
python3 --version
python3 -m pip freeze | egrep 'sglang|vllm|torch|triton|flashinfer|flash-attn'
python3 -m sglang.launch_server --help || true
vllm serve --help || true
```

Expected hardware is exactly 8 visible NVIDIA B200 GPUs. If `nvidia-smi` cannot
talk to the driver, no local throughput result is trustworthy.

Package versions checked against PyPI on 2026-06-07:

- SGLang primary: `sglang[all]==0.5.12.post1`;
- vLLM fallback: `vllm==0.22.1`;
- benchmark client: `httpx==0.28.1`.

Install only one engine stack per virtual environment unless the target host
image has already validated that the SGLang and vLLM dependency sets coexist.

Local harness checks that do not require GPUs:

```bash
python -m unittest discover -s tests
python -m py_compile benchmark/bench_deterministic_inference.py \
  scripts/preflight_8xb200_deepseek_v4_flash.py \
  scripts/tune_deepseek_v4_flash_8xb200.py \
  scripts/summarize_deepseek_v4_flash_run.py \
  scripts/run_8xb200_deepseek_v4_flash_pipeline.py \
  tests/test_benchmark_harness.py \
  tests/test_benchmark_cli_http.py
```

Target-host preflight checks:

```bash
python scripts/preflight_8xb200_deepseek_v4_flash.py --engine sglang
VLLM_BATCH_INVARIANT=1 python scripts/preflight_8xb200_deepseek_v4_flash.py --engine vllm
```

The preflight exits non-zero if `nvidia-smi` does not show exactly 8 B200 GPUs,
the engine package is missing, the deterministic SGLang flag is absent, required
vLLM serve flags are absent, or `VLLM_BATCH_INVARIANT=1` is missing for vLLM.

One-command target-host pipeline:

```bash
python scripts/run_8xb200_deepseek_v4_flash_pipeline.py --engines sglang,vllm
```

This writes `pipeline.json`, per-engine preflight JSON/logs, the tuner logs,
the final proof report, and failure triage artifacts. It only tunes engines
that pass preflight.

## Determinism Requirements

The client request shape must be deterministic, but that is not sufficient.
The engine also has to use deterministic, batch-invariant execution.

SGLang request parameters:

```json
{"temperature":0,"top_p":1,"top_k":-1,"max_new_tokens":256}
```

vLLM request parameters:

```json
{"temperature":0,"top_p":1,"max_tokens":256,"seed":42}
```

No speculative decoding is used for the baseline. Do not tune with speculation
until the deterministic baseline has passed exact comparison and 5,000 output
tok/s.

## SGLang Primary Launch

```bash
scripts/apply_sglang_b200_deepseek_v4_flash_patch.sh
bash scripts/serve_sglang_deepseek_v4_flash_8xb200.sh
```

Equivalent command:

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

SGLang documents deterministic inference via
`--enable-deterministic-inference`. On B200, DeepSeek-V4-Flash needs the
DeepSeek V4 attention path first; `fa3` is kept only as a diagnostic because
the current FA3 guard rejects SM100.

### SGLang Tuning Ladder

Keep `--enable-deterministic-inference` and DeepSeek-V4-Flash fixed. Test the
original single-process `--tp 8` shape, but prioritize documented Flash shapes
such as `TP=4 DP_SIZE=2` when TP=8 is nondeterministic or slow.

1. Start with `ATTENTION_BACKEND=dsv4` and
   `MOE_RUNNER_BACKEND=flashinfer_mxfp4`.
2. If MXFP4 precision or kernels fail, try
   `FLASHINFER_MXFP4_MOE_PRECISION=bf16`.
3. Test `TP=4 DP_SIZE=2` before spending more time on single-process TP=8.
   Current upstream recipes describe DeepSeek-V4-Flash as a 4-GPU B200 serving
   target, so 8x B200 aggregate serving may need two TP=4 data-parallel
   replicas rather than one TP=8 graph.
4. Test Blackwell MegaMoE as the next high-throughput MoE path:
   `TP=4 DP_SIZE=2 MOE_RUNNER_BACKEND= MOE_A2A_BACKEND=megamoe
   SGLANG_OPT_USE_DEEPGEMM_MEGA_MOE=1`. Do not set
   `--moe-runner-backend` on this candidate. The tuner sets
   `SGLANG_OPT_DEEPGEMM_MEGA_MOE_NUM_MAX_TOKENS_PER_RANK=8320` for the
   high-throughput recipe and also tests W4A4 by enabling
   `SGLANG_OPT_DEEPGEMM_MEGA_MOE_USE_FP4_ACTS=1` and
   `SGLANG_OPT_DEEPGEMM_MEGA_MOE_USE_MXF4_KIND=1`.
5. Test `MOE_RUNNER_BACKEND=marlin` only as a diagnostic. The local
   Blackwell guard patch lets it start on B200, but the observed exact-output
   probe was not batch-invariant under concurrency.
6. Keep `MOE_RUNNER_BACKEND=flashinfer_trtllm_routed` only as a diagnostic;
   DeepSeek-V4-Flash declares `expert_dtype: fp4`, so routed FP8 TRT-LLM is
   not the first-choice expert kernel.
7. If load or kernels fail, try `ATTENTION_BACKEND=flashinfer`.
8. If outputs vary or FlashInfer fails, try `ATTENTION_BACKEND=triton`.
9. If outputs vary, set `DISABLE_RADIX_CACHE=1`.
10. For Triton variance or performance issues, try
   `TRITON_ATTENTION_SPLIT_TILE_SIZE=64`, `128`, and `256`.
11. Treat `QUANTIZATION=unquant` with `MOE_RUNNER_BACKEND=triton` as a smoke
   test, not a proven dequantized path. In local testing, SGLang still logged
   `quant=fp8` for this checkpoint and crashed with `Hidden size mismatch`.
12. If prefill OOMs, reduce static memory or chunk size:
   `MEM_FRACTION_STATIC=0.86` or `CHUNKED_PREFILL_SIZE=4096`.
13. If throughput is below 2,500 output tok/s, assume misconfiguration:
   verify all 8 GPUs, the intended TP/DP shape, deterministic backend support,
   no CPU fallback, no accidental model/config mismatch, and no stale model
   cache.

### Local 2026-06-08 Negative Evidence

These runs do not satisfy the goal, but they narrow the backend search.

- `ATTENTION_BACKEND=dsv4 MOE_RUNNER_BACKEND=marlin DISABLE_RADIX_CACHE=1`
  initially failed during weight postprocess because the SGLang Marlin guard
  accepted Hopper/SM90 but not Blackwell.
- After patching the guard to accept Blackwell, the Marlin server reached
  `/health`, but exact output varied under concurrency. The short probe saw
  mismatches at `max_new_tokens=4` and larger divergence at `8`, `16`, `32`,
  and `64`.
- `ATTENTION_BACKEND=dsv4 MOE_RUNNER_BACKEND=triton QUANTIZATION=unquant`
  did not bypass the checkpoint quantization path; logs still showed
  `quant=fp8`, then the fused Triton MoE path crashed with
  `AssertionError: Hidden size mismatch`.
- After the Triton crash, `nvidia-smi` stopped communicating with the driver.
  More live validation requires a healthy driver or node reset before any
  throughput or determinism result can be trusted.

## vLLM Fallback Launch

```bash
bash scripts/serve_vllm_deepseek_v4_flash_8xb200.sh
```

Equivalent command:

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

For offline vLLM probes, also test:

```bash
export VLLM_ENABLE_V1_MULTIPROCESSING=0
```

### vLLM Tuning Ladder

Keep `VLLM_BATCH_INVARIANT=1` exported before process start.

1. Start with TP=8, `--max-num-seqs 256`, and
   `--max-num-batched-tokens 8192`.
2. Raise `MAX_NUM_BATCHED_TOKENS` to `16384` if memory allows.
3. Try `MAX_NUM_SEQS=128`, `192`, `256`, then `384`.
4. Adjust `GPU_MEMORY_UTILIZATION=0.86`, `0.90`, and `0.94` after OOMs.
5. Keep `--max-model-len 8192` for the throughput baseline unless the model
   requires a longer context to load correctly.
6. If outputs vary, stop performance tuning and verify batch invariance,
   fixed seed, request parameters, and server version first.

## Determinism And Benchmark Probe

Primary SGLang run:

```bash
python benchmark/bench_deterministic_inference.py \
  --backend sglang-native \
  --base-url http://127.0.0.1:30000 \
  --model deepseek-ai/DeepSeek-V4-Flash \
  --hardware-label 8xB200
```

vLLM fallback run:

```bash
python benchmark/bench_deterministic_inference.py \
  --backend openai-completions \
  --base-url http://127.0.0.1:8000 \
  --model deepseek-ai/DeepSeek-V4-Flash \
  --hardware-label 8xB200
```

The probe performs:

- same prompt byte-for-byte;
- concurrency `1,8,32,128`;
- 3 repeats per determinism concurrency;
- exact text comparison;
- mixed-prompt batch-size invariance at each determinism concurrency;
- forward vs reverse order invariance;
- throughput benchmark at concurrency `1,4,8,16,32,64,128,256`.

Metrics reported:

- output tok/s;
- prompt tok/s;
- total tok/s;
- req/s;
- TTFT p50/p95/p99;
- latency p50/p95/p99.

The command exits with:

- `0` when deterministic checks pass and best output tok/s is at least 5,000;
- `2` when deterministic checks pass but throughput is below 5,000;
- `3` when best output tok/s is below 2,500, which is treated as
  misconfiguration;
- non-zero assertion/HTTP errors when exact deterministic comparison fails or
  the endpoint crashes.

## Automated Tuning Runner

On the target 8x B200 host, use the runner to execute the documented ladder and
keep an audit trail:

```bash
python scripts/tune_deepseek_v4_flash_8xb200.py --engines sglang,vllm
```

To rerun one configuration or a focused subset:

```bash
python scripts/tune_deepseek_v4_flash_8xb200.py \
  --engines sglang \
  --variants 'sglang-fa3-*'
```

The SGLang sequence tries:

- `dsv4`, `flashinfer`, `fa3`, and `triton` around the TP=8 baseline;
- `TP=4 DP_SIZE=2` with MXFP4 and Blackwell MegaMoE candidates;
- radix cache disabled when needed;
- Triton split tile sizes `64`, `128`, and `256`;
- chunked prefill `4096` with memory fraction `0.86`.

The vLLM fallback sequence keeps `VLLM_BATCH_INVARIANT=1`, TP=8, seed `0`, and
`VLLM_ENABLE_V1_MULTIPROCESSING=0`, then varies batch size, batched tokens, and
GPU memory utilization.

Every attempt writes:

- `server.log`;
- `benchmark.log`;
- `result.json`;
- `benchmark.md`;
- top-level `summary.json` and `summary.md`.
- top-level `environment.json`;
- top-level `pipeline.json` when using the full pipeline wrapper.
- top-level `triage.json` and `triage.md` when a run fails.

The runner stops at the first deterministic benchmark that exits `0`, which
means best aggregate output throughput is at least 5,000 tok/s. If all variants
fail, inspect the per-variant logs in order; a result below 2,500 output tok/s
should be treated as misconfiguration rather than a tuned result.

Manual failure triage:

```bash
python scripts/triage_deepseek_v4_flash_run.py runs/<timestamp>
```

The triage script recognizes missing GPU drivers, wrong GPU inventory, missing
packages, unsupported flags, Hugging Face auth failures, CUDA OOMs, kernel/arch
errors, NCCL/distributed failures, determinism mismatches, and below-target
throughput.

Generate and validate the final proof report after a passing run:

```bash
python scripts/summarize_deepseek_v4_flash_run.py runs/<timestamp>
```

The summary script reads `summary.json`, the passing variant's `result.json`,
`benchmark.md`, and `environment.json`. It writes `final_report.md` and exits
`0` only when exact determinism checks passed, the model is
`deepseek-ai/DeepSeek-V4-Flash`, the hardware label is `8xB200`, the environment
snapshot contains exactly eight B200 GPU rows, package versions are recorded,
and best output throughput is at least 5,000 tok/s.

## Result Template

Record each run with:

```text
engine:
launch command:
engine deterministic flags:
request params:
package versions:
GPU:
determinism same-prompt:
determinism mixed-batch:
determinism order:
best concurrency:
best output tok/s:
benchmark table:
proof report:
notes:
```
