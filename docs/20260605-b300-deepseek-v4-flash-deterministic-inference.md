# B300 DeepSeek V4 Flash Deterministic Inference

This note records the validation harness used to check deterministic
OpenAI-compatible vLLM inference for `deepseek-ai/deepseek-v4-flash` on B300.
It keeps the server launch, request shape, and acceptance checks in the
repository so future runs can reproduce the same determinism and throughput
probe.

## Fresh Machine Setup

Start from a Linux B300 machine with a working NVIDIA driver and Python 3.12:

```bash
git clone https://github.com/kingcharlezz/deterministic-inference-b300-deepseekv4flash.git
cd deterministic-inference-b300-deepseekv4flash

python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
bash scripts/apply_vllm_batch_invariant_patch.sh
```

The run used `vllm==0.22.1`. The important custom deterministic requirement is
that the vLLM build must be patched to expose `VLLM_BATCH_INVARIANT` and the
deterministic greedy-logit controls.

```bash
python - <<'PY'
import importlib.metadata as md
import vllm.envs as envs

print("vllm", md.version("vllm"))
assert hasattr(envs, "VLLM_BATCH_INVARIANT")
assert hasattr(envs, "VLLM_DETERMINISTIC_LOGIT_BAND")
assert hasattr(envs, "VLLM_DETERMINISTIC_LOGIT_QUANTUM")
print("VLLM_BATCH_INVARIANT support is present")
PY
```

If the model requires Hugging Face auth, set `HF_TOKEN` before starting the
server. Set `HF_HOME` to keep downloaded model files in a stable location:

```bash
export HF_HOME="$PWD/hf-cache"
```

## Request Settings

The benchmark sends `/v1/completions` requests with:

- `temperature=0`
- `top_p=1`
- `seed=42`
- `max_tokens=256`

The deterministic pass compares exact response text for the same prompt across multiple concurrent request levels, then runs distinct prompts in forward and reverse order to catch batch-order dependent outputs.

## Server Settings

The server side must request vLLM's batch-invariant mode:

```bash
export VLLM_BATCH_INVARIANT=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export CUDA_VISIBLE_DEVICES=0
```

`VLLM_BATCH_INVARIANT=1` is the non-negotiable deterministic piece. In the vLLM
build used here, it routes execution through batch-invariant attention,
linear/MoE, normalization, and greedy sampling paths and disables cascade
attention. Fixed sampling parameters alone do not provide that guarantee.

The patch file is `patches/vllm-0.22.1-batch-invariant.patch`. It preserves the
local vLLM changes used on the original machine, including DeepSeek V4 decode
padding to fixed `max_num_seqs` geometry, deterministic greedy-logit controls,
and batch-invariant hooks in the affected attention, linear/MoE, and routing
paths.

Start the OpenAI-compatible server with:

```bash
bash scripts/serve_b300_deepseek_v4_flash.sh
```

That script expands to the live command used for this setup:

```bash
vllm serve deepseek-ai/deepseek-v4-flash \
  --host 0.0.0.0 \
  --port 8000 \
  --seed 0 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.94 \
  --max-model-len 131072 \
  --max-num-seqs 50 \
  --max-num-batched-tokens 32768 \
  --enable-prefix-caching \
  --kv-cache-dtype fp8 \
  --moe-backend triton_unfused \
  --attention-backend TRITON_MLA \
  --served-model-name deepseek-ai/deepseek-v4-flash \
  --generation-config vllm \
  --async-scheduling \
  --cudagraph-capture-sizes 1 2 4 8 16 24 32 40 48 50 \
  --max-cudagraph-capture-size 50
```

## Running The Probe

Start the vLLM OpenAI-compatible server separately, then run:

```bash
python benchmark/bench_vllm_deterministic_inference.py \
  --base-url http://127.0.0.1:8000 \
  --model deepseek-ai/deepseek-v4-flash \
  --hardware-label B300 \
  --determinism-concurrencies 1,8,32,128,300 \
  --concurrencies 1,4,8,16,32,64,128,256,300 \
  --min-requests 300
```

Set `OPENAI_API_KEY` or pass `--api-key` when the server requires an authorization header.

The command prints JSON for machine-readable records and a Markdown throughput table for quick comparison. It exits non-zero when any deterministic comparison fails or when the best streamed output throughput is below `--min-output-tok-s`.
