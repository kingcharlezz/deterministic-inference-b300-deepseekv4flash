# deterministic-inference-b300-deepseekv4flash

Reproducible deterministic inference harness for
`deepseek-ai/deepseek-v4-flash` on B300 using an OpenAI-compatible vLLM
endpoint.

This setup is intentionally specific. The deterministic run depended on both
client-side request settings and vLLM's batch-invariant execution mode. Greedy
sampling with `temperature=0` and a fixed seed is not enough by itself if the
server can change outputs based on batch composition or request ordering.

## What Is Included

- `benchmark/bench_vllm_deterministic_inference.py`: same-prompt determinism,
  order-invariance, and throughput checks.
- `scripts/serve_b300_deepseek_v4_flash.sh`: the vLLM server command used for
  the B300 DeepSeek V4 Flash run.
- `docs/20260605-b300-deepseek-v4-flash-deterministic-inference.md`: detailed
  runbook and notes on the deterministic pieces.
- `requirements.txt`: Python packages for a fresh-machine setup.

## Fresh Machine Setup

These commands assume a Linux B300 host with a working NVIDIA driver, internet
access, and Python 3.12 available. The live run used `CUDA_VISIBLE_DEVICES=0`,
`vllm==0.22.1`, `torch==2.11.0`, CUDA 13.3 Python wheels, and vLLM's
`VLLM_BATCH_INVARIANT=1` path.

```bash
git clone https://github.com/kingcharlezz/deterministic-inference-b300-deepseekv4flash.git
cd deterministic-inference-b300-deepseekv4flash

python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
bash scripts/apply_vllm_batch_invariant_patch.sh
```

If the model requires Hugging Face auth in your environment, set a token before
starting the server:

```bash
export HF_TOKEN=hf_...
```

Use a persistent model/cache directory so restarts do not redownload weights:

```bash
export HF_HOME="$PWD/hf-cache"
```

## Verify The Deterministic vLLM Build

The server must use the patched vLLM build that exposes `VLLM_BATCH_INVARIANT`
and the deterministic greedy-logit controls. Check that before serving:

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

If this check fails, the machine does not have the deterministic vLLM support
used by this run. Re-run `bash scripts/apply_vllm_batch_invariant_patch.sh`
after installing `requirements.txt`.

## Start The Server

In terminal 1:

```bash
source .venv/bin/activate
export HF_HOME="${HF_HOME:-$PWD/hf-cache}"
bash scripts/serve_b300_deepseek_v4_flash.sh
```

The script sets the deterministic runtime knobs and starts:

- `VLLM_BATCH_INVARIANT=1`
- `VLLM_ENABLE_V1_MULTIPROCESSING=0`
- `CUDA_VISIBLE_DEVICES=0`
- `--seed 0`
- `--kv-cache-dtype fp8`
- `--moe-backend triton_unfused`
- `--attention-backend TRITON_MLA`
- fixed CUDA graph capture sizes up to `50`

Do not drop `VLLM_BATCH_INVARIANT=1`. That is the custom deterministic path
that makes the run batch-composition invariant.

## What The Patch Preserves

`patches/vllm-0.22.1-batch-invariant.patch` captures the local vLLM changes
that made this work. It changes the vLLM wheel in place to add deterministic
environment flags, greedy-logit tie controls, DeepSeek V4 decode padding for a
fixed scheduler geometry, and batch-invariant hooks in the affected attention,
linear/MoE, and routing paths.

## Run The Probe

In terminal 2:

```bash
source .venv/bin/activate
python benchmark/bench_vllm_deterministic_inference.py \
  --base-url http://127.0.0.1:8000 \
  --model deepseek-ai/deepseek-v4-flash \
  --hardware-label B300 \
  --determinism-concurrencies 1,8,32,128,300 \
  --concurrencies 1,4,8,16,32,64,128,256,300 \
  --min-requests 300
```

The benchmark fails non-zero if:

- identical deterministic prompts produce different text at any checked
  concurrency;
- the forward-order and reverse-order prompt checks disagree;
- the best streamed throughput is below `--min-output-tok-s`.

The JSON output is the machine-readable record. The Markdown table is for quick
throughput review.
