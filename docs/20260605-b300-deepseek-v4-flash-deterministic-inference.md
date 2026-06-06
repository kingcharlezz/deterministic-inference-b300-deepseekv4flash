# B300 DeepSeek V4 Flash Deterministic Inference

This note records the validation harness used to check deterministic OpenAI-compatible vLLM inference for `deepseek-ai/deepseek-v4-flash` on B300. It keeps the request shape and acceptance checks in the repository so future runs can reproduce the same determinism and throughput probe.

## Request Settings

The benchmark sends `/v1/completions` requests with:

- `temperature=0`
- `top_p=1`
- `seed=42`
- `max_tokens=256`

The deterministic pass compares exact response text for the same prompt across multiple concurrent request levels, then runs distinct prompts in forward and reverse order to catch batch-order dependent outputs.

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
