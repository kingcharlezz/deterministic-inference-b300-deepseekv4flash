# deterministic-inference-b300-deepseekv4flash

Reproducible deterministic inference harness for `deepseek-ai/deepseek-v4-flash` on B300 using an OpenAI-compatible vLLM endpoint.

The repository includes:

- `benchmark/bench_vllm_deterministic_inference.py`: same-prompt determinism, order-invariance, and throughput checks.
- `docs/20260605-b300-deepseek-v4-flash-deterministic-inference.md`: runbook and request settings.
