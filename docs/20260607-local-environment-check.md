# 2026-06-07 Local Environment Check

This workspace was used to update the repository harness, but it is not an
8x B200 inference host.

Commands and observed results:

```text
nvidia-smi
NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver.

python3 --version
Python 3.12.3

python3 -m pip freeze | egrep 'sglang|vllm|torch|triton|flashinfer|flash-attn'
<no matching packages>

python3 -m sglang.launch_server --help
ModuleNotFoundError: No module named 'sglang'

vllm serve --help
vllm: command not found

python3 -m pip index versions sglang
latest observed: 0.5.12.post1

python3 -m pip index versions vllm
latest observed: 0.22.1
```

Fast local validation completed:

```text
python3 -m py_compile benchmark/bench_deterministic_inference.py
python3 benchmark/bench_deterministic_inference.py --help
python3 -m unittest discover -s tests
4 benchmark harness tests passed.

python3 -m py_compile benchmark/bench_deterministic_inference.py \
  scripts/tune_deepseek_v4_pro_8xb200.py \
  tests/test_benchmark_harness.py
completed without syntax errors.

bash scripts/serve_sglang_deepseek_v4_pro_8xb200.sh
sglang is not installed. Install requirements or your SGLang wheel first.

bash scripts/serve_vllm_deepseek_v4_pro_8xb200.sh
vllm is not installed. Install requirements or your vLLM wheel first.

python3 scripts/tune_deepseek_v4_pro_8xb200.py --dry-run --engines sglang,vllm
listed the SGLang-first and vLLM fallback tuning variants.

python3 scripts/tune_deepseek_v4_pro_8xb200.py \
  --engines sglang \
  --variants sglang-fa3-mem090 \
  --startup-timeout-s 2 \
  --cooldown-s 0 \
  --run-dir /tmp/deepseek-tune-negative-one
recorded server_failed_to_start and wrote summary/server logs. The server log
contains the expected local failure:
sglang is not installed. Install requirements or your SGLang wheel first.
```

No deterministic throughput proof can be produced on this local machine until
it has a working NVIDIA driver, exactly 8 visible B200 GPUs, and SGLang or vLLM
installed.
