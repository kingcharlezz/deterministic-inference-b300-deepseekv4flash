#!/usr/bin/env python3
"""Proof-of-concept: a *batch-invariant* GEMM written in Triton.

The whole determinism problem reduces to one property: every output element's
reduction must be computed in a FIXED order, independent of how many rows
(tokens) are in the batch. cuBLAS violates this — it picks split-K / tile
heuristics based on M, so row r of `X @ W` is not bit-identical when M changes.

This kernel computes one output tile per program with a single fixed-order
K-loop and NO split-K, so the accumulation order for output[r, :] depends only
on (r's input, W) — never on M or on the other rows. That makes it
batch-invariant: row r is bit-identical whether the batch has 1 row or 8192.

We then (1) prove bit-exactness across batch sizes and (2) benchmark vs cuBLAS
to show the determinism tax is modest. This is the technique a deterministic
DeepSeek-V4 MoE / attention kernel would use to drop the 2048 pad and the
no-mix scheduler hack (both of which only exist because the stock custom
kernels are NOT batch-invariant).
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _bi_gemm_kernel(
    X, W, Y,
    M, N, K,
    sxm, sxk, swk, swn, sym, syn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    # Single fixed-order K loop, no split-K, no atomics -> the reduction order
    # for each output element is identical regardless of M / grid shape.
    for k0 in range(0, K, BLOCK_K):
        k = k0 + offs_k
        x = tl.load(
            X + offs_m[:, None] * sxm + k[None, :] * sxk,
            mask=(offs_m[:, None] < M) & (k[None, :] < K), other=0.0,
        )
        w = tl.load(
            W + k[:, None] * swk + offs_n[None, :] * swn,
            mask=(k[:, None] < K) & (offs_n[None, :] < N), other=0.0,
        )
        acc += tl.dot(x, w, allow_tf32=False)
    tl.store(
        Y + offs_m[:, None] * sym + offs_n[None, :] * syn, acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


def bi_gemm(X, W, BLOCK_M=64, BLOCK_N=128, BLOCK_K=64):
    M, K = X.shape
    K2, N = W.shape
    assert K == K2
    Y = torch.empty((M, N), device=X.device, dtype=torch.float32)
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    _bi_gemm_kernel[grid](
        X, W, Y, M, N, K,
        X.stride(0), X.stride(1), W.stride(0), W.stride(1), Y.stride(0), Y.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return Y


def main():
    torch.manual_seed(0)
    dev = "cuda"
    K, N = 4096, 4096  # representative hidden sizes
    W = torch.randn(K, N, device=dev, dtype=torch.bfloat16)
    # One fixed "probe" row whose output we track across batch sizes.
    probe = torch.randn(1, K, device=dev, dtype=torch.bfloat16)

    print("=== batch-invariance: is probe-row output bit-identical across M? ===")
    print(f"{'M':>6} | {'cuBLAS bit-exact vs M=1':>24} | {'BI-Triton bit-exact vs M=1':>26}")
    ref_cublas = None
    ref_bi = None
    for M in (1, 8, 64, 512, 2048, 8192):
        batch = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
        batch[0] = probe[0]  # row 0 is always the same probe row
        y_cublas = (batch @ W).float()[0]
        y_bi = bi_gemm(batch, W)[0]
        if M == 1:
            ref_cublas, ref_bi = y_cublas.clone(), y_bi.clone()
            c_ok = bi_ok = "ref"
        else:
            c_ok = "YES" if torch.equal(y_cublas, ref_cublas) else "NO (drift!)"
            bi_ok = "YES" if torch.equal(y_bi, ref_bi) else "NO (drift!)"
        print(f"{M:>6} | {c_ok:>24} | {bi_ok:>26}")

    print("\n=== throughput (TFLOP/s), bf16 KxN=4096x4096 ===")
    print(f"{'M':>6} | {'cuBLAS':>10} | {'BI-Triton':>10} | {'ratio':>6}")
    for M in (512, 2048, 8192):
        batch = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
        flop = 2 * M * N * K

        def bench(fn):
            for _ in range(5):
                fn()
            torch.cuda.synchronize()
            s = torch.cuda.Event(True); e = torch.cuda.Event(True)
            s.record()
            for _ in range(20):
                fn()
            e.record(); torch.cuda.synchronize()
            return s.elapsed_time(e) / 20 / 1e3  # sec

        t_cublas = bench(lambda: batch @ W)
        t_bi = bench(lambda: bi_gemm(batch, W))
        print(
            f"{M:>6} | {flop/t_cublas/1e12:>10.1f} | {flop/t_bi/1e12:>10.1f} | "
            f"{t_bi/t_cublas:>5.2f}x"
        )


if __name__ == "__main__":
    main()
