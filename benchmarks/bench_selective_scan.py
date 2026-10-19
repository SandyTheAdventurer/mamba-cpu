"""Compare the native kernel against the pure-PyTorch reference scan.

The reference loops over time in Python, like the HF transformers "slow path".

    python benchmarks/bench_selective_scan.py --batch 1 --dim 1536 --seqlen 2048
"""

import argparse
import time

import torch

from mamba_cpu import selective_scan_fn, selective_scan_ref


def bench(fn, warmup=1, iters=5):
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) / iters


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--dim", type=int, default=1536)
    p.add_argument("--seqlen", type=int, default=1024)
    p.add_argument("--dstate", type=int, default=16)
    p.add_argument("--threads", type=int, default=None)
    p.add_argument("--skip-ref", action="store_true")
    p.add_argument("--backward", action="store_true", help="time forward + backward")
    args = p.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    b, d, l, n = args.batch, args.dim, args.seqlen, args.dstate
    u = torch.randn(b, d, l)
    delta = torch.rand(b, d, l) * 0.5
    A = -torch.rand(d, n) - 0.5
    B = torch.randn(b, n, l)
    C = torch.randn(b, n, l)
    D = torch.randn(d)
    z = torch.randn(b, d, l)
    bias = torch.rand(d) * 0.1
    tensors = [u, delta, A, B, C, D, z, bias]

    def run(fn):
        if not args.backward:
            return torch.no_grad()(lambda: fn(*tensors, True))
        leaves = [t.clone().requires_grad_() for t in tensors]
        return lambda: fn(*leaves, True).sum().backward()

    mode = "fwd+bwd" if args.backward else "fwd"
    print(f"{mode}, shape b={b} d={d} l={l} n={n}, threads={torch.get_num_threads()}")
    t_native = bench(run(selective_scan_fn))
    print(f"mamba_cpu  : {t_native * 1e3:9.2f} ms")
    if not args.skip_ref:
        t_ref = bench(run(selective_scan_ref))
        print(f"reference  : {t_ref * 1e3:9.2f} ms  ({t_ref / t_native:.1f}x slower)")


if __name__ == "__main__":
    main()
