"""Benchmark matrix multiplication on NumPy and CuPy.

Usage:
    uv run matmul.py
    uv run matmul.py --sizes 512,1024,2048 --dtype float32 --repeats 10
"""

from __future__ import annotations

import argparse

import numpy as np

from backend import Backend, get_backends, gflops, timeit


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--sizes",
        default="256,512,1024,2048",
        help="comma-separated square matrix sizes",
    )
    p.add_argument(
        "--dtype",
        default="float32",
        choices=["float16", "float32", "float64"],
    )
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--cpu-only", action="store_true", help="skip CuPy")
    return p.parse_args()


def bench_size(backend: Backend, n: int, dtype: str, repeats: int, warmup: int) -> float:
    rng = np.random.default_rng(0)
    a_np = rng.standard_normal((n, n))
    b_np = rng.standard_normal((n, n))

    a = backend.asarray(a_np, dtype=dtype)
    b = backend.asarray(b_np, dtype=dtype)

    seconds = timeit(lambda: a @ b, backend, warmup=warmup, repeats=repeats)
    return seconds


def main() -> None:
    args = parse_args()
    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]
    backends = get_backends(include_cupy=not args.cpu_only)

    header = f"{'size':>7} {'backend':>8} {'time (ms)':>12} {'GFLOP/s':>10}"
    print(header)
    print("-" * len(header))

    for n in sizes:
        for backend in backends:
            try:
                seconds = bench_size(
                    backend, n, args.dtype, args.repeats, args.warmup
                )
            except Exception as exc:  # OOM on GPU, unsupported dtype, ...
                print(f"{n:>7} {backend.name:>8} {'FAILED':>12}  ({exc})")
                continue
            print(
                f"{n:>7} {backend.name:>8} {seconds * 1e3:>12.3f} "
                f"{gflops(n, seconds):>10.2f}"
            )


if __name__ == "__main__":
    main()