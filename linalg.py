"""Benchmark common linear algebra operations on NumPy and CuPy.

Operations: solve (A x = b), LU/inverse, SVD, eigen-decomposition, QR.

Usage:
    uv run linalg.py
    uv run linalg.py --size 1024 --ops solve,svd,eig --repeats 5
"""

from __future__ import annotations

import argparse
from typing import Callable

import numpy as np

from backend import Backend, get_backends, timeit


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--size", type=int, default=512, help="square matrix size")
    p.add_argument(
        "--ops",
        default="solve,inv,svd,eig,qr",
        help="comma-separated operations",
    )
    p.add_argument("--dtype", default="float64", choices=["float32", "float64"])
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--cpu-only", action="store_true", help="skip CuPy")
    return p.parse_args()


def make_inputs(backend: Backend, n: int, dtype: str) -> dict[str, object]:
    rng = np.random.default_rng(0)
    a_np = rng.standard_normal((n, n))
    b_np = rng.standard_normal(n)

    a = backend.asarray(a_np, dtype=dtype)
    b = backend.asarray(b_np, dtype=dtype)
    return {"a": a, "b": b}


def make_ops(inp: dict[str, object]) -> dict[str, Callable[[], object]]:
    a = inp["a"]
    b = inp["b"]
    xp = a.__class__.__module__.split(".")[0]  # "numpy" or "cupy"

    if xp == "cupy":
        import cupy as cp

        linalg = cp.linalg
    else:
        linalg = np.linalg

    return {
        "solve": lambda: linalg.solve(a, b),
        "inv": lambda: linalg.inv(a),
        "svd": lambda: linalg.svd(a, full_matrices=False),
        "eig": lambda: linalg.eig(a),
        "qr": lambda: linalg.qr(a),
        "det": lambda: linalg.det(a),
    }


def main() -> None:
    args = parse_args()
    requested = [o.strip() for o in args.ops.split(",") if o.strip()]
    backends = get_backends(include_cupy=not args.cpu_only)

    header = f"{'backend':>8} {'op':>7} {'time (ms)':>12}"
    print(header)
    print("-" * len(header))

    for backend in backends:
        try:
            inp = make_inputs(backend, args.size, args.dtype)
            ops = make_ops(inp)
        except Exception as exc:
            print(f"{backend.name:>8} {'-':>7} {'FAILED':>12}  ({exc})")
            continue

        for name in requested:
            if name not in ops:
                print(f"{backend.name:>8} {name:>7} {'UNKNOWN':>12}")
                continue
            try:
                seconds = timeit(ops[name], backend, warmup=args.warmup,
                                 repeats=args.repeats)
            except Exception as exc:
                print(f"{backend.name:>8} {name:>7} {'FAILED':>12}  ({exc})")
                continue
            print(f"{backend.name:>8} {name:>7} {seconds * 1e3:>12.3f}")


if __name__ == "__main__":
    main()