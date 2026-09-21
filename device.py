"""Report GPU FP64 capability: FP64 units per SM and theoretical peak.

CUDA does not expose the number of FP64 ALUs per streaming multiprocessor
(SM) through any API. Two ways to obtain it:

  1. Look it up from the compute capability (this is what the tables below
     do). This is the authoritative source.
  2. Measure it: run a compute-bound SGEMM (float32) and DGEMM (float64).
     This is a sanity check, but the achieved ratio is distorted when one
     of the kernels is not running at peak (memory-bound, low occupancy,
     clock throttling), so it can differ from the architectural ratio.

This script reports both.

Usage:
    uv run device.py
    uv run device.py --size 4096 --repeats 3
"""

from __future__ import annotations

import argparse
import platform

import numpy as np

from backend import get_backends, gflops, timeit

# FP32 FMA lanes per SM, by compute capability (major, minor).
FP32_LANES_PER_SM: dict[tuple[int, int], int] = {
    (7, 0): 64,   # Volta (V100, Titan V)
    (7, 2): 64,   # Xavier
    (7, 5): 64,   # Turing (T4, RTX 20, GTX 16)
    (8, 0): 64,   # Ampere data center (A100)
    (8, 6): 128,  # Ampere consumer (RTX 30)
    (8, 7): 128,  # Jetson Orin
    (8, 9): 128,  # Ada (RTX 40, L4)
    (9, 0): 128,  # Hopper (H100)
    (10, 0): 128, # Blackwell data center (B100/B200)
    (12, 0): 128, # Blackwell consumer (RTX 50)
}

# FP64:FP32 throughput ratio, by compute capability.
FP64_RATIO: dict[tuple[int, int], int] = {
    (7, 0): 2,
    (7, 2): 32,
    (7, 5): 32,
    (8, 0): 2,
    (8, 6): 64,
    (8, 7): 64,
    (8, 9): 64,
    (9, 0): 2,
    (10, 0): 2,
    (12, 0): 64,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--size", type=int, default=4096, help="square matrix size")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--warmup", type=int, default=1)
    return p.parse_args()


def decode(value: object) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def measure(backend, n: int, dtype: str, repeats: int, warmup: int) -> float:
    rng = np.random.default_rng(0)
    a_np = rng.standard_normal((n, n))
    b_np = rng.standard_normal((n, n))
    a = backend.asarray(a_np, dtype=dtype)
    b = backend.asarray(b_np, dtype=dtype)
    seconds = timeit(lambda: a @ b, backend, warmup=warmup, repeats=repeats)
    return gflops(n, seconds)


def main() -> None:
    args = parse_args()

    print(f"Host: {platform.processor() or platform.machine()} "
          f"({platform.system()} {platform.release()})")

    backends = get_backends(include_cupy=True)
    gpu = next((b for b in backends if b.name == "cupy"), None)
    if gpu is None:
        print("\nNo GPU backend available; cannot determine FP64 units.")
        print("(FP64 throughput on CPU depends on AVX-512/FMA support, "
              "not on SM FP64 blocks.)")
        return

    props = gpu.xp.cuda.runtime.getDeviceProperties(0)
    major, minor = props["major"], props["minor"]
    cc = (major, minor)
    sms = props["multiProcessorCount"]
    clock_ghz = props["clockRate"] / 1e6

    print(f"GPU:                {decode(props['name'])}")
    print(f"Compute capability: {major}.{minor}")
    print(f"SMs:                {sms}")
    print(f"Core clock:         {clock_ghz:.3f} GHz")

    fp32_lanes = FP32_LANES_PER_SM.get(cc)
    ratio = FP64_RATIO.get(cc)

    if fp32_lanes is None or ratio is None:
        print("\nUnknown compute capability; no FP64 entry in the tables.")
        print("Add it to FP32_LANES_PER_SM / FP64_RATIO in device.py.")
        return

    fp64_lanes = fp32_lanes // ratio
    fp32_peak = sms * fp32_lanes * 2 * clock_ghz / 1000  # TFLOPS
    fp64_peak = fp32_peak / ratio

    print(f"FP32 lanes/SM:      {fp32_lanes}")
    print(f"FP64:FP32 ratio:    1:{ratio}")
    print(f"FP64 lanes/SM:      {fp64_lanes}")
    print(f"Total FP64 lanes:   {sms * fp64_lanes}")
    print(f"FP32 peak:          {fp32_peak:.3f} TFLOPS")
    print(f"FP64 peak:          {fp64_peak:.3f} TFLOPS "
          f"(at {clock_ghz:.3f} GHz)")

    print(f"\nMeasuring {args.size}x{args.size} matmul "
          f"(repeats={args.repeats})...")
    m32 = measure(gpu, args.size, "float32", args.repeats, args.warmup)
    m64 = measure(gpu, args.size, "float64", args.repeats, args.warmup)

    print(f"\n{'dtype':>8} {'measured':>12} {'peak':>10} {'eff.':>8}")
    print("-" * 40)
    print(f"{'float32':>8} {m32:>9.1f} GF {fp32_peak * 1000:>7.0f} GF "
          f"{m32 / (fp32_peak * 1000) * 100:>7.1f}%")
    print(f"{'float64':>8} {m64:>9.1f} GF {fp64_peak * 1000:>7.0f} GF "
          f"{m64 / (fp64_peak * 1000) * 100:>7.1f}%")
    print("\nNote: the architectural ratio above is authoritative. The "
          "measured\nratio can look better or worse when a kernel is not "
          "running at peak.")


if __name__ == "__main__":
    main()