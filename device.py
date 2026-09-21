"""Report CPU/GPU FP64 capability: FP64 units per SM and FLOP throughput.

For the GPU, CUDA does not expose the number of FP64 ALUs per streaming
multiprocessor (SM) through any API. Two ways to obtain it:

  1. Look it up from the compute capability (this is what the tables below
     do). This is the authoritative source.
  2. Measure it: run a compute-bound SGEMM (float32) and DGEMM (float64).
     This is a sanity check, but the achieved ratio is distorted when one
     of the kernels is not running at peak (memory-bound, low occupancy,
     clock throttling), so it can differ from the architectural ratio.

For the CPU, FP64 throughput depends on the SIMD width (SSE/AVX/AVX-512)
and the BLAS backend, not on "FP64 blocks", so it is only measured.

Usage:
    uv run device.py
    uv run device.py --size 4096 --cpu-size 2048 --repeats 3
"""

from __future__ import annotations

import argparse
import os
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
    p.add_argument("--size", type=int, default=4096, help="GPU square size")
    p.add_argument("--cpu-size", type=int, default=2048, help="CPU square size")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--cpu-only", action="store_true", help="skip the GPU part")
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


def report_cpu(cpu, size: int, repeats: int, warmup: int) -> dict[str, float]:
    print("=== CPU ===")
    print(f"Processor:     {platform.processor() or platform.machine()}")
    print(f"Logical cores: {os.cpu_count()}")

    print(f"\nMeasuring {size}x{size} matmul (repeats={repeats})...")
    m32 = measure(cpu, size, "float32", repeats, warmup)
    m64 = measure(cpu, size, "float64", repeats, warmup)

    print(f"\n{'dtype':>8} {'GFLOP/s':>12} {'TFLOPS':>9}")
    print("-" * 31)
    print(f"{'float32':>8} {m32:>12.1f} {m32 / 1000:>9.3f}")
    print(f"{'float64':>8} {m64:>12.1f} {m64 / 1000:>9.3f}")
    if m64 > 0:
        print(f"\nFP32:FP64 throughput ratio: {m32 / m64:.1f}:1")
    print("(CPU uses NumPy's BLAS backend; FP64 speed depends on SIMD width "
          "and core count.)")
    return {"float32": m32, "float64": m64}


def report_gpu(gpu, size: int, repeats: int, warmup: int) -> dict[str, float] | None:
    props = gpu.xp.cuda.runtime.getDeviceProperties(0)
    major, minor = props["major"], props["minor"]
    cc = (major, minor)
    sms = props["multiProcessorCount"]
    clock_ghz = props["clockRate"] / 1e6

    print("=== GPU ===")
    print(f"GPU:                {decode(props['name'])}")
    print(f"Compute capability: {major}.{minor}")
    print(f"SMs:                {sms}")
    print(f"Core clock:         {clock_ghz:.3f} GHz")

    fp32_lanes = FP32_LANES_PER_SM.get(cc)
    ratio = FP64_RATIO.get(cc)

    if fp32_lanes is None or ratio is None:
        print("\nUnknown compute capability; no FP64 entry in the tables.")
        print("Add it to FP32_LANES_PER_SM / FP64_RATIO in device.py.")
        return None

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

    print(f"\nMeasuring {size}x{size} matmul (repeats={repeats})...")
    m32 = measure(gpu, size, "float32", repeats, warmup)
    m64 = measure(gpu, size, "float64", repeats, warmup)

    print(f"\n{'dtype':>8} {'measured':>12} {'peak':>10} {'eff.':>8}")
    print("-" * 40)
    print(f"{'float32':>8} {m32:>9.1f} GF {fp32_peak * 1000:>7.0f} GF "
          f"{m32 / (fp32_peak * 1000) * 100:>7.1f}%")
    print(f"{'float64':>8} {m64:>9.1f} GF {fp64_peak * 1000:>7.0f} GF "
          f"{m64 / (fp64_peak * 1000) * 100:>7.1f}%")
    print("\nNote: the architectural ratio above is authoritative. The "
          "measured\nratio can look better or worse when a kernel is not "
          "running at peak.")
    return {"float32": m32, "float64": m64}


def print_summary(
    cpu: dict[str, float] | None,
    gpu: dict[str, float] | None,
) -> None:
    """Print a side-by-side CPU/GPU table and the GPU:CPU speedup."""
    if cpu is None and gpu is None:
        return

    print("=== Summary ===")
    header = f"{'backend':>8} {'float32 (GF)':>14} {'float64 (GF)':>14}"
    print(header)
    print("-" * len(header))

    def row(name: str, res: dict[str, float] | None) -> None:
        if res is None:
            print(f"{name:>8} {'n/a':>14} {'n/a':>14}")
        else:
            print(f"{name:>8} {res['float32']:>14.1f} "
                  f"{res['float64']:>14.1f}")

    row("cpu", cpu)
    row("gpu", gpu)

    if cpu and gpu:
        print("\nGPU:CPU throughput ratio (speedup):")
        for dtype in ("float32", "float64"):
            c, g = cpu[dtype], gpu[dtype]
            ratio = g / c if c > 0 else float("inf")
            verdict = "GPU faster" if ratio >= 1 else "CPU faster"
            print(f"  {dtype:>7}: {ratio:>6.2f}x  ({verdict})")


def main() -> None:
    args = parse_args()

    print(f"Host: {platform.processor() or platform.machine()} "
          f"({platform.system()} {platform.release()})")

    backends = get_backends(include_cupy=not args.cpu_only)
    cpu = next((b for b in backends if b.name == "numpy"), None)
    gpu = next((b for b in backends if b.name == "cupy"), None)

    cpu_res = gpu_res = None

    print()
    if cpu is not None:
        cpu_res = report_cpu(cpu, args.cpu_size, args.repeats, args.warmup)

    if gpu is None:
        print("\nNo GPU backend available; skipping the GPU part.")
    else:
        print()
        gpu_res = report_gpu(gpu, args.size, args.repeats, args.warmup)

    print()
    print_summary(cpu_res, gpu_res)

    if cpu_res and gpu_res and args.cpu_size != args.size:
        print(f"\nNote: CPU measured at {args.cpu_size}x{args.cpu_size} and GPU "
              f"at {args.size}x{args.size}; sizes differ, so the ratio is "
              "indicative.")


if __name__ == "__main__":
    main()