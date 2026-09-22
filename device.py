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
and the BLAS backend, not on "FP64 blocks", so the peak is estimated from
the detected core count, SIMD width and clock (overridable via flags).

Usage:
    uv run device.py
    uv run device.py --size 4096 --repeats 3
"""

from __future__ import annotations

import argparse
import ctypes
import os
import platform
import sys

import numpy as np

from backend import get_backends, gflops, timeit

# Width of report section headers and table separators.
LINE_WIDTH = 48


def print_section(title: str) -> None:
    """Print a full-width section header rule."""
    print(f"=== {title} " + "=" * (LINE_WIDTH - len(title) - 5))


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

# FP32 FMA lanes per cycle for one x86 core, by SIMD level.
SIMD_LANES: dict[str, int] = {
    "sse": 4,
    "avx": 8,
    "avx2": 8,
    "avx512": 16,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--size", type=int, default=4096, help="square matrix size")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--cpu-only", action="store_true", help="skip the GPU part")
    p.add_argument("--cpu-cores", type=int, default=None,
                   help="physical cores for the CPU peak estimate (auto)")
    p.add_argument("--cpu-ghz", type=float, default=None,
                   help="CPU clock in GHz for the peak estimate (auto)")
    p.add_argument("--cpu-simd", choices=sorted(SIMD_LANES), default=None,
                   help="CPU SIMD level for the peak estimate (auto)")
    return p.parse_args()


def _windows_physical_cores() -> int | None:
    """Physical core count via GetLogicalProcessorInformation (Windows)."""

    class _Info(ctypes.Structure):
        _fields_ = [
            ("ProcessorMask", ctypes.c_size_t),
            ("Relationship", ctypes.c_int),
            ("_pad", ctypes.c_int),
            ("_data", ctypes.c_byte * 16),
        ]

    try:
        kernel32 = ctypes.windll.kernel32
        size = ctypes.c_ulong(0)
        kernel32.GetLogicalProcessorInformation(None, ctypes.byref(size))
        count = size.value // ctypes.sizeof(_Info)
        if count <= 0:
            return None
        info = (_Info * count)()
        if not kernel32.GetLogicalProcessorInformation(info, ctypes.byref(size)):
            return None
        return sum(1 for i in range(count) if info[i].Relationship == 0)
    except Exception:
        return None


def detect_cores() -> tuple[int, str]:
    """Best-effort physical core count; returns ``(count, source)``."""
    try:
        with open("/proc/cpuinfo", encoding="utf-8", errors="ignore") as f:
            pairs: set[tuple[str, str]] = set()
            phys = core = None
            for line in f:
                if line.startswith("physical id"):
                    phys = line.split(":", 1)[1].strip()
                elif line.startswith("core id"):
                    core = line.split(":", 1)[1].strip()
                elif not line.strip():
                    if phys is not None and core is not None:
                        pairs.add((phys, core))
                    phys = core = None
            if pairs:
                return len(pairs), "physical"
    except OSError:
        pass

    if sys.platform == "win32":
        n = _windows_physical_cores()
        if n:
            return n, "physical"

    return os.cpu_count() or 1, "logical"


def detect_simd() -> str | None:
    """Best-effort SIMD level: ``sse``/``avx``/``avx2``/``avx512``."""
    try:
        with open("/proc/cpuinfo", encoding="utf-8", errors="ignore") as f:
            flags = f.read()
        if "avx512f" in flags:
            return "avx512"
        if "avx2" in flags:
            return "avx2"
        if "avx" in flags:
            return "avx"
        if "sse2" in flags:
            return "sse"
    except OSError:
        pass

    # Windows exposes AVX/AVX2 (but not AVX-512) through this API.
    if sys.platform == "win32":
        try:
            kernel32 = ctypes.windll.kernel32
            if kernel32.IsProcessorFeaturePresent(40):  # AVX2
                return "avx2"
            if kernel32.IsProcessorFeaturePresent(39):  # AVX
                return "avx"
        except Exception:
            pass
    return None


def detect_clock_ghz() -> tuple[float | None, str]:
    """Best-effort CPU clock; returns ``(GHz, source)``."""
    if sys.platform == "win32":
        try:
            import winreg

            path = r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, path) as key:
                mhz, _ = winreg.QueryValueEx(key, "~MHz")
            if mhz:
                return mhz / 1000.0, "registry"
        except OSError:
            pass

    try:
        with open("/proc/cpuinfo", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.lower().startswith("cpu mhz"):
                    return float(line.split(":", 1)[1]) / 1000.0, "procfs"
    except OSError:
        pass
    return None, "unknown"


def resolve_cpu_params(args: argparse.Namespace) -> dict[str, object]:
    """Resolve CPU peak parameters, honouring CLI overrides."""
    cores, cores_src = detect_cores()
    if args.cpu_cores:
        cores, cores_src = args.cpu_cores, "override"

    simd = args.cpu_simd or detect_simd()

    clock, clock_src = detect_clock_ghz()
    if args.cpu_ghz:
        clock, clock_src = args.cpu_ghz, "override"

    return {
        "cores": cores,
        "cores_src": cores_src,
        "simd": simd,
        "clock": clock,
        "clock_src": clock_src,
    }


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


def report_cpu(
    cpu,
    size: int,
    repeats: int,
    warmup: int,
    params: dict[str, object],
) -> dict[str, float]:
    print_section("CPU")
    print(f"Processor:     {platform.processor() or platform.machine()}")
    print(f"Logical cores: {os.cpu_count()}")

    print(f"\nMeasuring {size}x{size} matmul (repeats={repeats})...")
    m32 = measure(cpu, size, "float32", repeats, warmup)
    m64 = measure(cpu, size, "float64", repeats, warmup)

    cores = params["cores"]
    simd = params["simd"]
    clock = params["clock"]
    lanes = SIMD_LANES.get(simd) if simd else None

    if lanes and clock:
        theor32 = cores * lanes * 2 * clock / 1000  # TFLOPS
        theor64 = theor32 / 2  # x86 FP32:FP64 lane ratio is 2:1
    else:
        theor32 = theor64 = None

    print(f"\n{'dtype':>8} {'measured':>15} {'theor':>13} {'eff.':>9}")
    print("-" * LINE_WIDTH)

    def row(dtype: str, measured: float, theor: float | None) -> None:
        if theor:
            print(f"{dtype:>8} {measured:>12.1f} GF {theor * 1000:>10.0f} GF "
                  f"{measured / (theor * 1000) * 100:>8.1f}%")
        else:
            print(f"{dtype:>8} {measured:>12.1f} GF {'n/a':>13} {'n/a':>9}")

    row("float32", m32, theor32)
    row("float64", m64, theor64)

    if m64 > 0:
        print(f"\nFP32:FP64 throughput ratio: {m32 / m64:.1f}:1")
    print("Note: theor = cores x SIMD lanes x 2 (FMA) x clock; FP64 assumes "
          "a 2:1 FP32:FP64 lane ratio. The clock is the nominal (non-boost) "
          "value, and some cores (e.g. AMD Zen) issue more than one FMA per "
          "lane, so eff. can exceed 100%.")

    simd_txt = (f"{simd} ({lanes} FP32 lanes)"
                if simd and lanes else "unknown")
    clock_txt = (f"{clock:.3f} GHz ({params['clock_src']})"
                 if clock else "unknown")
    print(f"CPU params: cores={cores} ({params['cores_src']}), "
          f"SIMD={simd_txt}, clock={clock_txt}")
    return {"float32": m32, "float64": m64}


def report_gpu(gpu, size: int, repeats: int, warmup: int) -> dict[str, float] | None:
    props = gpu.xp.cuda.runtime.getDeviceProperties(0)
    major, minor = props["major"], props["minor"]
    cc = (major, minor)
    sms = props["multiProcessorCount"]
    clock_ghz = props["clockRate"] / 1e6

    print_section("GPU")
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
    print(f"FP32 theor:         {fp32_peak:.3f} TFLOPS")
    print(f"FP64 theor:         {fp64_peak:.3f} TFLOPS "
          f"(at {clock_ghz:.3f} GHz)")

    print(f"\nMeasuring {size}x{size} matmul (repeats={repeats})...")
    m32 = measure(gpu, size, "float32", repeats, warmup)
    m64 = measure(gpu, size, "float64", repeats, warmup)

    print(f"\n{'dtype':>8} {'measured':>15} {'theor':>13} {'eff.':>9}")
    print("-" * LINE_WIDTH)
    print(f"{'float32':>8} {m32:>12.1f} GF {fp32_peak * 1000:>10.0f} GF "
          f"{m32 / (fp32_peak * 1000) * 100:>8.1f}%")
    print(f"{'float64':>8} {m64:>12.1f} GF {fp64_peak * 1000:>10.0f} GF "
          f"{m64 / (fp64_peak * 1000) * 100:>8.1f}%")
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

    print_section("Summary")
    header = f"{'backend':>8} {'float32':>19} {'float64':>19}"
    print(header)
    print("-" * len(header))

    def row(name: str, res: dict[str, float] | None) -> None:
        if res is None:
            print(f"{name:>8} {'n/a':>19} {'n/a':>19}")
        else:
            print(f"{name:>8} {res['float32']:>16.1f} GF "
                  f"{res['float64']:>16.1f} GF")

    row("cpu", cpu)
    row("gpu", gpu)

    if cpu and gpu:
        print("\nGPU:CPU Speedup:")
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
        params = resolve_cpu_params(args)
        cpu_res = report_cpu(
            cpu, args.size, args.repeats, args.warmup, params)

    if gpu is None:
        print("\nNo GPU backend available; skipping the GPU part.")
    else:
        print()
        gpu_res = report_gpu(gpu, args.size, args.repeats, args.warmup)

    print()
    print_summary(cpu_res, gpu_res)


if __name__ == "__main__":
    main()