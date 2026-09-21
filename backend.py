"""Backend selection and timing helpers for the benchmark scripts.

Exposes a small uniform interface so ``matmul.py`` and ``linalg.py`` can run
the same workload on NumPy (CPU) and, when available, CuPy (GPU).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np


@dataclass
class Backend:
    name: str
    xp: Any  # numpy or cupy module
    device: str

    def sync(self) -> None:
        """Wait for pending GPU work; no-op on CPU."""
        if self.name == "cupy":
            self.xp.cuda.Stream.null.synchronize()

    def asarray(self, data: Any, dtype: Any = np.float64) -> Any:
        return self.xp.asarray(data, dtype=dtype)

    def to_numpy(self, arr: Any) -> np.ndarray:
        if self.name == "cupy":
            return self.xp.asnumpy(arr)
        return np.asarray(arr)


def get_backends(include_cupy: bool = True) -> list[Backend]:
    """Return the available backends.

    NumPy is always available. CuPy is included only if installed and a GPU
    is usable; otherwise it is skipped silently (or with a note).
    """
    backends = [Backend("numpy", np, "cpu")]

    if not include_cupy:
        return backends

    try:
        import cupy as cp  # type: ignore

        # Touch the device to make sure a GPU is actually present.
        cp.empty(1).sum()
    except Exception as exc:  # ImportError / CUDARuntimeError / ...
        print(f"[backend] CuPy unavailable, skipping GPU: {exc}")
        return backends

    try:
        device = cp.cuda.runtime.getDeviceProperties(0)["name"]
        if isinstance(device, bytes):
            device = device.decode()
    except Exception:
        device = "gpu"
    backends.append(Backend("cupy", cp, device))
    return backends


def timeit(
    fn: Callable[[], Any],
    backend: Backend,
    *,
    warmup: int = 1,
    repeats: int = 5,
) -> float:
    """Run ``fn`` and return the best wall-clock time in seconds.

    ``warmup`` iterations are not measured. GPU timings are synchronised
    before stopping the clock.
    """
    for _ in range(warmup):
        fn()
    backend.sync()

    best = float("inf")
    for _ in range(repeats):
        backend.sync()
        start = time.perf_counter()
        fn()
        backend.sync()
        best = min(best, time.perf_counter() - start)
    return best


def gflops(n: int, seconds: float) -> float:
    """FLOPS estimate for an n x n matrix multiply (2 n^3)."""
    if seconds <= 0:
        return float("inf")
    return 2.0 * n**3 / seconds / 1e9