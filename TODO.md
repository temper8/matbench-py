# TODO

## Apple Silicon (M3) benchmark support

Goal: run the same matrix-multiplication / linear-algebra benchmarks on a
Mac with an M3 chip, comparing CPU and GPU.

### Background

- CuPy is CUDA-only and does not work on macOS / Apple Silicon.
- GPU acceleration on Apple Silicon goes through **Metal**, not CUDA. Possible
  frameworks: **MLX** (Apple, NumPy-like API), **PyTorch MPS**, or JAX with a
  Metal plugin (experimental).
- **FP64 on the GPU is effectively unavailable** on Apple Silicon:
  - MLX does not support `float64` at all (only float16 / bfloat16 / float32).
  - PyTorch's MPS backend also does not support `float64`.
  - Apple GPUs have no practical FP64 units.
- Therefore an FP64 GPU benchmark on M3 is not possible. FP64 stays a
  CPU-only measurement. This is the same conclusion as on a consumer
  GeForce (GTX 1650: 2 FP64 units per SM), only stricter.
- CPU NumPy on macOS uses Apple's **Accelerate** framework for BLAS/LAPACK,
  which is fast and supports FP64.
- Apple Silicon uses **unified memory**, so there is no host-to-device copy
  to measure or pay for.

### What is worth measuring

| Workload            | How                        | Notes                              |
|---------------------|----------------------------|------------------------------------|
| CPU FP32 / FP64     | NumPy (Accelerate BLAS)    | fast CPU BLAS, native FP64         |
| GPU FP32 / FP16     | MLX or PyTorch MPS         | where the M3 GPU wins              |
| Host<->device copy  | not applicable             | unified memory                     |

### Tasks

- [ ] Generalize `backend.py`: replace the hard-coded `cupy` backend with a
      pluggable interface (`asarray`, `sync`, `timeit`) so `mlx` and/or
      `torch-mps` can be added.
- [ ] Add an optional `mlx` backend with graceful fallback, mirroring the
      CuPy detection (skip silently when unavailable).
- [ ] Metal sync differs: use `mx.eval(...)` instead of CUDA stream sync.
- [ ] Restrict the GPU part to `float32` / `float16` for MLX and MPS;
      explicitly reject or fall back on `float64` with a clear message.
- [ ] Add an Apple Silicon device report (analogous to `device.py`); the
      compute-capability FP64 tables do not apply. Report measured FP32/FP16
      throughput and note that FP64 is CPU-only.
- [ ] Optionally add a PyTorch MPS backend as an alternative to MLX.

### Open questions

- Is MLX or PyTorch MPS the better target for a fair NumPy comparison?
- Which dtype set to report by default on Apple Silicon: `float32` only,
  or `float32` + `float16`?
- Should the summary table mark FP64 GPU cells as `n/a` on Apple Silicon?