# matbench-py

Benchmarks for matrix multiplication and linear algebra on NumPy (CPU) and
CuPy (GPU).

## Setup

Uses [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

CuPy is optional and ships as per-CUDA wheels. Install the one matching your
toolkit:

```bash
uv sync --extra cuda12   # or cuda11 / cuda13
```

The scripts detect CuPy automatically and fall back to NumPy-only if no GPU
is available.

### Determining your CUDA version

CuPy wheels are built against a specific CUDA major version, so pick the
extra that matches your installed toolkit. Check it with:

```bash
nvidia-smi
```

The `CUDA Version` field in the top-right corner is what you need, e.g.
`CUDA Version: 12.4` -> use `--extra cuda12`.

If `nvidia-smi` is not available, try the CUDA toolkit itself:

```bash
nvcc --version
```

Look for the `release` line, e.g. `release 12.4` -> `--extra cuda12`.

Mapping:

| Reported version | Extra         |
|------------------|---------------|
| 11.x             | `--extra cuda11` |
| 12.x             | `--extra cuda12` |
| 13.x             | `--extra cuda13` |

Note: `nvidia-smi` reports the maximum CUDA version supported by the driver,
which is the safe choice for the CuPy wheel.

## Scripts

Matrix multiplication:

```bash
uv run matmul.py
uv run matmul.py --sizes 512,1024,2048 --dtype float32 --repeats 10 --cpu-only
```

Linear algebra ops (`solve`, `inv`, `svd`, `eig`, `qr`, `det`):

```bash
uv run linalg.py
uv run linalg.py --size 1024 --ops solve,svd,eig
```

CPU/GPU FP64 capability (FP64 units per SM, theoretical peaks, measured
throughput):

```bash
uv run device.py
uv run device.py --size 4096 --cpu-size 2048
```