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