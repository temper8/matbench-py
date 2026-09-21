# AGENTS.md

Guidance for AI agents working in this repository.

## Language

- **Everything committed to the repository must be in English**: code, comments,
  docstrings, commit messages, README, documentation, issue/PR text, etc.
- **Reply to the user in Russian.** Chat responses, explanations, and summaries
  are written in Russian, even though the repository itself is English.

## Commits

- Every commit message must carry a trailing tag identifying the agent and the
  model that produced it, in this format:

  ```
  <short summary>

  [<agent>/<model>]
  ```

  Example:

  ```
  Add CuPy fallback detection to backend

  [pi/deepseek-v4.1-flash]
  ```

- Use the values of the `PI_CODING_AGENT` / `PI_MODEL` environment variables when
  available to build the tag (agent name / model name). Fall back to a clear
  literal if they are not set.
- Keep commit messages concise, imperative, and in English.

## Project layout

- Package manager: **uv**. Dependencies live in `pyproject.toml`, resolved into
  `uv.lock`.
- Keep it small: this repo is a set of standalone benchmark scripts in the
  repository root, not a library.
- Scripts:
  - `backend.py` - backend selection (NumPy/CuPy) and timing helpers.
  - `matmul.py` - matrix multiplication benchmark.
  - `linalg.py` - linear algebra operations benchmark.
- No CI, no test framework, and no packaging overhead unless explicitly asked.

## Running

```bash
uv sync                # CPU only (NumPy)
uv sync --extra cuda12 # optional CuPy wheel for CUDA 12
uv run matmul.py
uv run linalg.py --size 1024 --ops solve,svd,eig
```

CuPy is optional; scripts must degrade gracefully to NumPy when no GPU or
CuPy installation is present.

## Conventions

- Python 3.10+ syntax, `from __future__ import annotations`.
- Keep scripts runnable directly (`uv run <script>.py`) with `argparse` options.
- Prefer `numpy`-style APIs that CuPy also implements, so the same code path
  works on both backends.