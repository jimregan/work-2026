# Dev Container

This project uses Pynini/OpenFST, which is often easier to keep stable in
Linux than on macOS. The dev container installs the Python stack through
conda-forge, including `pynini`, then installs this package in editable mode
without resolving the native dependencies again through pip.

Open the repo in VS Code or Codex and choose "Reopen in Container". The
`postCreateCommand` runs:

```bash
python -m pip install --no-deps -e '.[dev]'
python -m pytest -q
```

`k2` is intentionally not pinned in the container because its install target
depends on CPU/GPU, PyTorch, Python, and CUDA choices. For decoder work,
install the matching wheel inside the container after it starts, then run:

```bash
python -c "import pynini, k2; print('native FST stack ok')"
python -m pytest -q
```

For rule-learning work only, `k2` is not required.
