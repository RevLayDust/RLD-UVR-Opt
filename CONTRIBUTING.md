# Contributing to RLD UVR-Opt

Thank you for your interest in contributing to **RLD UVR-Opt**! We welcome bug fixes, performance optimizations, and documentation improvements.

---

## Repository Structure

Contributions should be organized according to the existing module boundaries:

| Component | Path | Description |
| :--- | :--- | :--- |
| **CLI & Benchmarking** | `uvr_cli/` | Headless execution interface (`cli.py`), runner bridge (`runner.py`), model resolver (`model_resolver.py`), telemetry monitor (`telemetry.py`), and benchmark engine (`engine.py`). |
| **Core Inference Engine** | `separate.py` | Core model inference routines, in-memory model cache (`MODEL_CACHE`), precision policies (`SmartPrecisionPolicy`), and memory lifecycle handlers. |
| **Automated Tests** | `tests/` | Unit tests for CLI parsing, headless resolution, in-memory caching, progress bars, and batch line endings. |
| **Installer Scripts** | `install/` | Windows batch installer (`install_packages.bat`) and runtime dependency specifications (`requirements.txt`). |

---

## Development Guidelines

1. **CLI Changes**:
   - New CLI arguments, commands, or telemetry metrics belong inside `uvr_cli/`.
   - Update `uvr_cli/README.md` whenever adding or changing command-line flags.

2. **Core Inference Changes**:
   - Changes to model separation logic or memory handling belong in `separate.py`.
   - Ensure in-memory caching semantics and model eviction behaviors are preserved.

3. **Writing Tests**:
   - Any modifications to the CLI or model cache must include corresponding unit tests in `tests/test_uvr_cli.py` or `tests/test_in_memory_model_cache.py`.

4. **Git Hygiene & Clean Pull Requests**:
   - Do **NOT** commit private audio tracks, raw model weights (`*.onnx`, `*.ckpt`, `*.pth`), stem files, or unredacted system logs.
   - The `.gitignore` is already configured to exclude model checkpoints and output directories.

---

## Running Verification Tests

Before submitting changes, run the test suite to ensure everything passes:

```powershell
# Run the complete test suite
.\venv\Scripts\python.exe -m unittest discover tests -v

# Or run targeted test files
.\venv\Scripts\python.exe -m unittest tests.test_uvr_cli tests.test_in_memory_model_cache -v
```

You can also run an end-to-end synthetic audio smoke test to verify the CLI inference pipeline:

```powershell
.\venv\Scripts\python.exe -m uvr_cli smoke-test
```
