# Testing Guide

This guide details how to execute and write automated tests for **RLD UVR-Opt**.

---

## 1. Test Suite Overview

The repository maintains automated test suites covering Python inference/CLI components and PowerShell installer pipelines:

| Suite | Runner | Location | Purpose |
| :--- | :--- | :--- | :--- |
| **Python Regression Tests** | `unittest` | [tests/](../../tests/) | Tests CLI argument parsing, model resolution, in-memory cache eviction, TRT guards, and batch file CRLF line endings. |
| **Installer Tests** | `Pester` | [install/tests/](../../install/tests/) | Tests 15 installer scenarios covering plan generation, hardware gates, driver checks, and atomic state recording. |
| **Headless Smoke Test** | `uvr_cli` | [uvr_cli/](../../uvr_cli/) | Performs end-to-end inference verification on procedurally generated synthetic audio. |

---

## 2. Running Python Unit Tests

### Run Full Test Suite
To run all 41 repository unit tests:
```powershell
.\venv\Scripts\python.exe -m unittest discover tests -v
```

### Run Targeted Test Modules
You can execute individual test modules during focused development:

```powershell
# Test CLI commands and flag parsing
.\venv\Scripts\python.exe -m unittest tests.test_uvr_cli -v

# Test in-memory resident model cache and FP16 lifecycle
.\venv\Scripts\python.exe -m unittest tests.test_in_memory_model_cache -v

# Test TensorRT-RTX provider guards and Ampere check
.\venv\Scripts\python.exe -m unittest tests.test_tensorrt_rtx -v

# Test batch file line endings (verifies CRLF formatting)
.\venv\Scripts\python.exe -m unittest tests.test_windows_batch_files -v
```

---

## 3. Running PowerShell Installer Tests (Pester)

The installer logic is verified through [Pester 3.4.0+](https://github.com/pester/Pester) (bundled with Windows 10/11):

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -Command "Invoke-Pester -Path ./install/tests"
```

### What the Pester Suite Covers:
- **Phase 2 (Plan Generation)**:
  - Migrating CPU venv to GPU (Option A) replaces PyTorch CPU with CUDA wheels.
  - Upgrading Option A to Option B keeps PyTorch and installs TRT-RTX plugin.
  - Resolving conflicting `onnxruntime` vs `onnxruntime-gpu` packages.
  - Rerunning identical options produces zero destructive changes (`Keep`).
- **Phase 3 (Verification & Gate)**:
  - No GPU or AMD GPU detected &rarr; adjusts selection to Option C.
  - NVIDIA Turing (CC 7.5) selecting Option B &rarr; adjusts to Option A.
  - NVIDIA driver version too old ($< 570$) &rarr; blocks installation.
  - Insufficient disk space ($< 5\text{ GB}$) or active file locks &rarr; blocks installation.
- **Phase 5 (Atomic Record)**:
  - Verifies atomic writing of `install_record.json` without file locking or syntax errors.

---

## 4. Running the Headless Smoke Test

The smoke test verifies end-to-end model inference without requiring existing music files. It generates a short 440 Hz synthetic stereo sine wave in memory and separates it:

```powershell
# Run smoke test on CPU
.\venv\Scripts\python.exe -m uvr_cli smoke-test --device cpu

# Run smoke test on GPU (CUDA)
.\venv\Scripts\python.exe -m uvr_cli smoke-test --device cuda:0

# Run smoke test on TensorRT-RTX
.\venv\Scripts\python.exe -m uvr_cli smoke-test --device trt_rtx:0
```

---

## 5. Testing Without Physical GPUs (Headless & Mocking)

Many contributors or automated CI environments do not possess physical NVIDIA RTX GPUs:

- **Mocking GPU Hardware**:
  In [tests/test_tensorrt_rtx.py](../../tests/test_tensorrt_rtx.py), GPU architecture checks and ONNX Runtime provider sessions are mocked via `unittest.mock.patch`:
  ```python
  with patch("torch.cuda.is_available", return_value=True), \
       patch("torch.cuda.get_device_capability", return_value=(7, 5)):
      # Simulates Turing GPU (CC 7.5) to test fallback
      providers = get_supported_execution_providers(device_id=0)
      self.assertNotIn("NvTensorRTRTXExecutionProvider", providers)
  ```
- **PRs Touching GPU Paths**:
  While mocked tests allow validating control flow without a GPU, any PR that modifies raw CUDA kernels, TensorRT engine options, or memory layout must include verified test logs or benchmark telemetry from a physical GPU machine.

---

## 6. Prohibited Test Artifacts

- **No Model Weights in Git**: Never commit binary model weights (`*.onnx`, `*.pth`, `*.ckpt`, `*.safetensors`). Use synthetic models (e.g. tiny base64 ONNX graphs) for unit testing.
- **No Copyrighted Audio**: Never commit commercial or copyrighted music files to `input/` or `tests/`. Unit tests should synthesize audio arrays in memory (`np.sin(...)`).
