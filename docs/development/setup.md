# Environment Setup & Installation Guide

This guide describes how to configure a reproducible development environment for **RLD UVR-Opt**.

---

## 1. Prerequisites

| Requirement | Specification | Notes |
| :--- | :--- | :--- |
| **Operating System** | Windows 10 / 11 64-bit (x64 / AMD64) | Primary target platform for installer, CUDA, and TensorRT-RTX. *(Linux/macOS contributors can test CPU-only headless CLI).* |
| **Python** | Python 3.11.x (64-bit) | Python 3.11.9 recommended. 32-bit and ARM64 architectures are unsupported. |
| **Shell** | Windows PowerShell 5.1 | Built into Windows 10/11. PowerShell 7 is supported but not required. |
| **Git** | Git 2.30+ | Required for version control and submodule handling. |
| **Privileges** | Standard User | Administrator privileges are only required if Visual C++ Redistributable needs system-level installation. |

---

## 2. Cloning the Repository

Clone your fork of the repository into a directory path without non-ASCII characters:

```powershell
git clone https://github.com/<your-username>/RLD-UVR-Opt.git
cd "RLD-UVR-Opt"
```

---

## 3. Automated Setup via PowerShell Installer

The repository includes a modern 5-phase installer pipeline ([install/Install-UVR.ps1](../../install/Install-UVR.ps1)). Run the script selecting the mode matching your hardware:

### Option A: NVIDIA GPU (CUDA Execution Provider)
Recommended for all NVIDIA GPUs (RTX 20-series / Turing or newer, including GTX 16-series):
```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\install\Install-UVR.ps1 -Mode A
```
- **Installs**: PyTorch (CUDA wheel), `onnxruntime-gpu`, and dependencies as specified in [install/reqs/requirements.txt](../../install/reqs/requirements.txt) and defined in [install/lib/Config.ps1](../../install/lib/Config.ps1).

### Option B: NVIDIA GPU (TensorRT-RTX Execution Provider, Experimental)
Designed for NVIDIA RTX 30-series (Ampere), RTX 40-series (Ada Lovelace), and newer (Compute Capability $\ge$ 8.0):
```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\install\Install-UVR.ps1 -Mode B
```
- **Installs**: PyTorch (CUDA wheel), `onnxruntime-gpu`, TensorRT-RTX execution provider plugin, and dependencies as specified in [install/reqs/requirements_trt_rtx.txt](../../install/reqs/requirements_trt_rtx.txt) and defined in [install/lib/Config.ps1](../../install/lib/Config.ps1). CUDA EP remains active as an automatic fallback.

### Option C: CPU Only (Fallback)
For non-NVIDIA GPUs (AMD, Intel) or systems without a dedicated GPU:
```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\install\Install-UVR.ps1 -Mode C
```
- **Installs**: PyTorch (CPU wheel), `onnxruntime` (CPU), and dependencies as specified in [install/reqs/requirements_cpu.txt](../../install/reqs/requirements_cpu.txt) and defined in [install/lib/Config.ps1](../../install/lib/Config.ps1).

### Installer Pipeline Behavior
When running, the installer executes the following deterministic phases:
1. **Detect**: Inspects hardware via CIM controllers and `nvidia-smi`, validates Python 64-bit, and inspects `.dist-info` packages.
2. **Plan**: Formulates an action plan (`Keep`, `Install`, `Replace`, `Remove`) without modifying files.
3. **Verify**: Read-only verification gate checking driver version ($\ge 570$), disk space ($\ge 5\text{ GB}$), network connectivity, and running process locks.
4. **Install**: Creates or reuses `.venv`, records rollback snapshots, removes variant conflicts, and installs packages via Astral `uv`.
5. **Validate**: Submits an in-memory test ONNX model to the active Execution Provider, validates tensor allocation, and writes [install/install_record.json](../../install/install_record.json).

---

## 4. Manual Setup (Alternative with Astral uv)

If you prefer configuring your virtual environment manually without running the orchestrator:

```powershell
# 1. Install or update Astral uv
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# 2. Create the virtual environment
uv venv --python 3.11 venv
.\venv\Scripts\activate.ps1

# 3. Install PyTorch stack using definitions from install/lib/Config.ps1
# Query configuration from the single source of truth:
$cfg = & powershell -NoProfile -Command ". .\install\lib\Config.ps1; Get-InstallerConfig"

# For GPU (Options A and B):
uv pip install $cfg.Torch.Packages --index-url $cfg.Torch.Cu128IndexUrl

# For CPU (Option C):
uv pip install $cfg.Torch.Packages --index-url $cfg.Torch.CpuIndexUrl

# 4. Pre-install local binary wheels (provided in install/wheels/)
uv pip install (Get-ChildItem .\install\wheels\*.whl).FullName --no-deps

# 5. Install requirements based on target mode
# Option A (CUDA):
uv pip install -r .\install\reqs\requirements.txt
# Option B (TensorRT-RTX):
uv pip install -r .\install\reqs\requirements_trt_rtx.txt
# Option C (CPU):
uv pip install -r .\install\reqs\requirements_cpu.txt
```

---

## 5. External Media Binaries

The application requires `ffmpeg.exe`, `rubberband.exe`, and `sndfile.dll`.

- **FFmpeg**: Place `ffmpeg.exe` in the application root directory or ensure it is accessible in your system `PATH`.
- **Rubber Band & Libsndfile**: Ensure `rubberband.exe` and `sndfile.dll` are located in the repository root (both are provided in the repo root).

Verify external binary availability:
```powershell
.\ffmpeg.exe -version
.\rubberband.exe --version
```

---

## 6. Verifying the Setup

Run these verification steps before writing code:

1. **Run Unit Tests**:
   ```powershell
   .\venv\Scripts\python.exe -m unittest discover tests -v
   ```
   *Expected result: All tests passing (`OK`).*

2. **Run Headless Smoke Test**:
   ```powershell
   # Test inference with synthetic sine wave audio
   .\venv\Scripts\python.exe -m uvr_cli smoke-test --device cpu
   # Or --device cuda:0 / --device trt_rtx:0 if GPU hardware is present
   ```
   *Expected result: `[OK] Smoke test passed.`*

3. **Launch the Graphical Interface**:
   ```powershell
   .\run_uvr.bat
   ```
   *Expected result: UVR GUI window opens.*

---

## 7. Troubleshooting Real Failures

### 1. PowerShell Script Execution Disabled
- **Symptom**: `File Install-UVR.ps1 cannot be loaded because running scripts is disabled on this system.`
- **Fix**: Launch PowerShell and set execution policy for the process:
  ```powershell
  Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
  ```

### 2. Missing Visual C++ Runtime (VCRedist)
- **Symptom**: `ImportError: DLL load failed while importing onnxruntime_pybind11_state` or `sndfile.dll`.
- **Fix**: Install the latest official Microsoft Visual C++ 2015-2022 Redistributable (x64) from [aka.ms/vs/17/release/vc_redist.x64.exe](https://aka.ms/vs/17/release/vc_redist.x64.exe).

### 3. File Lock During Installation / Package Update
- **Symptom**: `PermissionError: [WinError 5] Access is denied: ... python.exe` or `DLL in use`.
- **Fix**: Close any open instances of Ultimate Vocal Remover GUI or background `python.exe` processes holding files in `venv/`:
  ```powershell
  Get-Process | Where-Object { $_.Path -like "*\venv\*" } | Stop-Process -Force
  ```

### 4. GPU Architecture Mismatch on Option B
- **Symptom**: Installer reports `Adjusted` from Option B to Option A.
- **Explanation**: TensorRT-RTX Execution Provider requires Compute Capability $\ge 8.0$ (Ampere / RTX 30-series or newer). Turing GPUs (RTX 20-series, GTX 16-series, CC 7.5) are automatically redirected to Option A (CUDA EP).
