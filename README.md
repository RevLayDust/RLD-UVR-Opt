<div align="center">

# RLD UVR-Opt: Ultimate Vocal Remover GUI (Optimized)

[![Release](https://img.shields.io/github/v/release/revlaydust/rld-uvr-opt?color=blue)](https://github.com/revlaydust/rld-uvr-opt/releases/latest)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python: 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/downloads/latest/python3.11/)
[![Platform: Windows](https://img.shields.io/badge/Platform-Windows%2010%2F11%20(64--bit)-informational.svg?logo=windows&logoColor=white)](#-installation--prerequisites)
[![Acceleration: CUDA & CPU](https://img.shields.io/badge/Acceleration-CUDA%20%7C%20CPU-76B900.svg?logo=nvidia&logoColor=white)](#-installation--prerequisites)

**High-performance, memory-optimized distribution of Ultimate Vocal Remover GUI (UVR v5.6).**  
Focused on accelerated **ONNX Runtime inference**, lower hardware VRAM usage, and faster CLI startup.

<br/>
<img
  src="./assets/demo_uvr-opt.avifs"
  alt="RLD UVR-Opt Performance Demo"
  width="1200"
/>
<br/>

## Up to ~4x Faster Inference &nbsp;|&nbsp; ~53% Lower VRAM Usage

[Prerequisites & Installation](#installation--prerequisites) $\cdot$ [Usage & Quick Start](#usage--quick-start) $\cdot$ [Key Features](#key-features) $\cdot$ [Verified Benchmarks](#verified-benchmarks) $\cdot$ [Roadmap](#roadmap) $\cdot$ [Credits](#credits--acknowledgments)

</div>

---

## Installation & Prerequisites

### 1. System Requirements & Platform Support
> [!IMPORTANT]
> **Platform Support**: RLD UVR-Opt is actively tested and optimized exclusively for **Windows 10 / 11 (64-bit)**.

* **Compute Device**:
  * **NVIDIA GPU (CUDA)**: NVIDIA GeForce RTX 20 Series GPU or higher with at least 6GB VRAM is the minimum requirement for GPU conversions (**8GB+ VRAM recommended for optimal performance**).
  * **CPU**: Multi-core modern 64-bit x86 processor (fully supported via multi-threaded CPU inference).
  * *(Note: Hardware focus is strictly on **CUDA** and **CPU**).*
* **Python**: **Optional** (the automated installer automatically bootstraps a standalone Python 3.11 environment via `uv`). Installing system **Python 3.11+ (64-bit)** is only required if you plan on manual setup, development, or debugging.

### 2. Model & Architecture Scope

> [!NOTE]
> **ONNX Model Priority**:  
 RLD UVR-Opt's core acceleration pipeline is built around ONNX Runtime for `.onnx` models.  
 **Officially supported and tested**: `.onnx` models from the **MDX-Net family** on the documented CUDA and CPU paths.
 **Legacy/Experimental**: non-`.onnx` models may still load through existing UVR components, but they are currently untested and may not function as expected. 

---

### 3. One-Click Automated Installation (Windows)

RLD UVR-Opt provides an automated setup script that handles your entire environment setup from start to finish:

#### Step 1: Clone or Download the Repository
```powershell
git clone https://github.com/revlaydust/rld-uvr-opt.git
cd rld-uvr-opt
```
*(Or [Download the ZIP](https://github.com/revlaydust/rld-uvr-opt/archive/refs/heads/main.zip) and extract it to a folder).*

#### Step 2: Run the Automated Installer
Simply double-click `install\install_packages.bat` in File Explorer, or run in terminal:
```powershell
.\install\install_packages.bat
```

#### Step 3: Run the GUI
> [!IMPORTANT]
> If you installed the **GPU version**, you must follow these steps:

1. Open UVR using `run_uvr.bat` or `python UVR.py`.
2. Go to **Main Settings** (to the left of the **Start Processing** button with the wrench icon) → **Additional Settings** → Change **GPU Device** from the default setting to your GPU using the dropdown menu.
3. Go back to the main menu, then enable **GPU Conversions**.
4. Re-open UVR.
5. Done! You can now start using it!

If you installed the **CPU version**, simply launch UVR using `run_uvr.bat`, `python UVR.py`, or by double-clicking `run_uvr.bat` in File Explorer.

> If you encounter any specific issues, feel free to [open an issue](https://github.com/revlaydust/rld-uvr-opt/issues)! 

#### What the One-Click Installer automates for you:
* **Hardware & Driver Detection**: Automatically identifies your GPU architecture (NVIDIA RTX 20/30/40/50 series) and selects PyTorch CUDA 12.8 or optimized CPU fallback.
* **Python Environment Management**: Automatically bootstraps `uv` and provisions an isolated Python 3.11 virtual environment (`venv`).
* **Fast Dependency Installation**: Pulls and configures all required packages, including PyTorch and ONNX Runtime GPU/CPU.
* **Visual C++ Redistributable Auto-Install**: Checks whether Microsoft Visual C++ 2015–2022 Redistributable (x64) is installed. If missing or outdated, it automatically downloads and silently installs the official runtime to ensure native C-extensions and runtime DLLs load without errors.
* **Automated Media Binaries**: Detects existing system installations of **FFmpeg** and **rubberband-cli**. If either is missing, it automatically downloads and extracts portable binaries into the application folder—**zero manual downloading required!**
* **Smoke Test Verification**: Runs an immediate runtime sanity check on tensor operations and ONNX execution providers before launching.

<details>
  <summary><b>🛠️ Manual Installation via Astral uv (Click to Expand for Advanced Users)</b></summary>

<br />

If you prefer to configure your environment manually instead of running `install\install_packages.bat`, use **[Astral uv](https://github.com/astral-sh/uv)** for lightning-fast setup:

1. **Install Astral uv** (if not already installed):
   ```powershell
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

2. **Create Python 3.11 Virtual Environment via uv**:
   ```powershell
   uv venv venv --python 3.11
   .\venv\Scripts\activate
   ```

3. **Install PyTorch & ONNX Runtime with uv pip**:
   - **For NVIDIA GPU (CUDA 12.8)**:
     ```powershell
     uv pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128
     ```
   - **For CPU Only Fallback**:
     ```powershell
     uv pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cpu
     ```

4. **Install Base Dependencies with uv pip**:
    - **For NVIDIA GPU**:
      ```powershell
      uv pip install -r .\install\requirements.txt
      ```
    - **For CPU Only**:
      ```powershell
      uv pip install -r .\install\requirements_cpu.txt
      ```

5. **Place External Media Binaries**:
   - Place `ffmpeg.exe` in the application root folder (or ensure it is in your system `PATH`).
   - Place `rubberband.exe` and `sndfile.dll` in the application root folder (or ensure it is in your system `PATH`).

</details>

## Usage / Quick Start

Get from zero to audio separation in seconds:

### A. Launch the GUI
Simply run the included launcher batch file or start via Python:
```powershell
.\run_uvr.bat
```
*Or via terminal:*
```powershell
python UVR.py
```

### B. UVR Command-Line Interface (UVR-CLI)
RLD UVR-Opt provides a dedicated, lightweight CLI designed for automation and headless environments:

```powershell
# 1. View all available commands instantly
uvr_cli --help

# 2. Run high-performance separation with MDX-Net Inst_HQ_4
uvr_cli run --audio "path/to/song.wav" --model "UVR-MDX-NET-Inst_HQ_4.onnx" --output "output_stems"

# 3. High-throughput separation with custom overlap and segment size
uvr_cli run -a "song.wav" -m "UVR-MDX-NET-Inst_HQ_4.onnx" --overlap 0.50 --segment-size 4000 -o "output_stems"
```

---

## Key Features

* **Accelerated ONNX Inference**: Optimized tensor pipelines and buffer handling reduce unnecessary overhead during chunk processing.
* **In-Memory FP16 Conversion**: FP16 conversion is performed in memory without requiring persistent `.fp16.onnx` files.
* **Memory & Session Management**: Improved session and allocation handling helps reduce unnecessary resource usage across runs.
* **Faster CLI Startup**: Deferred loading of heavy runtime modules reduces startup work before inference is requested.
* **Benchmark Telemetry**: Built-in measurements and metadata help compare inference performance across controlled test runs.

---

## Verified Benchmarks

All metrics were gathered using the built-in isolated benchmark suite on identical test audio (**48.34s** stereo WAV, model: `UVR-MDX-NET-Inst_HQ_4.onnx`, Ultra Quality FP32). The results below reflect the documented test configuration and are not a universal performance guarantee.

> **Test Hardware:** NVIDIA GeForce RTX 5060 Ti &nbsp;|&nbsp; AMD Ryzen 5 5600 6-Core Processor &nbsp;|&nbsp; 32 GB RAM &nbsp;|&nbsp; Windows 11

### 1. GPU Inference Performance (CUDA)

| Configuration | Metric | UVR v5.6 (Legacy) | RLD UVR-Opt | Improvement |
| :--- | :--- | :---: | :---: | :---: |
| **Overlap 0.75**<br>*(Segment: 4000)* | **Inference Time**<br>**Processing Speed**<br>**Peak Device VRAM**<br>PyTorch VRAM Alloc | 13.52 s<br>3.58x Realtime<br>8,210 MB<br>6,360 MB | **3.85 s**<br>**12.54x Realtime**<br>**3,941 MB**<br>**177.5 MB** | **~3.5x Faster**<br>+250% Throughput<br>**-52.0% VRAM Usage (4.2 GB Saved)**<br>-97.2% PyTorch VRAM Allocation |
| **Overlap 0.50**<br>*(Segment: 4000)* | **Inference Time**<br>**Processing Speed**<br>**Peak Device VRAM** | 7.99 s<br>6.05x Realtime<br>8,214 MB | **2.06 s**<br>**23.44x Realtime**<br>**3,893 MB** | **~3.87x Faster**<br>+287% Throughput<br>**-52.6% VRAM Usage (4.3 GB Saved)** |

### 2. CPU Inference Performance

| Configuration | Metric | UVR v5.6 (Legacy) | RLD UVR-Opt | Improvement |
| :--- | :--- | :---: | :---: | :---: |
| **Overlap 0.50**<br>*(Segment: 4000)* | **Inference Time**<br>**Processing Speed**<br>**Peak System RAM** | 79.84 s<br>0.61x Realtime<br>8,928 MB | **32.17 s**<br>**1.50x Realtime**<br>**3,210 MB** | **~2.48x Faster**<br>+146% Throughput<br>**-64.0% RAM Usage (5.7 GB Saved)** |

> [!NOTE]
> Raw per-round session logs and metric datasets are preserved in [`benchmark_results/`](benchmark_results/) (`summary.csv` and `summary_legacy.csv`).

---

## Roadmap

- [x] Accelerated ONNX inference pipeline and buffer optimizations.
- [x] In-memory `FP16` downcasting (e.g., `FP32` $\rightarrow$ `FP16`) and cache lifecycle management.
- [ ] In-memory `BF16`/`FP8 E4M3FN` downcasting.
- [ ] **Windows ML (via ONNX Runtime)**: Evaluate as a broader Windows execution-provider path for supported hardware.
- [ ] **TensorRT Execution Provider (via ONNX Runtime)**: Evaluate as an optional NVIDIA acceleration path, with **CUDA EP** retained as the stable fallback.
- [ ] Support for Newer Model Architectures.
- [ ] UI/UX Overhaul.

---

## Credits & Acknowledgments

### RLD UVR-Opt Maintainer
* [**RevLayDust**](https://github.com/revlaydust) — Core architecture optimizations, ONNX pipeline acceleration, CLI engine, and memory lifecycle management.

### Core UVR Developers
* [**Anjok07**](https://github.com/anjok07) — UVR Creator & Core Developer.
* [**aufr33**](https://github.com/aufr33) — UVR Core Developer & AI Model Architect.

### Special Thanks & Upstream Contributors
* [**ZFTurbo**](https://github.com/ZFTurbo) — Created & trained weights for MDX23C models.
* [**DilanBoskan**](https://github.com/DilanBoskan) — Essential contributions at the start of the UVR project.
* [**Bas Curtiz**](https://www.youtube.com/user/bascurtiz) — Designed the official UVR logo, icon, banner, and splash screen.
* [**tsurumeso**](https://github.com/tsurumeso) — Developed the original VR Architecture code.
* [**Kuielab & Woosung Choi**](https://github.com/kuielab) — Developed the original MDX-Net AI code.
* [**Adefossez & Demucs**](https://github.com/facebookresearch/demucs) — Core developer of Facebook's Demucs Music Source Separation.
* [**KimberleyJSN**](https://github.com/KimberleyJensen) — Advised and aided implementation of training scripts for MDX-Net and Demucs.
* [**Hv**](https://github.com/NaJeongMo/Colab-for-MDX_B) — Helped implement chunk processing into MDX-Net AI code.
* **Audio Separation & CC Karaoke Discord Communities** — For continuous testing, feedback, and support.

---

## License & References

The **Ultimate Vocal Remover GUI** and **RLD UVR-Opt** codebase is open-source and released under the [**MIT License**](LICENSE).

```text
Copyright (c) 2022 Ultimate Vocal Remover
Copyright (c) 2026 RevLayDust
```

### Academic References
* [1] Takahashi et al., *"Multi-scale Multi-band DenseNets for Audio Source Separation"*, [arXiv:1706.09588](https://arxiv.org/pdf/1706.09588.pdf)
