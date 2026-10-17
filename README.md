<div align="center">

# RLD UVR-Opt: Ultimate Vocal Remover GUI (Optimized)

[![Release](https://img.shields.io/github/v/release/revlaydust/rld-uvr-opt?color=blue)](https://github.com/revlaydust/rld-uvr-opt/releases/latest)
[![Mixed License: MIT + AGPL-3.0-or-later](https://img.shields.io/badge/Mixed_License-MIT_%2B_AGPL--3.0--or--later-green.svg)](#license--references)
[![Python: 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/downloads/latest/python3.11/)
[![Platform: Windows](https://img.shields.io/badge/Platform-Windows%2010%2F11%20(64--bit)-informational.svg?logo=windows&logoColor=white)](#-installation--prerequisites)
[![Acceleration: CUDA, TensorRT-RTX & CPU](https://img.shields.io/badge/Acceleration-CUDA%20%7C%20TensorRT--RTX%20%7C%20CPU-76B900.svg?logo=nvidia&logoColor=white)](#-installation--prerequisites)

**High-performance, memory-optimized distribution of Ultimate Vocal Remover GUI (UVR v5.6).**  
Focused on accelerated **ONNX Runtime inference**, lower hardware VRAM usage, and faster CLI startup.

<br/>
<img
  src="./assets/demo_uvr-opt.avifs"
  alt="RLD UVR-Opt Performance Demo"
  width="1200"
/>
<br/>

## Up to ~5.3x Faster Inference &nbsp;|&nbsp; ~76% Lower VRAM Usage

<sub>*~3.5x faster & ~52% lower VRAM on stable CUDA EP. Up to ~5.3x faster & ~76% lower VRAM with experimental TensorRT-RTX EP (see [Verified Benchmarks](#verified-benchmarks)).*</sub>

[Prerequisites & Installation](#installation--prerequisites) $\cdot$ [Usage & Quick Start](#usage--quick-start) $\cdot$ [Key Features](#key-features) $\cdot$ [Verified Benchmarks](#verified-benchmarks) $\cdot$ [Roadmap](#roadmap) $\cdot$ [Credits](#credits--acknowledgments)

</div>

---

## Installation & Prerequisites

### 1. System Requirements & Platform Support
> [!IMPORTANT]
> **Platform Support**: RLD UVR-Opt is actively tested and optimized exclusively for **Windows 10 / 11 (64-bit)**.

* **Compute Device**:
  * **NVIDIA GPU (CUDA EP - Stable)**: NVIDIA GeForce RTX 20 Series (Turing) or higher with at least 6GB VRAM (**8GB+ VRAM recommended for optimal performance**).
  * **NVIDIA GPU (TensorRT-RTX EP - Experimental)**: NVIDIA GeForce RTX 30 Series (Ampere) or higher (RTX 30, 40, 50 series, RTX Ada/Blackwell).
    *(Note: Turing / RTX 20-series GPUs are not supported by the TensorRT-RTX execution provider; RTX 20 users should use the stable CUDA EP).*
  * **CPU**: Multi-core modern 64-bit x86 processor (fully supported via multi-threaded CPU inference).
  * *(Note: Hardware focus is strictly on **GPU** and **CPU**).*
* **Python**: **Optional** (the automated installer automatically bootstraps a standalone Python 3.11 environment via `uv`). Installing system **Python 3.11+ (64-bit)** is only required if you plan on manual setup, development, or debugging.

### 2. Model & Architecture Scope

> [!NOTE]
> **ONNX Model Priority**:  
 RLD UVR-Opt's core acceleration pipeline is built around ONNX Runtime for `.onnx` models.  
 **Officially supported and tested**: `.onnx` models from the **MDX-Net family** on the documented CUDA and CPU paths.
 **Legacy/Experimental**: non-`.onnx` models may still load through existing UVR components, but they are currently untested and may not function as expected. 

---

### 3. One-Click Automated Installation (Windows)

RLD UVR-Opt provides an automated installer that handles your entire environment setup from start to finish.

> [!TIP]
> 🔗 **[Download the installer from the Latest Release](https://github.com/revlaydust/rld-uvr-opt/releases/latest)**  
> [![Latest Release](https://img.shields.io/github/v/release/revlaydust/rld-uvr-opt?label=latest%20release&color=76B900)](https://github.com/revlaydust/rld-uvr-opt/releases/latest)

#### Step 1: Download the Installer
1. Open the **[Latest Release](https://github.com/revlaydust/rld-uvr-opt/releases/latest)** page.
2. Scroll down to the **Assets** section (click it to expand if it is collapsed) and click the file whose name ends with `Installation.bat`, for example `v1.2.3_Installation.bat`. The numbers are just the version and change with every release. Do **not** click *Source code (zip)* or *Source code (tar.gz)*, because those are not the installer.
3. Save it in the folder where you want RLD UVR-Opt to be set up. The installer will create a new folder whose name starts with `RLD-UVR-Opt` right next to it and download the project there.

> [!NOTE]
> Your browser or Windows SmartScreen may warn you about a downloaded `.bat` file. The script is plain text, so you can open it in any text editor to review it before running.

#### Step 2: Run the Installer
Double-click the file you just downloaded (the one ending with `Installation.bat`) in File Explorer, then choose an installation mode from the menu:

* **`[1]` Advanced Installation (Recommended)** — best if you need more flexibility. It clones the repository with `git`, so **Git is required**. If Git is not installed, the installer offers to download and install it for you, either automatically or manually.
* **`[2]` Standard Installation** — best if you want something simple that just works. It downloads and extracts the release ZIP, with no Git required.

The installer then launches the project's setup script automatically and takes care of the rest.

#### Step 3: Run the GUI
> [!IMPORTANT]
> If you installed the **GPU version**, you must follow these steps:

1. Open UVR using `run_uvr.bat` or `python UVR.py`.
2. Go to **Main Settings** (to the left of the **Start Processing** button with the wrench icon) → **Additional Settings** → Change **GPU Device** from the default setting to your GPU using the dropdown menu.
3. Still in **Additional Settings**, choose an **Execution Provider** (directly below **GPU Device**):
   * The **Execution Provider** dropdown stays locked (greyed out) until a **GPU Device** has been selected in step 2.
   * Once unlocked, UVR lists only the providers that are actually available for your GPU:
     * `CUDAExecutionProvider` — default and stable baseline for all supported NVIDIA GPUs (RTX 20 series / Turing and newer).
     * `NvTensorRTRTXExecutionProvider` — experimental and faster. Requires **NVIDIA RTX 30 series (Ampere) or newer**. It is only listed if you chose the **TensorRT-RTX EP** option in the installer and your GPU meets the hardware requirement (Turing / RTX 20 series users should select `CUDAExecutionProvider`).
   * If your installation only supports `CPUExecutionProvider`, both **GPU Device** and **Execution Provider** are hidden and disabled.

   > [!IMPORTANT]
   > **You MUST restart UVR every time you change the Execution Provider and GPU Conversions.** The provider is applied when UVR starts, so a change made while UVR is running does not take effect until you restart. Use the **Restart Application** button in the settings menu, or close and re-open UVR.

4. Go back to the main menu, then enable **GPU Conversions**.
5. Re-open UVR.
6. Done! You can now start using it!

If you installed the **CPU version**, simply launch UVR using `run_uvr.bat`, `python UVR.py`, or by double-clicking `run_uvr.bat` in File Explorer.

> If you encounter any specific issues, feel free to [open an issue](https://github.com/revlaydust/rld-uvr-opt/issues)! 

#### What the One-Click Installer automates for you:
* **Guided Setup**: Pick **Advanced** (Git) or **Standard** (ZIP) mode from a simple menu. In Advanced mode, a missing Git is downloaded and installed for you on request.
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
    - **For NVIDIA GPU (CUDA)**:
      ```powershell
      uv pip install -r .\install\reqs\requirements.txt
      ```
    - **For NVIDIA GPU (TensorRT-RTX)**
      ```powershell
      uv pip install -r .\install\reqs\requirements_trt_rtx.txt
      ```
    - **For CPU Only**:
      ```powershell
      uv pip install -r .\install\reqs\requirements_cpu.txt
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
RLD UVR-Opt provides a dedicated, lightweight CLI designed for automation and headless environments. Run it with the project's virtual environment:

```powershell
# 1. View all available commands instantly
venv\Scripts\python.exe -m uvr_cli --help

# 2. List installed models
venv\Scripts\python.exe -m uvr_cli list-models

# 3. Run high-performance separation with MDX-Net Inst_HQ_4 (default device: cuda:0)
venv\Scripts\python.exe -m uvr_cli run --audio "path/to/song.wav" --model "UVR-MDX-NET-Inst_HQ_4.onnx" --output-dir "output_stems"

# 4. High-throughput separation with custom overlap and segment size
venv\Scripts\python.exe -m uvr_cli run -a "song.wav" -m "UVR-MDX-NET-Inst_HQ_4.onnx" --overlap 0.50 --segment-size 4000 -o "output_stems"

# 5. Choose the execution device: CUDA, TensorRT-RTX, or CPU
venv\Scripts\python.exe -m uvr_cli run -a "song.wav" -m "UVR-MDX-NET-Inst_HQ_4.onnx" --device cuda:0
venv\Scripts\python.exe -m uvr_cli run -a "song.wav" -m "UVR-MDX-NET-Inst_HQ_4.onnx" --device trt_rtx:0
venv\Scripts\python.exe -m uvr_cli run -a "song.wav" -m "UVR-MDX-NET-Inst_HQ_4.onnx" --device cpu

# 6. Benchmark and smoke test
venv\Scripts\python.exe -m uvr_cli bench -a "song.wav" -m "UVR-MDX-NET-Inst_HQ_4.onnx" --device trt_rtx:0 --rounds 3 --warmup 1
venv\Scripts\python.exe -m uvr_cli smoke-test --device cuda:0
```

| Command | Description |
| :--- | :--- |
| `list-models` | List all installed MDX-Net, VR, and Demucs models. |
| `run` | Separate an audio file without the GUI. |
| `bench` | Run a multi-round benchmark with hardware telemetry and JSON/CSV export. |
| `smoke-test` | Run an end-to-end check using generated synthetic audio. |

**Device selection (`--device` / `-d`)** is available on `run`, `bench`, and `smoke-test`:

| Value | Execution Provider | Notes |
| :--- | :--- | :--- |
| `cuda[:N]` | `CUDAExecutionProvider` | Default (`cuda:0`). Stable choice for all NVIDIA GPUs (RTX 20+ / Turing or newer). |
| `trt_rtx[:N]` | `NvTensorRTRTXExecutionProvider` | Experimental. Requires the TensorRT-RTX EP install option and an **RTX 30-series (Ampere) or newer** GPU. Automatically falls back to CUDA EP if unavailable or on Turing (RTX 20) GPUs. |
| `cpu` | `CPUExecutionProvider` | Runs without a GPU. |

An invalid value is rejected with an error that lists the valid choices. If CUDA is unavailable, the CLI falls back to CPU and prints a warning. See [`uvr_cli/README.md`](uvr_cli/README.md) for the full option reference.

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
| **Overlap 0.75**<br>*(Segment: 4000)* | **Inference Time**<br>**Processing Speed**<br>**Peak Device VRAM**<br>PyTorch VRAM Alloc | 13.54 s<br>3.57x Realtime<br>8,043 MB<br>7,049 MB | **3.60 s**<br>**13.44x Realtime**<br>**3,816 MB**<br>**177.5 MB** | **~3.76x Faster**<br>+276% Throughput<br>**-52.6% VRAM Usage (4.1 GB Saved)**<br>-97.5% PyTorch VRAM Allocation |

### 2. CPU Inference Performance

| Configuration | Metric | UVR v5.6 (Legacy) | RLD UVR-Opt | Improvement |
| :--- | :--- | :---: | :---: | :---: |
| **Overlap 0.50**<br>*(Segment: 4000)* | **Inference Time**<br>**Processing Speed**<br>**Peak System RAM** | 77.56 s<br>0.62x Realtime<br>8,890 MB | **27.60 s**<br>**1.75x Realtime**<br>**3,238 MB** | **~2.81x Faster**<br>+180% Throughput<br>**-63.6% RAM Usage (5.5 GB Saved)** |

### 3. GPU Inference Performance (TensorRT-RTX EP, Experimental)

Same audio, model, and settings as above (**Overlap 0.75**, Segment 4000, Ultra Quality FP32), comparing UVR v5.6 (Legacy) against RLD UVR-Opt running on the **TensorRT-RTX Execution Provider**.

| Configuration | Metric | UVR v5.6 (Legacy) | RLD UVR-Opt (TensorRT-RTX) | Improvement |
| :--- | :--- | :---: | :---: | :---: |
| **Overlap 0.75**<br>*(Segment: 4000)* | **Inference Time**<br>**Processing Speed**<br>**Peak Device VRAM**<br>PyTorch VRAM Alloc | 13.54 s<br>3.57x Realtime<br>8,043 MB<br>7,049 MB | **2.56 s**<br>**18.91x Realtime**<br>**1,956 MB**<br>**177.5 MB** | **~5.3x Faster**<br>+429% Throughput<br>**-75.7% VRAM Usage (5.9 GB Saved)**<br>-97.5% PyTorch VRAM Allocation |

To isolate the effect of the Execution Provider, the table below compares CUDA EP and TensorRT-RTX EP **on the same software stack** (ONNX Runtime 1.24.4, same machine, same commit):

| Configuration | Metric | RLD UVR-Opt (CUDA EP) | RLD UVR-Opt (TensorRT-RTX EP) | Improvement |
| :--- | :--- | :---: | :---: | :---: |
| **Overlap 0.75**<br>*(Segment: 4000)* | **Inference Time**<br>**Processing Speed**<br>**Peak Device VRAM** | 3.60 s<br>13.44x Realtime<br>3,816 MB | **2.56 s**<br>**18.91x Realtime**<br>**1,956 MB** | **~1.41x Faster**<br>+41% Throughput<br>**-48.7% VRAM Usage** |

> [!NOTE]
> **How to read these numbers**:
> * Both CUDA and TensorRT-RTX results reflect the updated software stack (ONNX Runtime 1.24.4). Table 4 provides a direct same-stack comparison isolating the Execution Provider itself.
> * Each result is the mean of **9 measured rounds** after 1 warmup round (CPU: 4 measured rounds after 1 warmup round). The first (warmup) TensorRT-RTX run was slower at **5.73 s (8.44x Realtime)**, so expect a slower first run before the faster steady-state speed.
> * Only **Overlap 0.75** has been measured for TensorRT-RTX so far.
> * TensorRT-RTX is **experimental**. Results are from a single test machine and are not a universal performance guarantee.

> [!NOTE]
> Raw per-round session logs and metric datasets are preserved in [`benchmark_results/`](benchmark_results/) (`summary.csv` and `summary_legacy.csv`). The TensorRT-RTX run is session `session_20261012_112543_5611a8`, the same-stack CUDA run is session `session_20261012_112432_819a12`, the CPU run is session `session_20261012_112714_85d42d`, and the legacy baseline is session `legacy_20261012_184105_0a7fba`.

---

## Roadmap

- [x] Accelerated ONNX inference pipeline and buffer optimizations.
- [x] In-memory `FP16` downcasting (e.g., `FP32` $\rightarrow$ `FP16`) and cache lifecycle management.
- [x] **TensorRT-RTX Execution Provider (via ONNX Runtime)**: Evaluate optional acceleration for NVIDIA RTX GPUs (Ampere+), with CUDA EP retained as the stable fallback.
- [ ] In-memory `BF16`/`FP8 E4M3FN` downcasting.
- [ ] **Windows ML Execution Provider (via ONNX Runtime)**: Evaluate acceleration path for non-NVIDIA hardware (AMD, Intel) on Windows 11 (24H2+).
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

The **Ultimate Vocal Remover GUI** and **RLD UVR-Opt** core distribution is open-source and released under the [**MIT License**](LICENSE).

```text
Copyright (c) 2022 Ultimate Vocal Remover
Copyright (c) 2026 RevLayDust
```

The CLI and benchmarking suite ([`uvr_cli`](uvr_cli/)) is released under the [**GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later)**](uvr_cli/LICENSE).

```text
Copyright (C) 2026 RevLayDust
```


### Academic References
* [1] Takahashi et al., *"Multi-scale Multi-band DenseNets for Audio Source Separation"*, [arXiv:1706.09588](https://arxiv.org/pdf/1706.09588.pdf)
