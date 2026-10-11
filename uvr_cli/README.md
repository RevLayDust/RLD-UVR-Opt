# Ultimate Vocal Remover (UVR) - CLI & Performance Benchmark Suite

This package provides a **Command-Line Interface (CLI)** and a **Real Performance Benchmarking Suite** for UVR. It connects directly to UVR's actual inference engine in `separate.py` without requiring the Tkinter GUI or desktop automation.

---

## Folder Structure

```text
uvr_cli/
├── __init__.py           # Package exports and version
├── __main__.py           # Package runner for `python -m uvr_cli`
├── cli.py                # Command-line interface (`run`, `bench`, `list-models`, `smoke-test`)
├── model_resolver.py     # Headless model discovery & parameter builder
├── runner.py             # Direct execution bridge to separate.py
├── telemetry.py          # Real-time resource sampler (CPU, RAM, GPU %, VRAM, Power)
├── engine.py             # Benchmark orchestrator (warmup, rounds, statistics)
├── exporter.py           # Machine-readable JSON & CSV exporters with path sanitization
└── README.md             # This documentation
```

---

## How It Works

1. **Headless Model Resolution (`model_resolver.py`)**:
   Reads UVR's model directories (`models/MDX_Net_Models`, `models/VR_Models`, `models/Demucs_Models`), computes the exact model MD5 hash using UVR's chunked hash algorithm, and loads the model's architecture parameters (FFT, hop size, frequency bins, stems, compensation) from `model_data.json` without depending on Tkinter GUI variables.
2. **Direct Core Separation (`runner.py`)**:
   Directly instantiates `SeperateMDX`, `SeperateMDXC`, `SeperateVR`, or `SeperateDemucs` from `separate.py` and triggers `.seperate()` directly.
3. **Real-time Telemetry (`telemetry.py`)**:
   Runs a background sampling daemon measuring process CPU %, system CPU %, process RAM, GPU utilization %, and device VRAM. Synchronizes CUDA devices and resets PyTorch peak allocation statistics before/after inference to capture true peak VRAM.
4. **Machine-Readable Export (`exporter.py`)**:
   Outputs complete structured JSON documents and appends runs to a shared tabular CSV (`summary.csv`) suitable for cross-GPU comparisons and committing to GitHub. All user profile paths are automatically redacted for privacy.

---

## CLI Commands

### 1. List Available Models
Scan and display all downloaded models across MDX-Net, VR, and Demucs:
```powershell
venv\Scripts\python.exe -m uvr_cli list-models
```

### 2. Direct Separation (CLI-Only)
Separate an audio file directly from the terminal without GUI:

#### Separate using default CUDA (stable):
```powershell
venv\Scripts\python.exe -m uvr_cli run `
    --audio "path/to/song.wav" `
    --model "UVR-MDX-NET-Inst_HQ_4.onnx" `
    --precision fp16 `
    --device cuda:0 `
    --output-dir "separated_outputs"
```
#### Separate using TensorRT-RTX (faster, RTX-optimized):
```powershell
venv\Scripts\python.exe -m uvr_cli run `
    --audio "path/to/song.wav" `
    --model "UVR-MDX-NET-Inst_HQ_4.onnx" `
    --precision fp16 `
    --device trt_rtx:0 `
    --output-dir "separated_outputs"
```
#### Separate using CPU fallback:
```powershell
venv\Scripts\python.exe -m uvr_cli run `
    --audio "path/to/song.wav" `
    --model "UVR-MDX-NET-Inst_HQ_4.onnx" `
    --device cpu `
    --output-dir "separated_outputs"
```

#### Parameters (`uvr_cli run`)

| Option | Flag | Type | Default | Description |
| :--- | :---: | :---: | :---: | :--- |
| `--audio` | `-a` | `Path` | *Required* | Path to the input audio file to separate (WAV, FLAC, MP3, etc.). |
| `--model` | `-m` | `String` | *Required* | Model filename (e.g. `UVR-MDX-NET-Inst_HQ_4.onnx`), friendly name, or absolute path. |
| `--precision` | `-p` | `Choice` | `fp16` | Inference precision policy: `fp16` (default, in-memory downcast), `fp32`, or `bf16` (experimental). |
| `--device` | `-d` | `String` | `cuda:0` | Target execution device: `cuda[:N]` (default `cuda:0`), `trt_rtx[:N]`, or `cpu`. |
| `--output-dir` | `-o` | `Path` | `separated_outputs` | Output directory where separated stem files will be saved. |
| `--segment-size` | | `Integer` | *Model Default* | MDX chunk size in samples (e.g. `256`, `3000`, `4000`, `6000`). |
| `--overlap` | | `Float` | *Model Default* | Overlap ratio between adjacent chunks (e.g. `0.25`, `0.50`, `0.75`). |
| `--batch-size` | | `Integer` | `1` | Number of chunks processed concurrently per forward pass. |
| `--overwrite` | `-y` | `Flag` | `False` | Automatically overwrite existing stem files in `--output-dir` without interactive prompt. |

Supported precisions: `fp32`, `fp16` (default), and `bf16` (experimental).

#### Execution Device Selection & Fallback Behavior

| Device Argument | Provider Used | Requirements & Fallback |
| :--- | :--- | :--- |
| `cuda[:N]` | `CUDAExecutionProvider` | Requires NVIDIA GPU with CUDA (supports RTX 20+ / Turing and newer). Default is `cuda:0`. Falls back to `CPUExecutionProvider` if CUDA is not available. |
| `trt_rtx[:N]` | `NvTensorRTRTXExecutionProvider` | Requires NVIDIA RTX 30-series (Ampere) or newer and the TensorRT-RTX EP installation option. Turing architecture (RTX 20-series) is unsupported and will automatically fall back to `CUDAExecutionProvider`. |
| `cpu` | `CPUExecutionProvider` | Standard multi-threaded CPU inference. Safe fallback for non-NVIDIA or CPU-only environments. |

> [!WARNING]
> **BF16 Precision is Experimental**:  
> `bf16` is currently unstable and not recommended for general use.  
> In-memory BF16 downcasting is not yet fully stable (planned on the roadmap). If you still want to use the `bf16` option, you **must** manually downcast the model to BF16 beforehand. Even then, it may not work immediately on all hardware or Execution Providers and may fall back to another precision, such as `fp32` or `fp16`. For optimal stability and performance, **`fp16`** or **`fp32`** is strongly recommended.

If destination stem files already exist in `--output-dir`, the CLI will prompt:
`Overwrite? [Y/N]: `
To automatically overwrite without interactive confirmation, pass `--overwrite` or `-y`:
```powershell
venv\Scripts\python.exe -m uvr_cli run `
    --audio "path/to/song.wav" `
    --model "UVR-MDX-NET-Inst_HQ_4.onnx" `
    --overwrite
```

### 3. Real Performance Benchmark
Benchmark one or more models across multiple rounds with warmup:

#### Benchmark with TensorRT-RTX:
```powershell
venv\Scripts\python.exe -m uvr_cli bench `
    --audio "path/to/song.wav" `
    --model "UVR-MDX-NET-Inst_HQ_4.onnx" `
    --precision fp16 `
    --device trt_rtx:0 `
    --rounds 3 `
    --warmup 1 `
    --export-dir "benchmark_results"
```
#### Benchmark with standard CUDA:
```powershell
venv\Scripts\python.exe -m uvr_cli bench `
    --audio "path/to/song.wav" `
    --model "UVR-MDX-NET-Inst_HQ_4.onnx" `
    --precision fp16 `
    --device cuda:0 `
    --rounds 3 `
    --warmup 1 `
    --export-dir "benchmark_results"
```

#### Parameters (`uvr_cli bench`)

| Option | Flag | Type | Default | Description |
| :--- | :---: | :---: | :---: | :--- |
| `--audio` | `-a` | `Path` | *Required* | Path to input audio file for benchmarking (uncompressed WAV recommended). |
| `--model` | `-m` | `String` | *Required* | Model filename, friendly name, or absolute path to benchmark. |
| `--precision` | `-p` | `Choice` | `fp16` | Inference precision policy: `fp16`, `fp32`, or `bf16` (experimental). |
| `--device` | `-d` | `String` | `cuda:0` | Target execution device: `cuda[:N]`, `trt_rtx[:N]`, or `cpu`. |
| `--rounds` | `-r` | `Integer` | `3` | Number of measured benchmark rounds to run after warmup. |
| `--warmup` | `-w` | `Integer` | `1` | Number of untimed warmup rounds to prime model sessions, ONNX memory, and CUDA allocations. |
| `--export-dir` | `-e` | `Path` | `benchmark_results` | Directory where JSON telemetry reports and `summary.csv` will be saved. |
| `--segment-size` | | `Integer` | *Model Default* | Custom MDX chunk size in samples to evaluate segment performance. |
| `--overlap` | | `Float` | *Model Default* | Custom overlap ratio (e.g. `0.25`, `0.50`, `0.75`) to evaluate throughput vs quality trade-offs. |
| `--batch-size` | | `Integer` | `1` | Number of chunks evaluated per forward pass. |

### 4. Automated Smoke Test
Run an end-to-end verification test using an automatically generated synthetic audio clip:
```powershell
# Run smoke test on CUDA (default):
venv\Scripts\python.exe -m uvr_cli smoke-test --device cuda:0

# Run smoke test on TensorRT-RTX:
venv\Scripts\python.exe -m uvr_cli smoke-test --device trt_rtx:0

# Run smoke test on CPU:
venv\Scripts\python.exe -m uvr_cli smoke-test --device cpu
```

#### Parameters (`uvr_cli smoke-test`)

| Option | Flag | Type | Default | Description |
| :--- | :---: | :---: | :---: | :--- |
| `--model` | `-m` | `String` | *First Available* | Optional model name or absolute path to run smoke test on. |
| `--precision` | `-p` | `Choice` | `fp16` | Inference precision policy: `fp16`, `fp32`, or `bf16`. |
| `--device` | `-d` | `String` | `cuda:0` | Target execution device: `cuda[:N]`, `trt_rtx[:N]`, or `cpu`. |
| `--export-dir` | `-e` | `Path` | `benchmark_results` | Directory where smoke test reports will be saved. |

---

## Output Data Format

### 1. Tabular CSV (`benchmark_results/summary.csv`)
Columns included in the CSV summary:
- `session_id`: Unique identifier for the benchmark session
- `timestamp_utc`: UTC timestamp of the run
- `gpu_name`: Hardware GPU device name
- `cpu_name`: Hardware CPU processor model
- `model_name`: File name of the model
- `model_hash`: MD5 hash of the model
- `backend`: UVR backend architecture (MDX-Net, VR Arc, Demucs)
- `precision`: Effective precision policy (FP32, FP16, or BF16)
- `audio_file`: Name of the benchmarked audio file
- `audio_duration_sec`: Duration of the audio file in seconds
- `round`: Measurement round index or `warmup_1`
- `cache_state`: `cold` (first run) or `warm` (subsequent runs)
- `total_time_sec`: Total wall-clock execution time
- `inference_time_sec`: Pure inference time
- `speed_factor_rt`: Real-time processing speed multiplier (`audio_duration / total_time`)
- `peak_torch_vram_mb`: Peak VRAM allocated by PyTorch tensors (MiB)
- `peak_device_vram_mb`: Peak overall hardware VRAM in use (MiB)
- `avg_gpu_util_percent`: Average GPU core utilization percentage
- `peak_gpu_util_percent`: Peak GPU core utilization percentage
- `avg_cpu_util_percent`: Average CPU process utilization percentage
- `peak_process_ram_mb`: Peak system RAM consumed by the process (MiB)
- `status`: Execution outcome (`COMPLETED` or error)

### 2. Structured JSON (`benchmark_results/<session_id>.json`)
Contains the full system hardware report, audio metadata, model configuration parameters, individual per-round telemetry profiles, and aggregated statistics (mean, min, max).

---

## License

The **Ultimate Vocal Remover CLI & Performance Benchmark Suite (`uvr_cli`)** is open-source and released under the [**GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later)**](LICENSE).

```text
Copyright (C) 2026 RevLayDust
```

This guarantees copyleft protection, software freedom, and source-code availability across CLI operations and network-interactive environments while linking with the core UVR engine backends.
