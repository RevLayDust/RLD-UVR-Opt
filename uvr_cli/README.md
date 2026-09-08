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
```powershell
venv\Scripts\python.exe -m uvr_cli run `
    --audio "path/to/song.wav" `
    --model "UVR-MDX-NET-Inst_HQ_4.onnx" `
    --precision fp16 `
    --device cuda:0 `
    --output-dir "separated_outputs"
```
Supported precisions: `fp32`, `fp16`, `bf16`.

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
```powershell
venv\Scripts\python.exe -m uvr_cli bench `
    --audio "path/to/song.wav" `
    --model "UVR-MDX-NET-Inst_HQ_4.onnx" `
    --precision fp16 `
    --rounds 3 `
    --warmup 1 `
    --export-dir "benchmark_results"
```

### 4. Automated Smoke Test
Run an end-to-end verification test using an automatically generated synthetic audio clip:
```powershell
venv\Scripts\python.exe -m uvr_cli smoke-test
```

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
- `precision`: Effective precision policy (FP32, FP16, BF16, etc.)
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
