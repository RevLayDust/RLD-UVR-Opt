# Benchmarking & Performance Profiling

This document specifies the official methodology, metric definitions, and reporting requirements for performance benchmarks in **RLD UVR-Opt**.

---

## 1. When Benchmarking is Mandatory

Benchmarking is strictly required for any Pull Request that:
1. Modifies inference routines in `separate.py` (chunking, windowing, STFT/iSTFT).
2. Alters Execution Provider registration or session options in `engine/tensorrt_rtx/`.
3. Changes precision downcasting policies (`SmartPrecisionPolicy`, in-memory FP16).
4. Modifies buffer allocation or headless runner bridges in `uvr_cli/`.

---

## 2. Metric Definitions & Formulas

To ensure reproducible comparisons across contributors and hardware:

| Metric | Identifier | Definition & Formula | Unit |
| :--- | :--- | :--- | :--- |
| **Inference Time** | `total_time_sec` | Total wall-clock time from tensor chunk submission until final stems are written. | Seconds ($s$) |
| **Real-Time Factor** | `speed_factor_rt` | Processing speed relative to audio playback speed: $\text{Speed Factor} = \frac{\text{Audio Duration (s)}}{\text{Inference Time (s)}}$ | Multiplier ($x\text{ RT}$) |
| **Throughput** | `throughput` | Audio duration processed per unit of time: $\text{Throughput} = \frac{\text{Audio Duration}}{\text{Total Time}}$ | Audio sec/sec |
| **Chunk Latency (p50 / p95)** | `chunk_latency_ms` | Per-chunk inference latency evaluated at median (50th percentile) and tail (95th percentile). | Milliseconds ($\text{ms}$) |
| **Active Tensor VRAM** | `peak_torch_vram_mb` | Peak memory explicitly allocated to active PyTorch tensors: `torch.cuda.max_memory_allocated()`. | Megabytes ($\text{MB}$) |
| **Peak Device VRAM** | `peak_device_vram_mb` | Total physical GPU VRAM consumed across CUDA contexts and ORT memory arenas, measured via NVML `nvmlDeviceGetMemoryInfo()`. | Megabytes ($\text{MB}$) |
| **Peak Process RAM** | `peak_process_ram_mb` | Maximum resident set size (RSS) of the application process: `psutil.Process().memory_info().rss`. | Megabytes ($\text{MB}$) |

### Statistical Reporting Standards
When reporting steady-state metrics, contributors must provide:
- **Sample Size ($n$)**: Minimum 9 measured repetitions `[PROPOSED]`.
- **Central Tendency**: Median or Mean.
- **Dispersion**: Standard Deviation ($s$) or Interquartile Range ($\text{IQR} = Q_3 - Q_1$).
- **Single-run submissions are strictly rejected.**

---

## 3. Engine Build vs. Steady-State Measurement

Accelerated Execution Providers (especially TensorRT) introduce significant latency on the first inference pass due to kernel timing calibration, graph optimization, and engine compilation:

1. **Cold Run (Overhead)**:
   - The initial pass where ONNX Runtime compiles the execution graph and writes plan caches to `models/MDX_Net_Models/trt_cache/`.
   - **Must be reported separately** as the cold/warmup run latency and must **never** be averaged into steady-state numbers.
2. **Warm Run (Steady-State)**:
   - Measured rounds after the engine has been compiled and cached in resident memory.
   - At least **2 warmup rounds** `[PROPOSED]` with dummy tensors or test audio must be completed before recording measurement rounds to eliminate cold-start spikes.

---

## 4. Controlled Variables & Benchmark Profile

All standardized benchmarks must be conducted using the following baseline profile:

| Parameter | Standard Value | Description |
| :--- | :--- | :--- |
| **Test Model** | `UVR-MDX-NET-Inst_HQ_4.onnx` | MDX-Net architecture FP32 model. |
| **Segment Size** | `4000` | Standard sample window length. |
| **Overlap** | `0.75` | Standard high-quality overlap factor. |
| **Batch Size** | `1` | Single stream inference. |
| **Precision** | `Ultra Quality (FP32)` | Baseline precision mode. Explicitly pass `--precision fp32` in CLI. |
| **Test Audio** | Standard stereo WAV | Minimum 30–60 seconds duration (e.g. 48.34s test sample). Free-licensed or synthetic only. |

### Test Audio Licensing Rules
- Audio samples must come from freely licensed sources (CC0, Creative Commons Attribution, or public domain) or synthetic waveforms (`np.sin`).
- **Never commit copyrighted audio files, commercial music tracks, or raw stems to the repository.**

Contributors must also attach their [install/install_record.json](../../install/install_record.json) or include OS version, GPU model, and NVIDIA display driver version.

---

## 5. Audio Integrity & Quality Tolerances

Performance optimization must **not** degrade audio separation quality. When submitting optimization PRs:

1. **Numerical Residual Check**:
   - Compare output stem arrays against the baseline FP32 output:
     $$\Delta_{\max} = \max |A_{\text{opt}} - A_{\text{base}}|$$
   - In pure FP32 code optimization, $\Delta_{\max} \le 10^{-5}$.
2. **Signal Degradation Tolerance (FP16 / Quantization)**:
   - For lower-precision paths, compute Signal-to-Distortion Ratio (SDR) or Signal-to-Noise Ratio (SNR):
     $$\text{SNR} = 10 \log_{10} \left( \frac{\sum A_{\text{base}}^2}{\sum (A_{\text{base}} - A_{\text{opt}})^2} \right)$$
   - Minimum acceptable threshold: **$\text{SNR} \ge 40\text{ dB}$** `[PROPOSED]`.

---

## 6. Standard Execution Procedure

Follow this step-by-step procedure when running benchmark rounds:

### Step 1: Prepare Environment
- Close all background applications (web browsers, media players, IDEs).
- Verify Windows Power Plan is set to **High Performance**.
- Check that the GPU is at baseline temperature ($< 50^\circ\text{C}$) to avoid thermal throttling.

### Step 2: Warm-up Phase
Execute at least 2 warmup passes on each target provider to compile caches:
```powershell
.\venv\Scripts\python.exe -m uvr_cli bench -a "sample.wav" -m "UVR-MDX-NET-Inst_HQ_4.onnx" --precision fp32 --device cuda:0 --warmup 2 --rounds 1
.\venv\Scripts\python.exe -m uvr_cli bench -a "sample.wav" -m "UVR-MDX-NET-Inst_HQ_4.onnx" --precision fp32 --device trt_rtx:0 --warmup 2 --rounds 1
```

### Step 3: Interleaved Measurement Rounds (Mitigating Thermal Drift)
To avoid thermal bias where the second engine tested runs hotter than the first, execute alternating rounds across configurations:
```powershell
# Interleaved benchmark execution pattern:
# Alternate between Baseline and Target across rounds:
for ($r = 1; $r -le 9; $r++) {
    Write-Host "--- Executing Measurement Round $r ---"
    # 1. Baseline Run (CUDA EP)
    .\venv\Scripts\python.exe -m uvr_cli bench -a "sample.wav" -m "UVR-MDX-NET-Inst_HQ_4.onnx" --precision fp32 --device cuda:0 --rounds 1 --warmup 0
    # 2. Optimized Run (TensorRT-RTX EP)
    .\venv\Scripts\python.exe -m uvr_cli bench -a "sample.wav" -m "UVR-MDX-NET-Inst_HQ_4.onnx" --precision fp32 --device trt_rtx:0 --rounds 1 --warmup 0
}
```

### Step 4: Outlier Rejection Policy
- If an individual measurement run suffers an operating system spike (e.g. Windows background indexing, GPU interrupt), it may be excluded if it deviates by more than $1.5 \times \text{IQR}$ from the median.
- **All excluded outlier runs must be explicitly listed and justified in the PR submission table.**

Telemetry and per-round metrics are automatically saved to `benchmark_results/summary.csv` and timestamped session JSON files.

---

## 7. Submission Templates

### Markdown Table Template for PRs

Copy and fill this table in your Pull Request description:

```markdown
### Benchmark Results

**Test Hardware**: [e.g., NVIDIA GeForce RTX 4070 (Driver 572.16) | AMD Ryzen 7 7800X3D | Windows 11]  
**Audio Sample**: [e.g., sample.wav (48.34s stereo, 44.1kHz, CC0)]  
**Model**: `UVR-MDX-NET-Inst_HQ_4.onnx` (Overlap 0.75, Segment 4000, FP32)  
**Sample Count**: n = 9 measured rounds (after 2 warmups, interleaved order)

| Execution Provider | Warmup Time | Steady-State Time (Median / Mean ± Std) | Chunk Latency (p50 / p95) | Real-Time Factor | Peak Device VRAM | Peak Process RAM |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **CUDA EP (Baseline)** | 0.00 s | 0.00 s / 0.00 s ± 0.00 s | 0.0 ms / 0.0 ms | 0.00x RT | 0,000 MB | 0,000 MB |
| **Your Optimization** | 0.00 s | 0.00 s / 0.00 s ± 0.00 s | 0.0 ms / 0.0 ms | 0.00x RT | 0,000 MB | 0,000 MB |
| **Delta / Speedup** | — | **~0.00x Faster** | **-00.0% Latency** | **+00.0% Throughput** | **-00.0% VRAM** | **-00.0% RAM** |

*Session IDs: `session_<timestamp>_<id>` (Optimized) vs `session_<timestamp>_<id>` (Baseline).*
```

---

### Machine-Readable JSON Schema (Draft 2020-12)

Benchmark results can be validated against the formal schema below:

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "RLDUVRBenchmarkSession",
  "type": "object",
  "required": [
    "session_id",
    "timestamp_utc",
    "model_info",
    "summary"
  ],
  "properties": {
    "session_id": { "type": "string" },
    "timestamp_utc": { "type": "string" },
    "model_info": {
      "type": "object",
      "required": ["model_name", "backend", "device", "precision", "overlap", "segment_size"],
      "properties": {
        "model_name": { "type": "string" },
        "backend": { "type": "string" },
        "device": { "type": "string" },
        "precision": { "type": "string" },
        "overlap": { "type": "number" },
        "segment_size": { "type": "integer" }
      }
    },
    "summary": {
      "type": "object",
      "required": [
        "warmup_rounds",
        "rounds_measured",
        "avg_total_time_sec",
        "avg_speed_factor_rt",
        "peak_device_vram_mb"
      ],
      "properties": {
        "warmup_rounds": { "type": "integer" },
        "rounds_measured": { "type": "integer" },
        "avg_total_time_sec": { "type": "number" },
        "median_total_time_sec": { "type": "number" },
        "avg_speed_factor_rt": { "type": "number" },
        "median_speed_factor_rt": { "type": "number" },
        "min_total_time_sec": { "type": "number" },
        "max_total_time_sec": { "type": "number" },
        "peak_device_vram_mb": { "type": "number" },
        "peak_torch_allocated_vram_mb": { "type": "number" },
        "chunk_latency_p50_ms": { "type": "number" },
        "chunk_latency_p95_ms": { "type": "number" }
      }
    }
  }
}
```

---

### Valid Session JSON Example (Placeholders Only)

*Note: All values below are illustrative placeholders per project contribution standards.*

```json
{
  "session_id": "session_<YYYYMMDD_HHMMSS>_<hex>",
  "timestamp_utc": "2026-01-01T00:00:00Z",
  "model_info": {
    "model_name": "UVR-MDX-NET-Inst_HQ_4.onnx",
    "backend": "MDX-Net",
    "device": "trt_rtx:0",
    "precision": "Ultra Quality (FP32)",
    "overlap": 0.75,
    "segment_size": 4000
  },
  "summary": {
    "warmup_rounds": 2,
    "rounds_measured": 9,
    "avg_total_time_sec": 0.00,
    "median_total_time_sec": 0.00,
    "avg_speed_factor_rt": 0.00,
    "median_speed_factor_rt": 0.00,
    "min_total_time_sec": 0.00,
    "max_total_time_sec": 0.00,
    "peak_device_vram_mb": 0.0,
    "peak_torch_allocated_vram_mb": 0.0,
    "chunk_latency_p50_ms": 0.0,
    "chunk_latency_p95_ms": 0.0
  }
}
```

---

## 8. Common Benchmarking Pitfalls

1. **Averaging the Cold Run**: Counting the initial compilation pass into the average latency drastically inflates steady-state timing.
2. **Thermal Throttling**: Running 20 consecutive passes on an overheated GPU will skew measurements downward. Ensure adequate cooling pauses and use interleaved testing.
3. **Silent Provider Fallbacks**: Always confirm `Active execution provider` in terminal logs to ensure your run actually evaluated on the GPU.
4. **Different Test Audio Lengths**: Comparing speed factor across tracks of differing audio lengths or sample rates introduces bias.
5. **Missing Explicit Precision Flag**: Forgetting `--precision fp32` in the CLI causes `uvr_cli bench` to run in its default FP16 mode instead of the required FP32 baseline.
