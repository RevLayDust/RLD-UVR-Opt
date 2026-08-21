# GUI Runtime Benchmark Recorder

UVR records runtime benchmark metadata automatically whenever the GUI backend enters a top-level `seperate()` call after **Start Processing**. There is no benchmark CLI, case matrix, profiler command, result packager, or upload workflow.

Supported production backends:

- VR Architecture
- MDX-Net ONNX
- MDX PyTorch/MDXC
- Demucs

## Output

Each application session writes progressively to:

```text
benchmark_cache/gui_runtime/latest.txt
benchmark_cache/gui_runtime/<session>/benchmark.log
benchmark_cache/gui_runtime/<session>/inference_<number>_<model>.log
benchmark_cache/gui_runtime/<session>/inference_<number>_<model>.json
```

Every top-level audio job receives a separate block:

```text
1st Inference:
Model: <model filename>
Model round: 1
OS: <operating system>
...
Total-time: <backend entry through completed output>
=======
```

`Model round` increments while model path and requested precision remain unchanged. Changing either value resets the round to one. Nested secondary separators remain part of the parent pipeline and do not create overlapping benchmark records.

The recorder stores model identity/hash, backend, requested and effective precision, execution provider, input metadata, cache state, stage timings, total time, CPU/RAM, GPU utilization, device/Torch VRAM, power when available, output validation, warnings, and failure state. Start, progress, stage, warning, and completion data are flushed incrementally.

Recorder failures never cancel audio processing. When instrumentation setup fails, UVR continues without instrumentation and closes any record already opened with its final status and total time.

## Configuration

- Set `UVR_GUI_BENCHMARK=0` before launch to disable recording.
- Set `UVR_GUI_BENCHMARK_DIR=<directory>` to override the local output directory.
- Keep `nvidia-ml-py` installed for NVIDIA utilization, VRAM, driver, and power metadata. Missing NVML support produces a warning rather than stopping inference.

## Tests

```powershell
venv\Scripts\python.exe -m unittest tests.test_gui_runtime_recorder tests.test_in_memory_model_cache -v
```

`UVR.py` is not modified by the recorder. Hooks are installed from `separate.py` after all separator classes are defined.
