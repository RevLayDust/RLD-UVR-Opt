# Contributions

Runtime benchmarking is recorded automatically for each top-level inference started from the UVR GUI. The implementation and output format are documented in `benchmarks/README.md`.

Do not commit private audio, model files, generated stems, or local `benchmark_cache/` logs. Before submitting recorder changes, run:

```powershell
venv\Scripts\python.exe -m unittest tests.test_gui_runtime_recorder tests.test_in_memory_model_cache -v
```
