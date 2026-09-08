# Contributions

The CLI and performance benchmarking suite is located in `uvr_cli/`. The implementation and output format are documented in `uvr_cli/README.md`.

Do not commit private audio, model files, generated stems, or local `benchmark_results/` logs. Before submitting benchmark changes, run:

```powershell
venv\Scripts\python.exe -m unittest tests.test_uvr_cli tests.test_in_memory_model_cache -v
```

