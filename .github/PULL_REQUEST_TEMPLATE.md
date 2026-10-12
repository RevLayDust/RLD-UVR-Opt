## Description

<!-- Provide a concise summary of your changes and motivation. -->

## Related Issue

<!-- Closes #123 / Relates to #456 -->

## Type of Change

- [ ] `fix`: Bug fix (non-breaking change fixing an issue)
- [ ] `feat`: New feature or Execution Provider support
- [ ] `perf`: Performance optimization or memory reduction
- [ ] `refactor`: Code reorganization without functional changes
- [ ] `docs`: Documentation update or addition
- [ ] `test`: Unit test additions or test fixture updates

## How Has This Been Tested?

<!-- Describe tests run, commands used, and test environment (GPU model, driver version, OS). -->
```powershell
# Examples of executed test commands:
.\venv\Scripts\python.exe -m unittest discover tests -v
powershell -NoProfile -ExecutionPolicy Bypass -Command "Invoke-Pester -Path ./install/tests"
```

## Checklist

- [ ] My code adheres to the project's [Coding Standards](docs/development/coding-standards.md) (lazy imports, context lock prevention, decoupled pipeline).
- [ ] All Python unit tests pass locally (`python -m unittest discover tests`).
- [ ] All Pester installer tests pass locally if `install/` was modified (`Invoke-Pester -Path ./install/tests`).
- [ ] Batch files maintain Windows CRLF line endings (verified via `tests.test_windows_batch_files`).
- [ ] If changing inference, EP selection, or precision logic, benchmark results using `uvr_cli bench` are attached below following [Benchmarking Guidelines](docs/development/benchmarking.md).
- [ ] No raw audio files, stems, or binary model weights (`*.onnx`, `*.pth`) are included in this commit.
- [ ] Documentation has been updated to reflect new CLI flags, options, or behavior.
