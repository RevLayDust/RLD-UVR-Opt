# Contributing to RLD UVR-Opt

Thank you for your interest in contributing to **RLD UVR-Opt**! We welcome bug fixes, performance optimizations, hardware verification, and documentation improvements.

This project is a performance-optimized fork of [Ultimate Vocal Remover GUI](https://github.com/Anjok07/ultimatevocalremovergui) (v5.6) focusing on low-latency ONNX Runtime inference, hardware-accelerated Execution Providers (CUDA and TensorRT-RTX), and modular environment architecture for Windows x64.

---

## TL;DR 5-Step Contribution Flow

1. **Fork & Clone**: Fork the repository on GitHub and clone your fork locally:
   ```powershell
   git clone https://github.com/<your-username>/RLD-UVR-Opt.git
   cd "RLD-UVR-Opt"
   ```

2. **Set Up the Environment**: Run the modular PowerShell installer to configure a virtual environment:
   ```powershell
   powershell -NoProfile -ExecutionPolicy Bypass -File .\install\Install-UVR.ps1 -Mode A
   ```
   *(Select `A` for NVIDIA CUDA, `B` for TensorRT-RTX, or `C` for CPU-only. See [docs/development/setup.md](docs/development/setup.md) for manual setup.)*

3. **Create a Feature Branch**: Branch from the default branch with a descriptive prefix:
   ```powershell
   git checkout -b feat/my-optimization
   # Or fix/my-bugfix, docs/my-documentation
   ```

4. **Verify Locally**: Run test suites and verify line endings before committing:
   ```powershell
   # Run all Python regression tests
   .\venv\Scripts\python.exe -m unittest discover tests -v

   # Run PowerShell installer tests (if touching install scripts)
   powershell -NoProfile -ExecutionPolicy Bypass -Command "Invoke-Pester -Path ./install/tests"
   ```

5. **Submit a Pull Request**: Push your branch to GitHub and open a Pull Request using our [PR Template](.github/PULL_REQUEST_TEMPLATE.md).

---

## Before You Begin

- **Check Existing Issues**: Search [GitHub Issues](https://github.com/RevLayDust/RLD-UVR-Opt/issues) to avoid duplicate work.
- **Discuss Major Changes First**: If you plan to introduce an architectural change, rewrite chunking math, add dependencies, or alter CLI arguments, open an issue labeled `RFC` or `Feature Request` first.
- **Do Not Commit Large Files**: Never commit raw model checkpoints (`*.onnx`, `*.pth`, `*.ckpt`), audio tracks, or temporary stems. The [.gitignore](.gitignore) file excludes model weights and output directories.

---

## Detailed Development Guides

To keep this entry point concise, detailed documentation is organized in `docs/development/`:

| Guide | Description |
| :--- | :--- |
| [**Environment Setup**](docs/development/setup.md) | System prerequisites, installation modes (A, B, C), verification steps, and common troubleshooting. |
| [**Architecture Overview**](docs/development/architecture.md) | Module tree, Execution Provider fallback ladder, inference engine decoupling, and data flow. |
| [**Coding Standards**](docs/development/coding-standards.md) | Python & PowerShell rules, lazy imports, context locking avoidance, and commit conventions. |
| [**Testing Guide**](docs/development/testing.md) | Running unit tests, smoke tests, Pester suites, and testing without physical GPUs. |
| [**Benchmarking Methodology**](docs/development/benchmarking.md) | Mandatory performance profiling, telemetry capture, audio integrity verification, and report formats. |
| [**Installer Development**](docs/development/installer.md) | The 5-phase installer engine (`Detect` -> `Plan` -> `Verify` -> `Install` -> `Validate`), `-DryRun`, and Pester fixtures. |

---

## Pull Request Checklist

When opening a Pull Request, ensure:

- [ ] All tests pass via `.\venv\Scripts\python.exe -m unittest discover tests`.
- [ ] Installer tests pass via `Invoke-Pester -Path ./install/tests` if scripts in `install/` were modified.
- [ ] No extraneous model weights, temporary stems, or private WAV files are committed.
- [ ] Batch files maintain Windows CRLF line endings (tested via `tests.test_windows_batch_files`).
- [ ] If changing inference logic or EP selection, verified benchmark results using `uvr_cli bench` are attached.
- [ ] Code follows [Coding Standards](docs/development/coding-standards.md) (lazy imports, no Tkinter dependencies in computation modules).
- [ ] Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/) (e.g., `feat:`, `fix:`, `docs:`, `refactor:`).

---

## Reporting Bugs and Hardware Benchmarks

- **Bug Reports**: Open an issue using the [Bug Report Template](.github/ISSUE_TEMPLATE/bug_report.yml). Attach your `install/install_record.json` and error logs.
- **Hardware Benchmarks**: We actively collect performance telemetry across diverse GPU and CPU architectures. Please submit results using the [Performance Report Template](.github/ISSUE_TEMPLATE/performance_report.yml).

---

## License & Attribution

- **Core Application & GUI**: Licensed under the [MIT License](LICENSE). Upstream Ultimate Vocal Remover GUI is developed by Anjok07 and the UVR community (Copyright &copy; 2022 Ultimate Vocal Remover). Modifications, TensorRT-RTX backend, and installer pipelines are contributed by RevLayDust and project contributors (Copyright &copy; 2026 RevLayDust).
- **Headless CLI & Benchmark Engine (`uvr_cli/`)**: The CLI-only execution interface and benchmarking engine in `uvr_cli/` is licensed under the [GNU Affero General Public License v3.0 or later](uvr_cli/__init__.py) (`SPDX-License-Identifier: AGPL-3.0-or-later`, Copyright &copy; 2026 RevLayDust). Any new contributions inside `uvr_cli/` must comply with AGPL v3.
- **Third-Party Components**: Model architectures (MDX-Net, Demucs, VR Arch), weights, and execution provider binaries retain their respective upstream licenses.

---

## Community & Questions

- Have a question or proposal? Open a discussion on [GitHub Discussions](https://github.com/RevLayDust/RLD-UVR-Opt/discussions) or create an issue with the `question` label.
- Please review and abide by our [Code of Conduct](CODE_OF_CONDUCT.md) in all project interactions.

