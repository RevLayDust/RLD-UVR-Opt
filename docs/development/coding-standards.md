# Coding Standards & Architecture Constraints

This document defines the coding standards, project invariants, and architectural anti-patterns enforced in **RLD UVR-Opt**.

---

## 1. Technical Invariants & Anti-Patterns

### A. Deferred & Lazy Imports
Heavy machine learning dependencies (`torch`, `onnxruntime`, `tensorrt_rtx`) must **not** be imported at module root level during application startup. Importing heavy DLLs during root load slows down CLI command dispatch (`uvr_cli --help`, `list-models`) and causes unnecessary cold-start latency.

#### Bad (Root-Level Eager Import)
```python
# BAD: Imported when module is evaluated, slowing CLI launch
import torch
import onnxruntime as ort

def process_audio(audio_path: str) -> None:
    session = ort.InferenceSession("model.onnx")
    ...
```

#### Good (Deferred Local Import)
```python
# GOOD: Imports are deferred until the inference function is actually called
def process_audio(audio_path: str) -> None:
    import torch
    import onnxruntime as ort
    from engine.tensorrt_rtx.session import create_trt_rtx_session

    session = create_trt_rtx_session("model.onnx")
    ...
```

---

### B. Context Lock Prevention
Initializing CUDA context or calling `torch.cuda.init()` in the main thread before Execution Provider resolution locks GPU device context. This can cause ONNX Runtime providers (especially TensorRT) to fail or run into resource contention.

#### Bad (Premature CUDA Initialization)
```python
# BAD: Eagerly queries device before user or CLI arguments select EP
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch.cuda.init()
```

#### Good (Deferred Device Binding)
```python
# GOOD: Defer device context initialization until model session is instantiated
def resolve_inference_device(device_arg: str) -> str:
    # Validate string argument without initializing CUDA driver context
    if device_arg.startswith("cuda") or device_arg.startswith("trt_rtx"):
        return device_arg
    return "cpu"
```

---

### C. Pure Pipeline Decoupling
Inference routines in `separate.py`, `engine/tensorrt_rtx/`, and `uvr_cli/` must remain completely decoupled from GUI presentation widgets (Tkinter, `ttk`).

- **Forbidden**: Passing `tk.Tk`, `ttk.Progressbar`, or referencing GUI global state inside model runner functions.
- **Allowed**: Standard function signatures accepting primitive arguments (`str`, `Path`, `float`, `int`, `dict`) and reporting progress through callable callbacks:
  ```python
  def run_separation(
      audio_path: Path,
      model_path: Path,
      progress_callback: Optional[Callable[[float, str], None]] = None,
  ) -> Tuple[Dict[str, np.ndarray], bool]:
      ...
      if progress_callback:
          progress_callback(0.5, "Processing chunk 2/4...")
  ```

---

### D. PowerShell Invariants & Error Handling
All scripts in [install/](../../install/) must maintain strict compatibility with **Windows PowerShell 5.1**:

1. **Strict Mode**: Every `.ps1` file must begin with:
   ```powershell
   Set-StrictMode -Version Latest
   ```
2. **Native Process Invocation**: Every external command (`python.exe`, `uv.exe`, `git.exe`) must verify exit codes:
   ```powershell
   # Use the shared Invoke-Native helper or check $LASTEXITCODE explicitly
   $res = Invoke-Native -FilePath 'uv.exe' -ArgumentList @('pip', 'install', '-r', $reqFile)
   if (-not $res.Success) {
       throw "Command failed with exit code $($res.ExitCode): $($res.StdErr)"
   }
   ```
3. **No PS7 Syntactic Sugar**: Do not use PowerShell 7+ features:
   - Ternary operator (`?:`) &rarr; use `if (...) { } else { }`.
   - Null-coalescing (`??`) &rarr; use `if ($null -eq $x) { $x = ... }`.
   - Pipeline chaining (`&&`, `||`) &rarr; use explicit conditional statements.
4. **Encoding**: Save PowerShell scripts as UTF-8 without BOM or ASCII.

---

### E. Installer Architectural Invariants
1. **Zero Side-Effects in Early Phases**:
   - `Detect.ps1` (Phase 1), `Plan.ps1` (Phase 2), and `Verify.ps1` (Phase 3) must be strictly read-only.
   - `-DryRun` must exit at Phase 3 with zero modifications to filesystem, registry, or environment.
2. **Single Source of Truth**:
   - All version pins (PyTorch, ONNX Runtime, plugins, driver thresholds) must reside exclusively in [install/lib/Config.ps1](../../install/lib/Config.ps1).
3. **Additive Record Schema**:
   - Changes to [install/install_record.json](../../install/install_record.json) must be purely additive to preserve backward compatibility with `install_record.py inspect-stack`.
4. **No Silent EP Fallbacks**:
   - Inference sessions must explicitly inspect `session.get_providers()` and log the active provider.

---

## 2. Python Code Standards

- **Formatting**: Adhere to PEP 8 with 4-space indentation.
- **Type Annotations**: Provide type hints for public module functions and classes.
- **Logging vs Print**: In library code (`engine/`, `uvr_cli/`), use structured logging or verbose flags rather than raw `print()` statements.
- **Error Handling**: Catch specific exceptions rather than bare `except:`. Provide descriptive error messages.

---

## 3. Git Commit & Branch Conventions

### Commit Messages
We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```text
<type>(<scope>): <short description in imperative mood>

[optional body explaining context and rationale]

[optional footer(s)]
```

#### Allowed Types:
- `feat`: A new feature (e.g., new Execution Provider, CLI flag).
- `fix`: A bug fix (e.g., resolver error, memory leak).
- `docs`: Documentation updates.
- `refactor`: Code restructuring without functional changes.
- `perf`: A code change that improves performance.
- `test`: Adding or correcting tests.
- `chore`: Maintenance tasks (release metadata, config updates).

#### Example:
```text
feat(engine): add Ampere architecture guard for TensorRT-RTX

- Check compute capability >= 8.0 via pynvml
- Fall back gracefully to CUDAExecutionProvider on Turing GPUs
- Add unit test coverage in tests/test_tensorrt_rtx.py
```

### Branch Naming
- Features: `feat/<feature-name>`
- Bugfixes: `fix/<bug-name>`
- Documentation: `docs/<doc-name>`
