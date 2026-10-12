# Installer Architecture & Development

This guide documents the architecture, lifecycle phases, and testing practices for the **RLD UVR-Opt** PowerShell installer pipeline located in [install/](../../install/).

---

## 1. Five-Phase Pipeline Architecture

The installer replaces monolithic batch files with a deterministic, modular pipeline implemented in **Windows PowerShell 5.1**:

```text
  [Input Flags / Prompts]
             │
             ▼
   Phase 1: Detect       (lib/Detect.ps1)    - Read-only hardware & package inspection
             │
             ▼
   Phase 2: Plan         (lib/Plan.ps1)      - Pure action mapping (Keep, Install, Replace, Remove)
             │
             ▼
   Phase 3: Verify       (lib/Verify.ps1)    - Read-only gate; checks invariants & -DryRun
             │
             ▼
   Phase 4: Install      (lib/Install.ps1)   - Safe mutation with snapshot rollback & uv execution
             │
             ▼
   Phase 5: Validate     (lib/Validate.ps1)  - Live runtime validation with in-memory ONNX model
             │
             ▼
   Post: Record          (lib/Record.ps1)    - Atomic persistence to install_record.json
```

### Module Responsibilities:

| File | Phase | Responsibility |
| :--- | :--- | :--- |
| [**`Config.ps1`**](../../install/lib/Config.ps1) | Config | **Single Source of Truth**: Package pins (PyTorch, ONNX Runtime, plugins), URLs, driver threshold ($\ge 570$), CC threshold ($\ge 8.0$), and disk space ($5\text{ GB}$). |
| [**`Common.ps1`**](../../install/lib/Common.ps1) | Common | Shared logging (`Write-InstallerLog`), deadlock-free native process runner (`Invoke-Native`), file I/O, and interactive prompts. |
| [**`Detect.ps1`**](../../install/lib/Detect.ps1) | Phase 1 | Inspects OS bitness, GPU controllers via CIM, `nvidia-smi`, Python venv, and `.dist-info` package state (without importing torch). |
| [**`Plan.ps1`**](../../install/lib/Plan.ps1) | Phase 2 | Pure function `New-InstallPlan`: maps current state to desired target profile (A, B, C) and marks destructive operations. |
| [**`Verify.ps1`**](../../install/lib/Verify.ps1) | Phase 3 | Verifies plan invariants: architecture suitability, minimum driver version, disk space, and active process locks. Handles `-DryRun`. |
| [**`Install.ps1`**](../../install/lib/Install.ps1) | Phase 4 | Captures rollback snapshot, cleans variant conflicts, installs wheels and filtered requirements via Astral `uv`. |
| [**`Validate.ps1`**](../../install/lib/Validate.ps1) | Phase 5 | Deep runtime check inside venv: verifies GPU tensor allocation and creates live ONNX Runtime sessions using embedded IR 8 test models. |
| [**`Record.ps1`**](../../install/lib/Record.ps1) | Record | Atomically writes [install_record.json](../../install/install_record.json) while delegating base schema to `install_record.py write`. |

---

## 2. Running Non-Destructive Dry Runs

Contributors can simulate any installation or migration scenario on their local machine without touching the filesystem or installing packages:

```powershell
# Simulate Option A (CUDA)
powershell -NoProfile -ExecutionPolicy Bypass -File .\install\Install-UVR.ps1 -Mode A -DryRun

# Simulate Option B (TensorRT-RTX)
powershell -NoProfile -ExecutionPolicy Bypass -File .\install\Install-UVR.ps1 -Mode B -DryRun

# Simulate Option C (CPU)
powershell -NoProfile -ExecutionPolicy Bypass -File .\install\Install-UVR.ps1 -Mode C -DryRun
```

A `-DryRun` prints the generated plan table and verification status, then exits with code 0 without performing changes.

---

## 3. Single Source of Truth Rule

All version pins, index URLs, and hardware thresholds **must reside exclusively** in [install/lib/Config.ps1](../../install/lib/Config.ps1). Never hardcode package versions or driver versions inside `Detect.ps1`, `Plan.ps1`, or `Install.ps1`.

### Example: Adding or Updating a Dependency Pin
To bump PyTorch or ONNX Runtime, update the `Get-InstallerConfig` hashtable in [install/lib/Config.ps1](../../install/lib/Config.ps1):

```powershell
Torch = [PSCustomObject]@{
    Version = '2.11.0'
    CudaIndexUrl = 'https://download.pytorch.org/whl/cu128'
    CpuIndexUrl = 'https://download.pytorch.org/whl/cpu'
}
```

---

## 4. Writing Pester Tests for Installer Scenarios

Installer logic is thoroughly covered by Pester test fixtures located in [install/tests/](../../install/tests/).

### Adding a New Planning Scenario
To test plan generation for a new package state, add an `It` block to [install/tests/Plan.Tests.ps1](../../install/tests/Plan.Tests.ps1) using `New-TestFacts`:

```powershell
It 'Scenario: Venv with custom package condition' {
    $facts = New-TestFacts -TorchVariant 'CUDA' -TorchVersion '2.11.0' -HasOrtGpu $true
    $plan = New-InstallPlan -Facts $facts -Choice 'B'

    $plan.Profile | Should Be 'B'
    $plan.Actions | Where-Object { $_.Component -eq 'TRT-RTX-Plugin' } | Should Not BeNullOrEmpty
}
```

### Adding a New Verification Gate
To test a hardware gate or safety block, add an `It` block to [install/tests/Verify.Tests.ps1](../../install/tests/Verify.Tests.ps1):

```powershell
It 'Scenario: System with insufficient memory' {
    $facts = New-TestFacts -FreeDiskSpaceBytes (2GB) # Less than 5GB required
    $plan = New-InstallPlan -Facts $facts -Choice 'A'
    $verdict = Test-InstallPlan -Facts $facts -Choice 'A' -Plan $plan

    $verdict.Status | Should Be 'Blocked'
}
```

Execute the test suite to verify:
```powershell
powershell -NoProfile -ExecutionPolicy Bypass -Command "Invoke-Pester -Path ./install/tests"
```

---

## 5. The `install_record.py` Contract

To maintain full backward compatibility with legacy tooling and the GUI:
1. `install_record.py inspect-stack`:
   - Prints `TORCH_VARIANT|TORCH_VERSION|ORT_VARIANT|ORT_VERSION` used by external inspectors.
2. `install_record.py write`:
   - Writes the base schema fields (`schema_version`, `timestamp_utc`, `selected_option`, `install_mode`).
3. **Additive Extensions**:
   - [install/lib/Record.ps1](../../install/lib/Record.ps1) injects additional telemetry fields (`effective_option`, `detected`, `validation`) atomically without breaking base consumers.
