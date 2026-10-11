# Plan.Tests.ps1 - Pester tests for Phase 2: Install Plan generation
Set-StrictMode -Version Latest

$testDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$libDir = Join-Path -Path (Split-Path -Parent $testDir) -ChildPath 'lib'

. (Join-Path -Path $libDir -ChildPath 'Config.ps1')
. (Join-Path -Path $libDir -ChildPath 'Plan.ps1')
. (Join-Path -Path $testDir -ChildPath 'Fixtures.ps1')

Describe 'Phase 2: Plan Generation Tests' {

    It 'Scenario 6: CPU venv, user picks Option A -> Replaces torch CPU with CUDA, replaces ORT CPU with GPU' {
        $facts = New-TestFacts -TorchVariant 'CPU' -TorchVersion '2.11.0' -TorchLocalTag 'cpu' -TorchvisionVariant 'CPU' -TorchaudioVariant 'CPU' -HasOrtCpu $true -OrtCpuVer '1.22.0'
        $plan = New-InstallPlan -Facts $facts -Choice 'A'

        $plan.Profile | Should Be 'A'
        $plan.TargetTorchVariant | Should Be 'CUDA'
        $plan.HasDestructive | Should Be ($true)

        $torchAction = $plan.Actions | Where-Object { $_.Component -eq 'PyTorch' }
        $torchAction.Action | Should Be 'Replace'
        $torchAction.Destructive | Should Be ($true)

        $ortCpuAction = $plan.Actions | Where-Object { $_.Component -eq 'onnxruntime-cpu' }
        $ortCpuAction.Action | Should Be 'Remove'

        $ortGpuAction = $plan.Actions | Where-Object { $_.Component -eq 'onnxruntime-gpu' }
        $ortGpuAction.Action | Should Be 'Install'

        $uvAction = $plan.Actions | Where-Object { $_.Component -eq 'uv' }
        $uvAction.Action | Should Be 'Keep'
    }

    It 'Scenario 7: Option A venv, user picks Option B -> Torch remains Keep, only ORT upgrades and plugin installs' {
        $facts = New-TestFacts -TorchVariant 'CUDA' -TorchVersion '2.11.0' -TorchLocalTag 'cu128' -TorchvisionVariant 'CUDA' -TorchaudioVariant 'CUDA' -HasOrtGpu $true -OrtGpuVer '1.22.0' -HasTrtPlugin $false
        $plan = New-InstallPlan -Facts $facts -Choice 'B'

        $plan.Profile | Should Be 'B'
        $plan.TargetTorchVariant | Should Be 'CUDA'

        $torchAction = $plan.Actions | Where-Object { $_.Component -eq 'PyTorch' }
        $torchAction.Action | Should Be 'Keep'

        $ortAction = $plan.Actions | Where-Object { $_.Component -eq 'onnxruntime-gpu' }
        $ortAction.Action | Should Be 'Replace'
        $ortAction.Target | Should Be '1.24.4'

        $pluginAction = $plan.Actions | Where-Object { $_.Component -eq 'TRT-RTX-Plugin' }
        $pluginAction.Action | Should Be 'Install'
    }

    It 'Scenario 8: Option B venv, user picks Option A -> Torch remains Keep, plugin is removed, ORT downgraded to 1.22.0' {
        $facts = New-TestFacts -TorchVariant 'CUDA' -TorchVersion '2.11.0' -TorchLocalTag 'cu128' -TorchvisionVariant 'CUDA' -TorchaudioVariant 'CUDA' -HasOrtGpu $true -OrtGpuVer '1.24.4' -HasTrtPlugin $true -TrtPluginVer '0.4.0'
        $plan = New-InstallPlan -Facts $facts -Choice 'A'

        $plan.Profile | Should Be 'A'

        $torchAction = $plan.Actions | Where-Object { $_.Component -eq 'PyTorch' }
        $torchAction.Action | Should Be 'Keep'

        $pluginAction = $plan.Actions | Where-Object { $_.Component -eq 'TRT-RTX-Plugin' }
        $pluginAction.Action | Should Be 'Remove'

        $ortAction = $plan.Actions | Where-Object { $_.Component -eq 'onnxruntime-gpu' }
        $ortAction.Action | Should Be 'Replace'
        $ortAction.Target | Should Be '1.22.0'
    }

    It 'Scenario 9: Venv has both onnxruntime and onnxruntime-gpu conflict -> Removes conflicting variant' {
        $facts = New-TestFacts -TorchVariant 'CUDA' -TorchVersion '2.11.0' -TorchLocalTag 'cu128' -TorchvisionVariant 'CUDA' -TorchaudioVariant 'CUDA' -HasOrtCpu $true -OrtCpuVer '1.22.0' -HasOrtGpu $true -OrtGpuVer '1.22.0'
        $plan = New-InstallPlan -Facts $facts -Choice 'A'

        $ortCpuAction = $plan.Actions | Where-Object { $_.Component -eq 'onnxruntime-cpu' }
        $ortCpuAction.Action | Should Be 'Remove'

        $ortGpuAction = $plan.Actions | Where-Object { $_.Component -eq 'onnxruntime-gpu' }
        $ortGpuAction.Action | Should Be 'Replace'
    }

    It 'Scenario 10: Plain torch with no local tag (2.11.0) -> Treated as CPU, replaced when user picks A' {
        $facts = New-TestFacts -TorchVariant 'CPU' -TorchVersion '2.11.0' -TorchLocalTag 'none' -TorchvisionVariant 'CPU' -TorchaudioVariant 'CPU'
        $plan = New-InstallPlan -Facts $facts -Choice 'A'

        $torchAction = $plan.Actions | Where-Object { $_.Component -eq 'PyTorch' }
        $torchAction.Action | Should Be 'Replace'
        $torchAction.Reason | Should Match 'Torch variant mismatch'
    }

    It 'Scenario 14: Rerun with the exact same choice -> Everything is Keep, no destructive actions' {
        $facts = New-TestFacts -TorchVariant 'CUDA' -TorchVersion '2.11.0' -TorchLocalTag 'cu128' -TorchvisionVariant 'CUDA' -TorchaudioVariant 'CUDA' -HasOrtGpu $true -OrtGpuVer '1.22.0' -HasTrtPlugin $false
        $plan = New-InstallPlan -Facts $facts -Choice 'A'

        $torchAction = $plan.Actions | Where-Object { $_.Component -eq 'PyTorch' }
        $torchAction.Action | Should Be 'Keep'

        $ortAction = $plan.Actions | Where-Object { $_.Component -eq 'onnxruntime-gpu' }
        $ortAction.Action | Should Be 'Keep'

        $plan.HasDestructive | Should Be ($false)
    }
}
