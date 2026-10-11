# Plan.ps1 - Phase 2: Pure deterministic function producing an install plan
Set-StrictMode -Version Latest

function New-InstallPlan {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)]
        [PSCustomObject]$Facts,

        [Parameter(Mandatory = $true)]
        [ValidateSet('A', 'B', 'C')]
        [string]$Choice,

        [Parameter(Mandatory = $false)]
        [PSCustomObject]$Config = $null,

        [Parameter(Mandatory = $false)]
        [string]$ReuseVenv = 'Y'
    )

    if ($null -eq $Config) {
        $Config = Get-InstallerConfig
    }

    $actions = @()
    $orderIndex = 1

    # 1. Venv Strategy
    $venvStrategy = 'Reuse'
    $venvReason = 'Venv already exists and is healthy.'

    if (-not $Facts.Venv.Exists) {
        $venvStrategy = 'Create'
        $venvReason = 'Virtual environment does not exist.'
    } elseif ($ReuseVenv -eq 'N') {
        $venvStrategy = 'Recreate'
        $venvReason = 'User chose not to reuse existing venv.'
    } elseif (-not $Facts.Venv.Is64Bit) {
        $venvStrategy = 'Recreate'
        $venvReason = 'Existing venv is not 64-bit.'
    } elseif ($Facts.Venv.PythonVersion -ne 'Unknown' -and -not ($Facts.Venv.PythonVersion -like '3.11*')) {
        $venvStrategy = 'Recreate'
        $venvReason = "Existing venv uses Python $($Facts.Venv.PythonVersion), but 3.11.x is required."
    }

    $actions += [PSCustomObject]@{
        Component   = 'Venv'
        Current     = if ($Facts.Venv.Exists) { "Python $($Facts.Venv.PythonVersion)" } else { 'None' }
        Target      = "Python $($Config.App.PythonVersion)"
        Action      = $venvStrategy
        Reason      = $venvReason
        Destructive = ($venvStrategy -eq 'Recreate')
        Order       = $orderIndex++
    }

    # 2. Prerequisites: uv, VC++ Redistributable, FFmpeg, Rubberband
    # uv
    if ($Facts.Tools.UvInstalled) {
        $actions += [PSCustomObject]@{
            Component   = 'uv'
            Current     = $Facts.Tools.UvVersion
            Target      = 'Latest'
            Action      = 'Keep'
            Reason      = 'uv is already installed.'
            Destructive = $false
            Order       = $orderIndex++
        }
    } else {
        $actions += [PSCustomObject]@{
            Component   = 'uv'
            Current     = 'None'
            Target      = 'Latest'
            Action      = 'Install'
            Reason      = 'uv is required for fast environment management.'
            Destructive = $false
            Order       = $orderIndex++
        }
    }

    # VC++ Redistributable
    if ($Facts.Tools.VcRedistOk) {
        $actions += [PSCustomObject]@{
            Component   = 'VCRedist'
            Current     = $Facts.Tools.VcRedistVersion
            Target      = ">= $($Config.ExternalTools.VcRedistMinMajor).$($Config.ExternalTools.VcRedistMinMinor)"
            Action      = 'Keep'
            Reason      = 'Visual C++ Redistributable meets version requirement.'
            Destructive = $false
            Order       = $orderIndex++
        }
    } else {
        $actions += [PSCustomObject]@{
            Component   = 'VCRedist'
            Current     = $Facts.Tools.VcRedistVersion
            Target      = ">= $($Config.ExternalTools.VcRedistMinMajor).$($Config.ExternalTools.VcRedistMinMinor)"
            Action      = 'Install'
            Reason      = 'Required runtime for C-extension modules.'
            Destructive = $false
            Order       = $orderIndex++
        }
    }

    # FFmpeg
    if ($Facts.Tools.FFmpegInstalled) {
        $actions += [PSCustomObject]@{
            Component   = 'FFmpeg'
            Current     = "$($Facts.Tools.FFmpegSource): $($Facts.Tools.FFmpegPath)"
            Target      = 'Present'
            Action      = 'Keep'
            Reason      = 'FFmpeg already available.'
            Destructive = $false
            Order       = $orderIndex++
        }
    } else {
        $actions += [PSCustomObject]@{
            Component   = 'FFmpeg'
            Current     = 'None'
            Target      = 'Present (Root)'
            Action      = 'Install'
            Reason      = 'Required for audio decoding and conversions.'
            Destructive = $false
            Order       = $orderIndex++
        }
    }

    # Rubberband
    if ($Facts.Tools.RubberbandInstalled) {
        $actions += [PSCustomObject]@{
            Component   = 'Rubberband'
            Current     = "$($Facts.Tools.RubberbandSource): $($Facts.Tools.RubberbandPath)"
            Target      = 'Present'
            Action      = 'Keep'
            Reason      = 'Rubberband already available.'
            Destructive = $false
            Order       = $orderIndex++
        }
    } else {
        $actions += [PSCustomObject]@{
            Component   = 'Rubberband'
            Current     = 'None'
            Target      = 'Present (Root)'
            Action      = 'Install'
            Reason      = 'Required for time-stretch and pitch-shift tools.'
            Destructive = $false
            Order       = $orderIndex++
        }
    }

    # 3. PyTorch Stack
    $targetTorchVariant = if ($Choice -in @('A', 'B')) { 'CUDA' } else { 'CPU' }
    $targetTorchIndex = if ($targetTorchVariant -eq 'CUDA') { $Config.Torch.Cu128IndexUrl } else { $Config.Torch.CpuIndexUrl }
    $targetTorchVer = "$($Config.Torch.Version)$(if ($targetTorchVariant -eq 'CUDA') { '+cu128' } else { '+cpu' })"

    $torchAction = 'Install'
    $torchReason = 'Install target PyTorch packages.'
    $torchDestructive = $false

    $expectedTag = if ($targetTorchVariant -eq 'CUDA') { 'cu128' } else { 'cpu' }

    if ($venvStrategy -eq 'Recreate' -or -not $Facts.Packages.TorchInstalled) {
        $torchAction = 'Install'
        $torchReason = "Fresh installation of PyTorch $targetTorchVariant."
    } elseif ($Facts.Packages.TorchVariant -ne $targetTorchVariant) {
        $torchAction = 'Replace'
        $torchReason = "Torch variant mismatch: current is $($Facts.Packages.TorchVariant), target is $targetTorchVariant."
        $torchDestructive = $true
    } elseif ($Facts.Packages.TorchLocalTag -ne $expectedTag) {
        $torchAction = 'Replace'
        $torchReason = "Torch build mismatch: $($Facts.Packages.TorchLocalTag) vs $($Config.Torch.TargetCuda)."
        $torchDestructive = $true
    } elseif ($Facts.Packages.TorchvisionVariant -ne $targetTorchVariant -or $Facts.Packages.TorchaudioVariant -ne $targetTorchVariant) {
        $torchAction = 'Replace'
        $torchReason = "Ecosystem mismatch: torchvision ($($Facts.Packages.TorchvisionVariant)) or torchaudio ($($Facts.Packages.TorchaudioVariant)) differs from target $targetTorchVariant."
        $torchDestructive = $true
    } else {
        $torchAction = 'Keep'
        $torchReason = "PyTorch $targetTorchVariant stack is already correct."
    }

    $actions += [PSCustomObject]@{
        Component   = 'PyTorch'
        Current     = "$($Facts.Packages.TorchVariant) ($($Facts.Packages.TorchVersion))"
        Target      = "$targetTorchVariant ($targetTorchVer)"
        Action      = $torchAction
        Reason      = $torchReason
        Destructive = $torchDestructive
        Order       = $orderIndex++
    }

    # 4. ONNX Runtime & Plugins
    if ($Choice -eq 'A') {
        # Profile A: onnxruntime-gpu==1.22.0, NO onnxruntime (cpu), NO TRT-RTX plugin
        if ($Facts.Packages.HasOrtCpu) {
            $actions += [PSCustomObject]@{
                Component   = 'onnxruntime-cpu'
                Current     = $Facts.Packages.OrtCpuVersion
                Target      = 'None'
                Action      = 'Remove'
                Reason      = 'Conflicting CPU ONNX Runtime package must be uninstalled for CUDA EP.'
                Destructive = $true
                Order       = $orderIndex++
            }
        }
        if ($Facts.Packages.HasTrtPlugin) {
            $actions += [PSCustomObject]@{
                Component   = 'TRT-RTX-Plugin'
                Current     = $Facts.Packages.TrtPluginVersion
                Target      = 'None'
                Action      = 'Remove'
                Reason      = 'Switching to standard CUDA EP; removing TRT-RTX plugin.'
                Destructive = $true
                Order       = $orderIndex++
            }
        }
        $ortAction = 'Install'
        $ortReason = "Install onnxruntime-gpu $($Config.Ort.CudaVersion)."
        $ortDestructive = $false
        if ($Facts.Packages.HasOrtGpu -and $Facts.Packages.OrtGpuVersion -eq $Config.Ort.CudaVersion -and -not $Facts.Packages.HasOrtCpu) {
            $ortAction = 'Keep'
            $ortReason = 'onnxruntime-gpu 1.22.0 already matches target.'
        } elseif ($Facts.Packages.HasOrtGpu) {
            $ortAction = 'Replace'
            $ortReason = "Version change ($($Facts.Packages.OrtGpuVersion) -> $($Config.Ort.CudaVersion))."
            $ortDestructive = $true
        }
        $actions += [PSCustomObject]@{
            Component   = 'onnxruntime-gpu'
            Current     = if ($Facts.Packages.HasOrtGpu) { $Facts.Packages.OrtGpuVersion } else { 'None' }
            Target      = $Config.Ort.CudaVersion
            Action      = $ortAction
            Reason      = $ortReason
            Destructive = $ortDestructive
            Order       = $orderIndex++
        }
    } elseif ($Choice -eq 'B') {
        # Profile B: onnxruntime-gpu==1.24.4 + TRT-RTX plugin 0.4.0, NO onnxruntime (cpu)
        if ($Facts.Packages.HasOrtCpu) {
            $actions += [PSCustomObject]@{
                Component   = 'onnxruntime-cpu'
                Current     = $Facts.Packages.OrtCpuVersion
                Target      = 'None'
                Action      = 'Remove'
                Reason      = 'Conflicting CPU ONNX Runtime package must be uninstalled for TensorRT-RTX.'
                Destructive = $true
                Order       = $orderIndex++
            }
        }
        $ortAction = 'Install'
        $ortReason = "Install onnxruntime-gpu $($Config.Ort.TrtGpuVersion)."
        $ortDestructive = $false
        if ($Facts.Packages.HasOrtGpu -and $Facts.Packages.OrtGpuVersion -eq $Config.Ort.TrtGpuVersion -and -not $Facts.Packages.HasOrtCpu) {
            $ortAction = 'Keep'
            $ortReason = "onnxruntime-gpu $($Config.Ort.TrtGpuVersion) already matches target."
        } elseif ($Facts.Packages.HasOrtGpu) {
            $ortAction = 'Replace'
            $ortReason = "Version upgrade ($($Facts.Packages.OrtGpuVersion) -> $($Config.Ort.TrtGpuVersion))."
            $ortDestructive = $true
        }
        $actions += [PSCustomObject]@{
            Component   = 'onnxruntime-gpu'
            Current     = if ($Facts.Packages.HasOrtGpu) { $Facts.Packages.OrtGpuVersion } else { 'None' }
            Target      = $Config.Ort.TrtGpuVersion
            Action      = $ortAction
            Reason      = $ortReason
            Destructive = $ortDestructive
            Order       = $orderIndex++
        }
        $pluginAction = 'Install'
        $pluginReason = "Install TensorRT-RTX plugin $($Config.Ort.TrtPluginVersion)."
        $pluginDestructive = $false
        if ($Facts.Packages.HasTrtPlugin -and $Facts.Packages.TrtPluginVersion -eq $Config.Ort.TrtPluginVersion) {
            $pluginAction = 'Keep'
            $pluginReason = 'TensorRT-RTX plugin already matches target.'
        } elseif ($Facts.Packages.HasTrtPlugin) {
            $pluginAction = 'Replace'
            $pluginReason = "Plugin version upgrade ($($Facts.Packages.TrtPluginVersion) -> $($Config.Ort.TrtPluginVersion))."
            $pluginDestructive = $true
        }
        $actions += [PSCustomObject]@{
            Component   = 'TRT-RTX-Plugin'
            Current     = if ($Facts.Packages.HasTrtPlugin) { $Facts.Packages.TrtPluginVersion } else { 'None' }
            Target      = $Config.Ort.TrtPluginVersion
            Action      = $pluginAction
            Reason      = $pluginReason
            Destructive = $pluginDestructive
            Order       = $orderIndex++
        }
    } else {
        # Profile C: onnxruntime==1.22.0, NO onnxruntime-gpu, NO TRT-RTX plugin
        if ($Facts.Packages.HasOrtGpu) {
            $actions += [PSCustomObject]@{
                Component   = 'onnxruntime-gpu'
                Current     = $Facts.Packages.OrtGpuVersion
                Target      = 'None'
                Action      = 'Remove'
                Reason      = 'Conflicting GPU ONNX Runtime package must be uninstalled for CPU mode.'
                Destructive = $true
                Order       = $orderIndex++
            }
        }
        if ($Facts.Packages.HasTrtPlugin) {
            $actions += [PSCustomObject]@{
                Component   = 'TRT-RTX-Plugin'
                Current     = $Facts.Packages.TrtPluginVersion
                Target      = 'None'
                Action      = 'Remove'
                Reason      = 'GPU TensorRT plugin uninstalled for CPU mode.'
                Destructive = $true
                Order       = $orderIndex++
            }
        }
        $ortAction = 'Install'
        $ortReason = "Install onnxruntime CPU $($Config.Ort.CpuVersion)."
        $ortDestructive = $false
        if ($Facts.Packages.HasOrtCpu -and $Facts.Packages.OrtCpuVersion -eq $Config.Ort.CpuVersion -and -not $Facts.Packages.HasOrtGpu) {
            $ortAction = 'Keep'
            $ortReason = 'onnxruntime CPU already matches target.'
        } elseif ($Facts.Packages.HasOrtCpu) {
            $ortAction = 'Replace'
            $ortReason = "Version change ($($Facts.Packages.OrtCpuVersion) -> $($Config.Ort.CpuVersion))."
            $ortDestructive = $true
        }
        $actions += [PSCustomObject]@{
            Component   = 'onnxruntime-cpu'
            Current     = if ($Facts.Packages.HasOrtCpu) { $Facts.Packages.OrtCpuVersion } else { 'None' }
            Target      = $Config.Ort.CpuVersion
            Action      = $ortAction
            Reason      = $ortReason
            Destructive = $ortDestructive
            Order       = $orderIndex++
        }
    }

    # 5. Base Dependencies
    $reqFile = switch ($Choice) {
        'A' { $Config.Files.RequirementsMain }
        'B' { $Config.Files.RequirementsTrtRtx }
        'C' { $Config.Files.RequirementsCpu }
    }
    $actions += [PSCustomObject]@{
        Component   = 'Requirements'
        Current     = 'State-Dependent'
        Target      = $reqFile
        Action      = 'Install'
        Reason      = "Install application dependencies from $reqFile."
        Destructive = $false
        Order       = $orderIndex++
    }

    $isAnyDestructive = @($actions | Where-Object { $_.Destructive -eq $true }).Count -gt 0
    $isNothingToDo = @($actions | Where-Object { $_.Action -notin @('Keep') }).Count -eq 0

    return [PSCustomObject]@{
        Profile            = $Choice
        TargetTorchVariant = $targetTorchVariant
        TargetTorchIndex   = $targetTorchIndex
        TargetTorchVer     = $targetTorchVer
        RequirementsFile   = $reqFile
        VenvStrategy       = $venvStrategy
        HasDestructive     = $isAnyDestructive
        IsNothingToDo      = $isNothingToDo
        Actions            = $actions
    }
}
