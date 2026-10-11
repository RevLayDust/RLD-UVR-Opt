# Verify.ps1 - Phase 3: Read-only gate verifying user choice against machine facts
Set-StrictMode -Version Latest

function Test-InstallPlan {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)]
        [PSCustomObject]$Facts,

        [Parameter(Mandatory = $true)]
        [ValidateSet('A', 'B', 'C')]
        [string]$Choice,

        [Parameter(Mandatory = $true)]
        [PSCustomObject]$Plan,

        [Parameter(Mandatory = $false)]
        [PSCustomObject]$Config = $null
    )

    if ($null -eq $Config) {
        $Config = Get-InstallerConfig
    }

    $checks = @()
    $status = 'Approved'
    $recOption = $Choice
    $recReason = ''

    # 1. OS & Architecture Check
    if (-not $Facts.OS.IsWindows) {
        $checks += [PSCustomObject]@{
            Id          = 'OS_PLATFORM'
            Result      = 'Fail'
            Message     = "Operating system is not Windows ($($Facts.OS.VersionString))."
            Remediation = 'This installer only supports Windows 10/11 64-bit.'
        }
        $status = 'Blocked'
    } elseif (-not $Facts.OS.Is64Bit -or $Facts.OS.IsArm64) {
        $checks += [PSCustomObject]@{
            Id          = 'OS_ARCH'
            Result      = 'Fail'
            Message     = "Unsupported processor architecture: $($Facts.OS.Architecture)."
            Remediation = 'Only Windows 64-bit (x64 / AMD64) is supported. 32-bit and ARM64 are unsupported.'
        }
        $status = 'Blocked'
    } else {
        $checks += [PSCustomObject]@{
            Id          = 'OS_ARCH'
            Result      = 'Pass'
            Message     = "Windows 64-bit verified ($($Facts.OS.Architecture))."
            Remediation = ''
        }
    }

    # 2. Process Locking Check
    if ($Facts.LockingProcesses.Count -gt 0) {
        $procNames = ($Facts.LockingProcesses | ForEach-Object { "$($_.Name) (PID $($_.Id))" }) -join ', '
        $checks += [PSCustomObject]@{
            Id          = 'PROCESS_LOCK'
            Result      = 'Fail'
            Message     = "Active process is locking application files: $procNames"
            Remediation = 'Please close Ultimate Vocal Remover (UVR) or running Python instances before running installer.'
        }
        $status = 'Blocked'
    } else {
        $checks += [PSCustomObject]@{
            Id          = 'PROCESS_LOCK'
            Result      = 'Pass'
            Message     = 'No active processes locking venv or application files.'
            Remediation = ''
        }
    }

    # 3. Disk Space Check
    $minSpace = $Config.Hardware.MinFreeDiskSpaceBytes
    if ($Facts.Resources.FreeDiskSpaceBytes -lt $minSpace) {
        $freeGB = [math]::Round($Facts.Resources.FreeDiskSpaceBytes / 1GB, 2)
        $needGB = [math]::Round($minSpace / 1GB, 2)
        $checks += [PSCustomObject]@{
            Id          = 'DISK_SPACE'
            Result      = 'Fail'
            Message     = "Insufficient free disk space on install drive ($freeGB GB available, $needGB GB required)."
            Remediation = "Free up at least $needGB GB on the destination drive."
        }
        $status = 'Blocked'
    } else {
        $freeGB = [math]::Round($Facts.Resources.FreeDiskSpaceBytes / 1GB, 2)
        $checks += [PSCustomObject]@{
            Id          = 'DISK_SPACE'
            Result      = 'Pass'
            Message     = "Sufficient disk space available ($freeGB GB)."
            Remediation = ''
        }
    }

    # 4. Network Connectivity Check
    $isPyPIOnline = $Facts.Resources.Connectivity['https://pypi.org/simple']
    $isTorchOnline = $Facts.Resources.Connectivity['https://download.pytorch.org/whl/cu128']
    if (-not $isPyPIOnline -and -not $isTorchOnline) {
        $checks += [PSCustomObject]@{
            Id          = 'NETWORK'
            Result      = 'Fail'
            Message     = 'No network connectivity to PyPI or PyTorch repositories.'
            Remediation = 'Check your internet connection or proxy/firewall settings.'
        }
        $status = 'Blocked'
    } else {
        $checks += [PSCustomObject]@{
            Id          = 'NETWORK'
            Result      = 'Pass'
            Message     = 'Network connectivity verified.'
            Remediation = ''
        }
    }

    # 5. Hardware & Profile Validation
    if ($Choice -in @('A', 'B')) {
        # Check NVIDIA presence
        if (-not $Facts.Gpu.HasNvidiaAdapter) {
            $checks += [PSCustomObject]@{
                Id          = 'GPU_NVIDIA_PRESENCE'
                Result      = 'Fail'
                Message     = 'No NVIDIA GPU adapter was detected on this system.'
                Remediation = 'Switch to CPU fallback mode [C].'
            }
            if ($status -ne 'Blocked') {
                $status = 'Adjusted'
                $recOption = 'C'
                $recReason = 'No NVIDIA GPU detected; CPU mode [C] is required.'
            }
        } elseif (-not $Facts.Gpu.HasNvidiaSmi -or -not $Facts.Gpu.HasNvidiaDriver) {
            $checks += [PSCustomObject]@{
                Id          = 'GPU_NVIDIA_DRIVER'
                Result      = 'Fail'
                Message     = 'NVIDIA adapter found, but nvidia-smi or display driver is not installed.'
                Remediation = 'Install the latest official NVIDIA graphics driver from nvidia.com.'
            }
            $status = 'Blocked'
        } else {
            # Check Driver version >= 570
            if ($Facts.Gpu.DriverMajor -lt $Config.Hardware.MinDriverMajor) {
                $checks += [PSCustomObject]@{
                    Id          = 'GPU_DRIVER_VERSION'
                    Result      = 'Fail'
                    Message     = "NVIDIA driver ($($Facts.Gpu.DriverVersion)) is older than minimum required version ($($Config.Hardware.MinDriverMajor).xx for CUDA 12.8)."
                    Remediation = "Update NVIDIA driver to version $($Config.Hardware.MinDriverMajor) or newer from nvidia.com."
                }
                $status = 'Blocked'
            } else {
                $checks += [PSCustomObject]@{
                    Id          = 'GPU_DRIVER_VERSION'
                    Result      = 'Pass'
                    Message     = "NVIDIA driver $($Facts.Gpu.DriverVersion) meets requirement (>= $($Config.Hardware.MinDriverMajor))."
                    Remediation = ''
                }
            }

            # Check Option B (TensorRT-RTX) hardware requirement (Ampere+ / CC >= 8.0)
            if ($Choice -eq 'B') {
                if ($Facts.Gpu.ComputeCapability -lt $Config.Hardware.MinTrtComputeCapability) {
                    $checks += [PSCustomObject]@{
                        Id          = 'GPU_TRT_ARCHITECTURE'
                        Result      = 'Warn'
                        Message     = "GPU $($Facts.Gpu.Name) has Compute Capability $($Facts.Gpu.ComputeCapability). TensorRT-RTX EP requires Ampere or newer (Compute Capability >= $($Config.Hardware.MinTrtComputeCapability)). Turing (RTX 20-series) is unsupported."
                        Remediation = 'Downgrade to Option A (CUDA EP).'
                    }
                    if ($status -ne 'Blocked') {
                        $status = 'Adjusted'
                        $recOption = 'A'
                        $recReason = "GPU compute capability $($Facts.Gpu.ComputeCapability) does not support TensorRT-RTX. Downgrading to CUDA EP [A]."
                    }
                } else {
                    $checks += [PSCustomObject]@{
                        Id          = 'GPU_TRT_ARCHITECTURE'
                        Result      = 'Pass'
                        Message     = "GPU architecture verified for TensorRT-RTX (Compute Capability $($Facts.Gpu.ComputeCapability) >= $($Config.Hardware.MinTrtComputeCapability))."
                        Remediation = ''
                    }
                }
            }
        }
    } else {
        # Option C (CPU) is always valid
        $checks += [PSCustomObject]@{
            Id          = 'CPU_PROFILE'
            Result      = 'Pass'
            Message     = 'CPU profile is valid on any compatible x64 Windows machine.'
            Remediation = ''
        }
    }

    # 6. Plan Invariant Checks
    # Exactly one ORT package in target actions
    $ortTargetCount = 0
    foreach ($act in $Plan.Actions) {
        if ($act.Component -in @('onnxruntime-cpu', 'onnxruntime-gpu') -and $act.Action -in @('Install', 'Keep', 'Replace')) {
            $ortTargetCount++
        }
    }
    if ($ortTargetCount -ne 1) {
        $checks += [PSCustomObject]@{
            Id          = 'PLAN_ORT_INVARIANT'
            Result      = 'Fail'
            Message     = "Plan violates ORT invariant: expected 1 target ORT variant, got $ortTargetCount."
            Remediation = 'Internal Plan generation bug.'
        }
        $status = 'Blocked'
    } else {
        $checks += [PSCustomObject]@{
            Id          = 'PLAN_ORT_INVARIANT'
            Result      = 'Pass'
            Message     = 'Target ONNX Runtime variant invariant holds exactly 1 package.'
            Remediation = ''
        }
    }

    return [PSCustomObject]@{
        Status            = $status
        RecommendedOption = $recOption
        RecommendedReason = $recReason
        Checks            = $checks
    }
}
