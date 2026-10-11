# Validate.ps1 - Phase 5: Deep runtime verification of PyTorch, ORT, and external tools
Set-StrictMode -Version Latest

# Minimal valid ONNX model (Identity graph, opset 17, IR version 8, float32 [1, 2] tensor) encoded in Base64
# Compatible across all ONNX Runtime versions including 1.22.0 (max IR 10) and 1.24.4+ (max IR 13)
$script:MinimalOnnxBase64 = 'CAg6QgoQCgFYEgFZIghJZGVudGl0eRIEdGVzdFoTCgFYEg4KDAgBEggKAggBCgIIAmITCgFZEg4KDAgBEggKAggBCgIIAkIECgAQEQ=='

function Test-InstalledStack {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)]
        [PSCustomObject]$Plan,

        [Parameter(Mandatory = $true)]
        [PSCustomObject]$Facts,

        [Parameter(Mandatory = $false)]
        [PSCustomObject]$Config = $null
    )

    if ($null -eq $Config) {
        $Config = Get-InstallerConfig
    }

    Write-InstallerLog -Message "Starting end-to-end runtime validation..." -Level STEP

    $root = $Facts.RootDirectory
    $venvPython = $Facts.Venv.PythonExe
    $checks = @()
    $effectiveOption = $Plan.Profile
    $fallbackReason = ''
    $validationPassed = $true

    # 1. External Media Binaries Runtime Test
    # FFmpeg
    $ffmpegPath = Join-Path -Path $root -ChildPath 'ffmpeg.exe'
    if (-not (Test-Path -LiteralPath $ffmpegPath)) {
        $sysFf = Get-Command 'ffmpeg.exe' -ErrorAction SilentlyContinue
        if ($sysFf) { $ffmpegPath = $sysFf.Source }
    }
    if (Test-Path -LiteralPath $ffmpegPath) {
        $ffRes = Invoke-Native -FilePath $ffmpegPath -ArgumentList @('-version') -TimeoutSeconds 5
        if ($ffRes.Success) {
            $checks += [PSCustomObject]@{ Id = 'TOOL_FFMPEG'; Result = 'Pass'; Message = 'FFmpeg executable runs successfully.' }
        } else {
            $checks += [PSCustomObject]@{ Id = 'TOOL_FFMPEG'; Result = 'Warn'; Message = "FFmpeg execution failed: $($ffRes.StdErr)" }
        }
    } else {
        $checks += [PSCustomObject]@{ Id = 'TOOL_FFMPEG'; Result = 'Warn'; Message = 'FFmpeg executable not found.' }
    }

    # Rubber Band
    $rbPath = Join-Path -Path $root -ChildPath 'rubberband.exe'
    if (-not (Test-Path -LiteralPath $rbPath)) {
        $sysRb = Get-Command 'rubberband.exe' -ErrorAction SilentlyContinue
        if (-not $sysRb) { $sysRb = Get-Command 'rubberband-cli.exe' -ErrorAction SilentlyContinue }
        if ($sysRb) { $rbPath = $sysRb.Source }
    }
    if (Test-Path -LiteralPath $rbPath) {
        $rbRes = Invoke-Native -FilePath $rbPath -ArgumentList @('--version') -TimeoutSeconds 5
        if ($rbRes.Success -or $rbRes.ExitCode -eq 0) {
            $checks += [PSCustomObject]@{ Id = 'TOOL_RUBBERBAND'; Result = 'Pass'; Message = 'Rubber Band executable runs successfully.' }
        } else {
            $checks += [PSCustomObject]@{ Id = 'TOOL_RUBBERBAND'; Result = 'Warn'; Message = "Rubber Band execution failed: $($rbRes.StdErr)" }
        }
    } else {
        $checks += [PSCustomObject]@{ Id = 'TOOL_RUBBERBAND'; Result = 'Warn'; Message = 'Rubber Band executable not found.' }
    }

    # 2. PyTorch Runtime Verification
    Write-InstallerLog -Message "Validating PyTorch runtime..." -Level INFO
    $torchScript = @"
import sys
try:
    import torch
    is_cuda_req = ('$($Plan.TargetTorchVariant)' == 'CUDA')
    if is_cuda_req:
        if not torch.cuda.is_available():
            print('ERR: torch.cuda.is_available() is False')
            sys.exit(2)
        dev_name = torch.cuda.get_device_name(0)
        t = torch.zeros(1, device='cuda') + 1.0
        if t.item() != 1.0:
            print('ERR: Tensor calculation on GPU failed')
            sys.exit(3)
        print(f'OK: PyTorch CUDA operational on {dev_name}')
        sys.exit(0)
    else:
        t = torch.zeros(1) + 1.0
        if t.item() != 1.0:
            sys.exit(3)
        print('OK: PyTorch CPU operational')
        sys.exit(0)
except Exception as e:
    print(f'ERR: {e}')
    sys.exit(1)
"@

    $torchRes = Invoke-Native -FilePath $venvPython -ArgumentList @('-c', "`"$torchScript`"") -TimeoutSeconds 30
    if ($torchRes.Success) {
        $checks += [PSCustomObject]@{ Id = 'PYTORCH_RUNTIME'; Result = 'Pass'; Message = $torchRes.StdOut.Trim() }
        Write-InstallerLog -Message "PyTorch runtime verification passed." -Level INFO
    } else {
        $checks += [PSCustomObject]@{ Id = 'PYTORCH_RUNTIME'; Result = 'Fail'; Message = "PyTorch runtime failed: $($torchRes.StdOut) $($torchRes.StdErr)" }
        Write-InstallerLog -Message "PyTorch runtime verification failed." -Level ERROR
        $validationPassed = $false
    }

    # 3. ONNX Runtime & Execution Provider Verification
    Write-InstallerLog -Message "Validating ONNX Runtime Execution Provider..." -Level INFO
    $ortProfile = $Plan.Profile

    $ortScript = @"
import sys, base64
import numpy as np

try:
    import torch
    import onnxruntime as ort
    from engine.tensorrt_rtx import register_trt_rtx_libraries, is_trt_rtx_available

    register_trt_rtx_libraries()

    profile = '$ortProfile'
    b64_model = '$script:MinimalOnnxBase64'
    model_bytes = base64.b64decode(b64_model)

    avail = ort.get_available_providers()
    print('ORT Available providers:', avail)

    if profile == 'B':
        # Try TRT-RTX first via application session factory
        trt_ok = False
        if is_trt_rtx_available(0):
            try:
                from engine.tensorrt_rtx.session import create_trt_rtx_session
                sess = create_trt_rtx_session(model_bytes)
                out = sess.run(None, {'X': np.array([[1.0, 2.0]], dtype=np.float32)})
                active = sess.get_providers()[0] if hasattr(sess, 'get_providers') else ''
                if active == 'CUDAExecutionProvider':
                    print('CUDA_FALLBACK_OK: Fallback to CUDAExecutionProvider succeeded')
                    sys.exit(10)
                print('TRT_RTX_OK: Session created and evaluated successfully on', active)
                trt_ok = True
                sys.exit(0)
            except Exception as trt_err:
                print('TRT_RTX_FAIL:', trt_err)
        if not trt_ok:
            # Fallback to CUDA
            sess_cuda = ort.InferenceSession(model_bytes, providers=['CUDAExecutionProvider'])
            out_cuda = sess_cuda.run(None, {'X': np.array([[1.0, 2.0]], dtype=np.float32)})
            print('CUDA_FALLBACK_OK: Fallback to CUDAExecutionProvider succeeded')
            sys.exit(10) # Code 10 indicates TRT-RTX failed but CUDA fallback worked
        sys.exit(0)

    elif profile == 'A':
        sess = ort.InferenceSession(model_bytes, providers=['CUDAExecutionProvider'])
        out = sess.run(None, {'X': np.array([[1.0, 2.0]], dtype=np.float32)})
        print('CUDA_OK: Session created and evaluated successfully')
        sys.exit(0)

    else:
        # Profile C
        sess = ort.InferenceSession(model_bytes, providers=['CPUExecutionProvider'])
        out = sess.run(None, {'X': np.array([[1.0, 2.0]], dtype=np.float32)})
        print('CPU_OK: Session created and evaluated successfully')
        sys.exit(0)

except Exception as exc:
    print('ORT_FATAL:', exc)
    sys.exit(1)
"@

    $ortRes = Invoke-Native -FilePath $venvPython -ArgumentList @('-c', "`"$ortScript`"") -TimeoutSeconds 45 -WorkingDirectory $root
    if ($ortRes.ExitCode -eq 0) {
        $checks += [PSCustomObject]@{ Id = 'ORT_RUNTIME'; Result = 'Pass'; Message = $ortRes.StdOut.Trim() }
        Write-InstallerLog -Message "ONNX Runtime verification passed." -Level INFO
    } elseif ($ortRes.ExitCode -eq 10 -and $Plan.Profile -eq 'B') {
        # Fallback ladder step 1: B runtime failed -> downgrade to A
        Write-InstallerLog -Message "TensorRT-RTX EP runtime test failed. Gracefully falling back to stable CUDA EP [A]." -Level WARN
        $checks += [PSCustomObject]@{ Id = 'ORT_RUNTIME'; Result = 'Warn'; Message = 'TensorRT-RTX session creation failed; CUDA EP fallback verified.' }
        $effectiveOption = 'A'
        $fallbackReason = 'TensorRT-RTX session creation failed during verification; CUDA EP fallback active.'
    } else {
        $checks += [PSCustomObject]@{ Id = 'ORT_RUNTIME'; Result = 'Fail'; Message = "ONNX Runtime validation failed: $($ortRes.StdOut) $($ortRes.StdErr)" }
        Write-InstallerLog -Message "ONNX Runtime verification failed." -Level ERROR
        $validationPassed = $false
    }

    # 4. Smoke Import of Application Core Modules
    Write-InstallerLog -Message "Running smoke import test for application core modules..." -Level INFO
    $smokeScript = "import sys; import gui_data.constants; import separate; print('UVR_CORE_IMPORTS_OK')"
    $smokeRes = Invoke-Native -FilePath $venvPython -ArgumentList @('-c', "`"$smokeScript`"") -TimeoutSeconds 20 -WorkingDirectory $root
    if ($smokeRes.Success) {
        $checks += [PSCustomObject]@{ Id = 'UVR_CORE_IMPORTS'; Result = 'Pass'; Message = 'Core application modules imported successfully.' }
        Write-InstallerLog -Message "Core application modules imported successfully." -Level INFO
    } else {
        $checks += [PSCustomObject]@{ Id = 'UVR_CORE_IMPORTS'; Result = 'Warn'; Message = "Warning during core module imports: $($smokeRes.StdErr)" }
    }

    return [PSCustomObject]@{
        Passed          = $validationPassed
        EffectiveOption = $effectiveOption
        FallbackReason  = $fallbackReason
        Checks          = $checks
    }
}
