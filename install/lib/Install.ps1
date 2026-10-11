# Install.ps1 - Phase 4: State mutation executing the verified installation plan
Set-StrictMode -Version Latest

function Invoke-InstallPlan {
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

    $stepResults = @()
    $root = $Facts.RootDirectory
    $installDir = $Facts.InstallDirectory
    $venvDir = $Facts.Venv.Path
    $venvPython = $Facts.Venv.PythonExe
    $logDir = Join-Path -Path $root -ChildPath 'install_logs'
    $stateFile = Join-Path -Path $installDir -ChildPath $Config.Files.InstallStateJson

    # 1. Snapshot if destructive changes are planned
    $snapshotPath = $null
    if ($Plan.HasDestructive -and (Test-Path -LiteralPath $venvPython)) {
        Write-InstallerLog -Message "Creating environment rollback snapshot..." -Level STEP
        $nowStr = (Get-Date).ToString('yyyyMMdd_HHmmss')
        $snapshotPath = Join-Path -Path $logDir -ChildPath "snapshot_$nowStr.txt"
        $freezeRes = Invoke-Native -FilePath 'uv.exe' -ArgumentList @('pip', 'freeze', '--python', "`"$venvPython`"")
        if ($freezeRes.Success) {
            Write-FileUtf8NoBom -Path $snapshotPath -Content $freezeRes.StdOut
            Write-InstallerLog -Message "Rollback snapshot saved to $snapshotPath" -Level INFO
        }
    }

    # 2. Prerequisites: uv
    $uvAction = $Plan.Actions | Where-Object { $_.Component -eq 'uv' }
    if ($uvAction -and $uvAction.Action -eq 'Install') {
        Write-InstallerLog -Message "Installing uv..." -Level STEP
        $uvInstallScript = Join-Path -Path $env:TEMP -ChildPath "uv_install_$([guid]::NewGuid().ToString('N')).ps1"
        $downloadOk = Invoke-DownloadWithRetry -Uri 'https://astral.sh/uv/install.ps1' -OutFile $uvInstallScript
        if (-not $downloadOk) {
            return [PSCustomObject]@{ Success = $false; ErrorMessage = 'Failed to download uv installer.'; SnapshotPath = $snapshotPath }
        }
        $instRes = Invoke-Native -FilePath 'powershell.exe' -ArgumentList @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', "`"$uvInstallScript`"")
        Remove-Item -LiteralPath $uvInstallScript -Force -ErrorAction SilentlyContinue
        if (-not $instRes.Success) {
            return [PSCustomObject]@{ Success = $false; ErrorMessage = 'Failed to install uv.'; SnapshotPath = $snapshotPath }
        }
        $env:PATH = "$env:USERPROFILE\.local\bin;$env:USERPROFILE\.cargo\bin;$env:PATH"
    }

    # 3. Prerequisites: VC++ Redistributable
    $vcAction = $Plan.Actions | Where-Object { $_.Component -eq 'VCRedist' }
    if ($vcAction -and $vcAction.Action -eq 'Install') {
        Write-InstallerLog -Message "Downloading Visual C++ Redistributable 2015-2022 (x64)..." -Level STEP
        $vcExe = Join-Path -Path $env:TEMP -ChildPath "vc_redist_$([guid]::NewGuid().ToString('N')).exe"
        $downloadOk = Invoke-DownloadWithRetry -Uri $Config.ExternalTools.VcRedistUrl -OutFile $vcExe
        if ($downloadOk) {
            Write-InstallerLog -Message "Installing Visual C++ Redistributable..." -Level INFO
            $vcRes = Invoke-Native -FilePath $vcExe -ArgumentList @('/install', '/quiet', '/norestart')
            Remove-Item -LiteralPath $vcExe -Force -ErrorAction SilentlyContinue
            if ($vcRes.ExitCode -in @(0, 3010)) {
                Write-InstallerLog -Message "Visual C++ Redistributable installed successfully (Exit $($vcRes.ExitCode))." -Level INFO
            } else {
                Write-InstallerLog -Message "Visual C++ Redistributable installer returned code $($vcRes.ExitCode). Continuing." -Level WARN
            }
        } else {
            Write-InstallerLog -Message "Failed to download Visual C++ Redistributable. Some C-modules might fail to load." -Level WARN
        }
    }

    # 4. Prerequisites: FFmpeg
    $ffAction = $Plan.Actions | Where-Object { $_.Component -eq 'FFmpeg' }
    if ($ffAction -and $ffAction.Action -eq 'Install') {
        Write-InstallerLog -Message "Downloading portable FFmpeg..." -Level STEP
        $ffZip = Join-Path -Path $env:TEMP -ChildPath "ffmpeg_$([guid]::NewGuid().ToString('N')).zip"
        $downloadOk = Invoke-DownloadWithRetry -Uri $Config.ExternalTools.FFmpegUrl -OutFile $ffZip
        if ($downloadOk) {
            Write-InstallerLog -Message "Extracting FFmpeg to application root..." -Level INFO
            try {
                Add-Type -AssemblyName System.IO.Compression.FileSystem
                $archive = [System.IO.Compression.ZipFile]::OpenRead($ffZip)
                foreach ($entry in $archive.Entries) {
                    if ($entry.Name -in @('ffmpeg.exe', 'ffprobe.exe')) {
                        $targetPath = Join-Path -Path $root -ChildPath $entry.Name
                        [System.IO.Compression.ZipFileExtensions]::ExtractToFile($entry, $targetPath, $true)
                    }
                }
                $archive.Dispose()
                Write-InstallerLog -Message "FFmpeg binaries installed successfully." -Level INFO
            } catch {
                Write-InstallerLog -Message "Failed to extract FFmpeg: $($_.Exception.Message)" -Level WARN
            } finally {
                Remove-Item -LiteralPath $ffZip -Force -ErrorAction SilentlyContinue
            }
        }
    }

    # 5. Prerequisites: Rubber Band
    $rbAction = $Plan.Actions | Where-Object { $_.Component -eq 'Rubberband' }
    if ($rbAction -and $rbAction.Action -eq 'Install') {
        Write-InstallerLog -Message "Downloading portable Rubber Band..." -Level STEP
        $rbZip = Join-Path -Path $env:TEMP -ChildPath "rubberband_$([guid]::NewGuid().ToString('N')).zip"
        $downloadOk = Invoke-DownloadWithRetry -Uri $Config.ExternalTools.RubberbandUrl -OutFile $rbZip
        if ($downloadOk) {
            Write-InstallerLog -Message "Extracting Rubber Band to application root..." -Level INFO
            try {
                Add-Type -AssemblyName System.IO.Compression.FileSystem
                $archive = [System.IO.Compression.ZipFile]::OpenRead($rbZip)
                foreach ($entry in $archive.Entries) {
                    if ($entry.Name -in @('rubberband.exe', 'sndfile.dll')) {
                        $targetPath = Join-Path -Path $root -ChildPath $entry.Name
                        [System.IO.Compression.ZipFileExtensions]::ExtractToFile($entry, $targetPath, $true)
                    }
                }
                $archive.Dispose()
                Write-InstallerLog -Message "Rubber Band binaries installed successfully." -Level INFO
            } catch {
                Write-InstallerLog -Message "Failed to extract Rubber Band: $($_.Exception.Message)" -Level WARN
            } finally {
                Remove-Item -LiteralPath $rbZip -Force -ErrorAction SilentlyContinue
            }
        }
    }

    # 6. Virtual Environment Provisioning
    $venvAction = $Plan.Actions | Where-Object { $_.Component -eq 'Venv' }
    if ($venvAction -and $venvAction.Action -in @('Create', 'Recreate')) {
        Write-InstallerLog -Message "$($venvAction.Action) virtual environment with Python $($Config.App.PythonVersion)..." -Level STEP
        $venvArgs = @('venv', "`"$venvDir`"", '--python', $Config.App.PythonVersion)
        if ($venvAction.Action -eq 'Recreate') {
            $venvArgs = @('venv', '--clear', "`"$venvDir`"", '--python', $Config.App.PythonVersion)
        }
        $venvRes = Invoke-Native -FilePath 'uv.exe' -ArgumentList $venvArgs
        if (-not $venvRes.Success) {
            Write-InstallerLog -Message "Automatic Python $($Config.App.PythonVersion) not available. Trying any available Python 3.11..." -Level WARN
            $fallbackArgs = @('venv', "`"$venvDir`"", '--python', '3.11')
            $venvRes = Invoke-Native -FilePath 'uv.exe' -ArgumentList $fallbackArgs
            if (-not $venvRes.Success) {
                return [PSCustomObject]@{ Success = $false; ErrorMessage = "Failed to create virtual environment: $($venvRes.StdErr)"; SnapshotPath = $snapshotPath }
            }
        }
        Write-InstallerLog -Message "Virtual environment provisioned at $venvDir" -Level INFO
    }

    # 7. Uninstall Conflicting Components
    $removeActions = $Plan.Actions | Where-Object { $_.Action -in @('Remove', 'Replace') }
    foreach ($rem in $removeActions) {
        switch ($rem.Component) {
            'PyTorch' {
                Write-InstallerLog -Message "Uninstalling conflicting PyTorch ecosystem builds..." -Level STEP
                Invoke-Native -FilePath 'uv.exe' -ArgumentList @('pip', 'uninstall', '--python', "`"$venvPython`"", 'torch', 'torchvision', 'torchaudio') | Out-Null
            }
            'onnxruntime-cpu' {
                Write-InstallerLog -Message "Uninstalling CPU onnxruntime..." -Level STEP
                Invoke-Native -FilePath 'uv.exe' -ArgumentList @('pip', 'uninstall', '--python', "`"$venvPython`"", 'onnxruntime') | Out-Null
            }
            'onnxruntime-gpu' {
                Write-InstallerLog -Message "Uninstalling GPU onnxruntime..." -Level STEP
                Invoke-Native -FilePath 'uv.exe' -ArgumentList @('pip', 'uninstall', '--python', "`"$venvPython`"", 'onnxruntime-gpu') | Out-Null
            }
            'TRT-RTX-Plugin' {
                Write-InstallerLog -Message "Uninstalling TensorRT-RTX plugin..." -Level STEP
                Invoke-Native -FilePath 'uv.exe' -ArgumentList @('pip', 'uninstall', '--python', "`"$venvPython`"", 'onnxruntime-ep-nv-tensorrt-rtx-cu12') | Out-Null
            }
        }
    }

    # 8. Install Target PyTorch
    $torchAction = $Plan.Actions | Where-Object { $_.Component -eq 'PyTorch' }
    if ($torchAction -and $torchAction.Action -in @('Install', 'Replace')) {
        Write-InstallerLog -Message "Installing PyTorch ($($Plan.TargetTorchVariant)) from $($Plan.TargetTorchIndex)..." -Level STEP
        $torchInstallArgs = @(
            'pip', 'install', '--python', "`"$venvPython`"",
            '--reinstall-package', 'torch',
            '--reinstall-package', 'torchvision',
            '--reinstall-package', 'torchaudio'
        ) + $Config.Torch.Packages + @('--index-url', $Plan.TargetTorchIndex)

        $torchRes = Invoke-Native -FilePath 'uv.exe' -ArgumentList $torchInstallArgs -TimeoutSeconds 600
        if (-not $torchRes.Success) {
            return [PSCustomObject]@{ Success = $false; ErrorMessage = "Failed to install PyTorch: $($torchRes.StdErr)"; SnapshotPath = $snapshotPath }
        }
        Write-InstallerLog -Message "PyTorch ($($Plan.TargetTorchVariant)) installed successfully." -Level INFO
    }

    # 9. Install Target ONNX Runtime & Plugins
    if ($Plan.Profile -eq 'A') {
        $ortAction = $Plan.Actions | Where-Object { $_.Component -eq 'onnxruntime-gpu' }
        if ($ortAction -and $ortAction.Action -in @('Install', 'Replace')) {
            Write-InstallerLog -Message "Installing onnxruntime-gpu $($Config.Ort.CudaVersion)..." -Level STEP
            $ortRes = Invoke-Native -FilePath 'uv.exe' -ArgumentList @('pip', 'install', '--python', "`"$venvPython`"", '--reinstall-package', 'onnxruntime-gpu', "onnxruntime-gpu==$($Config.Ort.CudaVersion)")
            if (-not $ortRes.Success) {
                Write-InstallerLog -Message "Failed to install onnxruntime-gpu: $($ortRes.StdErr)" -Level WARN
            }
        }
    } elseif ($Plan.Profile -eq 'B') {
        Write-InstallerLog -Message "Installing ONNX Runtime GPU 1.24.4 and TensorRT-RTX EP plugin..." -Level STEP
        $ortTrtArgs = @(
            'pip', 'install', '--python', "`"$venvPython`"",
            '--reinstall-package', 'onnxruntime-gpu',
            '--reinstall-package', 'onnxruntime-ep-nv-tensorrt-rtx-cu12',
            "onnxruntime-gpu==$($Config.Ort.TrtGpuVersion)",
            $Config.Ort.TrtPluginPkg
        )
        $ortRes = Invoke-Native -FilePath 'uv.exe' -ArgumentList $ortTrtArgs
        if (-not $ortRes.Success) {
            Write-InstallerLog -Message "Failed to install TensorRT-RTX packages: $($ortRes.StdErr). CUDA EP will remain as fallback." -Level WARN
        }
    } else {
        # Profile C
        $ortAction = $Plan.Actions | Where-Object { $_.Component -eq 'onnxruntime-cpu' }
        if ($ortAction -and $ortAction.Action -in @('Install', 'Replace')) {
            Write-InstallerLog -Message "Installing onnxruntime CPU $($Config.Ort.CpuVersion)..." -Level STEP
            $ortRes = Invoke-Native -FilePath 'uv.exe' -ArgumentList @('pip', 'install', '--python', "`"$venvPython`"", '--reinstall-package', 'onnxruntime', "onnxruntime==$($Config.Ort.CpuVersion)")
            if (-not $ortRes.Success) {
                Write-InstallerLog -Message "Failed to install onnxruntime CPU: $($ortRes.StdErr)" -Level WARN
            }
        }
    }

    # 10. Install Pre-built Wheels and Requirements
    Write-InstallerLog -Message "Installing base dependencies from $($Plan.RequirementsFile)..." -Level STEP
    $wheelsDir = Join-Path -Path $installDir -ChildPath 'wheels'
    $diffqWheel = Join-Path -Path $wheelsDir -ChildPath 'diffq-0.2.4-cp311-cp311-win_amd64.whl'
    if (Test-Path -LiteralPath $diffqWheel) {
        Write-InstallerLog -Message "Pre-installing local binary wheel for diffq..." -Level DEBUG
        Invoke-Native -FilePath 'uv.exe' -ArgumentList @('pip', 'install', '--python', "`"$venvPython`"", '--no-deps', "`"$diffqWheel`"") | Out-Null
    }

    $reqFilePath = Join-Path -Path $installDir -ChildPath $Plan.RequirementsFile
    if (-not (Test-Path -LiteralPath $reqFilePath)) {
        $candidateReqs = Join-Path -Path (Join-Path -Path $installDir -ChildPath 'reqs') -ChildPath $Plan.RequirementsFile
        if (Test-Path -LiteralPath $candidateReqs) {
            $reqFilePath = $candidateReqs
        } else {
            $reqFilePath = Join-Path -Path $root -ChildPath $Plan.RequirementsFile
            if (-not (Test-Path -LiteralPath $reqFilePath)) {
                $candidateRootReqs = Join-Path -Path (Join-Path -Path $root -ChildPath 'reqs') -ChildPath $Plan.RequirementsFile
                if (Test-Path -LiteralPath $candidateRootReqs) {
                    $reqFilePath = $candidateRootReqs
                }
            }
        }
    }

    $reqInstallFile = $reqFilePath
    $tempFilteredReq = $null

    # If Profile C, filter out GPU packages to match .bat behavior
    if ($Plan.Profile -eq 'C') {
        $tempFilteredReq = Join-Path -Path $env:TEMP -ChildPath "req_cpu_$([guid]::NewGuid().ToString('N')).txt"
        $rawReqLines = Get-Content -LiteralPath $reqFilePath -ErrorAction SilentlyContinue
        $filteredLines = @()
        foreach ($line in $rawReqLines) {
            $skip = $false
            if ($line -match '^(torch==|torchvision==|torchaudio==)') { $skip = $true }
            if ($line -match '^(cupy-cuda|cuda-pathfinder|onnxruntime-gpu|nvidia-)') { $skip = $true }
            if ($line -match '^(polygraphy|nvidia-modelopt|onnx-graphsurgeon)') { $skip = $true }
            if (-not $skip -and -not [string]::IsNullOrWhiteSpace($line)) {
                $filteredLines += $line
            }
        }
        Write-FileUtf8NoBom -Path $tempFilteredReq -Content ($filteredLines -join "`r`n")
        $reqInstallFile = $tempFilteredReq
    }

    $reqArgs = @('pip', 'install', '--python', "`"$venvPython`"")
    if (Test-Path -LiteralPath $wheelsDir) {
        $reqArgs += @('--find-links', "`"$wheelsDir`"")
    }
    $reqArgs += @('-r', "`"$reqInstallFile`"")

    $reqRes = Invoke-Native -FilePath 'uv.exe' -ArgumentList $reqArgs -TimeoutSeconds 600

    if ($tempFilteredReq -and (Test-Path -LiteralPath $tempFilteredReq)) {
        Remove-Item -LiteralPath $tempFilteredReq -Force -ErrorAction SilentlyContinue
    }

    if (-not $reqRes.Success) {
        return [PSCustomObject]@{ Success = $false; ErrorMessage = "Failed to install base dependencies: $($reqRes.StdErr)"; SnapshotPath = $snapshotPath }
    }
    Write-InstallerLog -Message "Base dependencies installed successfully." -Level INFO

    return [PSCustomObject]@{
        Success      = $true
        ErrorMessage = ''
        SnapshotPath = $snapshotPath
    }
}
