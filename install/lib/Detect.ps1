# Detect.ps1 - Phase 1: Pure inspection gathering real machine facts
Set-StrictMode -Version Latest

function Get-InstallFacts {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)]
        [string]$RootDirectory,

        [Parameter(Mandatory = $true)]
        [PSCustomObject]$Config
    )

    Write-InstallerLog -Message "Gathering system and environment facts..." -Level INFO

    $installDir = Join-Path -Path $RootDirectory -ChildPath 'install'
    $venvDir = Join-Path -Path $RootDirectory -ChildPath 'venv'
    $venvPython = Join-Path -Path $venvDir -ChildPath 'Scripts\python.exe'

    # 1. OS Facts
    $osArch = [System.Environment]::GetEnvironmentVariable('PROCESSOR_ARCHITECTURE')
    $is64Bit = [System.Environment]::Is64BitOperatingSystem
    $isArm64 = ($osArch -eq 'ARM64')
    $psVersion = $PSVersionTable.PSVersion.ToString()
    
    $isAdmin = $false
    try {
        $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
        $principal = New-Object Security.Principal.WindowsPrincipal($identity)
        $isAdmin = $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
    } catch {
        $isAdmin = $false
    }

    $osFacts = [PSCustomObject]@{
        VersionString = [System.Environment]::OSVersion.VersionString
        IsWindows     = ($env:OS -eq 'Windows_NT')
        Is64Bit       = $is64Bit
        IsArm64       = $isArm64
        Architecture  = $osArch
        PSVersion     = $psVersion
        IsAdmin       = $isAdmin
    }

    # 2. GPU Facts
    $allAdapters = @()
    $hasAmd = $false
    $hasIntel = $false
    $hasNvidiaAdapter = $false
    $amdName = ''
    $intelName = ''

    try {
        $adapters = Get-CimInstance Win32_VideoController -ErrorAction SilentlyContinue
        if ($adapters) {
            foreach ($ad in $adapters) {
                $name = [string]$ad.Name
                $drv = [string]$ad.DriverVersion
                $ram = 0
                if ($ad.AdapterRAM) {
                    $ram = [math]::Round([double]$ad.AdapterRAM / 1MB)
                }

                $adapterObj = [PSCustomObject]@{
                    Name          = $name
                    DriverVersion = $drv
                    VramMB        = $ram
                }
                $allAdapters += $adapterObj

                if ($name -match 'AMD|Radeon') {
                    $hasAmd = $true
                    $amdName = $name
                }
                if ($name -match 'Intel') {
                    $hasIntel = $true
                    $intelName = $name
                }
                if ($name -match 'NVIDIA') {
                    $hasNvidiaAdapter = $true
                }
            }
        }
    } catch {
        Write-InstallerLog -Message "Could not query Win32_VideoController: $($_.Exception.Message)" -Level WARN
    }

    # Query nvidia-smi if available
    $nvidiaSmiPath = $null
    $hasNvidiaSmi = $false
    $nvidiaName = 'Unknown'
    $nvidiaDriver = 'Unknown'
    $nvidiaDriverMajor = 0
    $nvidiaComputeCap = 0.0
    $nvidiaVramMB = 0
    $hasNvidiaDriver = $false

    $smiCmd = Get-Command 'nvidia-smi.exe' -ErrorAction SilentlyContinue
    if ($smiCmd) {
        $nvidiaSmiPath = $smiCmd.Source
        $hasNvidiaSmi = $true
    } elseif (Test-Path -LiteralPath 'C:\Windows\System32\nvidia-smi.exe') {
        $nvidiaSmiPath = 'C:\Windows\System32\nvidia-smi.exe'
        $hasNvidiaSmi = $true
    }

    if ($hasNvidiaSmi) {
        $smiRes = Invoke-Native -FilePath $nvidiaSmiPath -ArgumentList @('--query-gpu=name,driver_version,compute_cap,memory.total', '--format=csv,noheader,nounits') -TimeoutSeconds 10
        if ($smiRes.Success -and (-not [string]::IsNullOrWhiteSpace($smiRes.StdOut))) {
            $lines = @($smiRes.StdOut -split "`r?`n" | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
            if ($lines.Count -gt 0) {
                $primaryGpu = $lines[0] -split ','
                if ($primaryGpu.Count -ge 2) {
                    $nvidiaName = $primaryGpu[0].Trim()
                    $nvidiaDriver = $primaryGpu[1].Trim()
                    $hasNvidiaDriver = $true

                    $driverParts = $nvidiaDriver -split '\.'
                    if ($driverParts.Count -ge 1) {
                        [int]::TryParse($driverParts[0], [ref]$nvidiaDriverMajor) | Out-Null
                    }
                }
                if ($primaryGpu.Count -ge 3) {
                    $ccStr = $primaryGpu[2].Trim()
                    [double]::TryParse($ccStr, [System.Globalization.NumberStyles]::Any, [System.Globalization.CultureInfo]::InvariantCulture, [ref]$nvidiaComputeCap) | Out-Null
                }
                if ($primaryGpu.Count -ge 4) {
                    $vramStr = $primaryGpu[3].Trim()
                    [int]::TryParse($vramStr, [ref]$nvidiaVramMB) | Out-Null
                }
            }
        }
    }

    $isRtxSupported = ($nvidiaName -match 'RTX.*20|RTX.*30|RTX.*40|RTX.*50|RTX A|RTX Ada|RTX PRO')

    $gpuFacts = [PSCustomObject]@{
        AllAdapters        = $allAdapters
        HasAmd             = $hasAmd
        AmdName            = $amdName
        HasIntel           = $hasIntel
        IntelName          = $intelName
        HasNvidiaAdapter   = $hasNvidiaAdapter
        HasNvidiaSmi       = $hasNvidiaSmi
        HasNvidiaDriver    = $hasNvidiaDriver
        Name               = $nvidiaName
        DriverVersion      = $nvidiaDriver
        DriverMajor        = $nvidiaDriverMajor
        ComputeCapability  = $nvidiaComputeCap
        VramMB             = $nvidiaVramMB
        IsRtxSupported     = $isRtxSupported
    }

    # 3. Venv Facts
    $venvExists = (Test-Path -LiteralPath $venvPython)
    $venvPythonVer = 'Unknown'
    $venvIs64Bit = $true

    if ($venvExists) {
        $pyVerRes = Invoke-Native -FilePath $venvPython -ArgumentList @('-c', "`"import sys; print(sys.version.split()[0]); print(sys.maxsize > 2**32)`"") -TimeoutSeconds 5
        if ($pyVerRes.Success) {
            $pyLines = @($pyVerRes.StdOut -split "`r?`n" | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
            if ($pyLines.Count -ge 1) { $venvPythonVer = $pyLines[0].Trim() }
            if ($pyLines.Count -ge 2) { $venvIs64Bit = ($pyLines[1].Trim() -eq 'True') }
        }
    }

    $venvFacts = [PSCustomObject]@{
        Exists        = $venvExists
        Path          = $venvDir
        PythonExe     = $venvPython
        PythonVersion = $venvPythonVer
        Is64Bit       = $venvIs64Bit
    }

    # 4. Package Facts (inspecting venv site-packages dist-info without importing torch)
    $sitePackagesDir = Join-Path -Path $venvDir -ChildPath 'Lib\site-packages'
    $pkgFacts = [PSCustomObject]@{
        TorchInstalled       = $false
        TorchVersion         = 'None'
        TorchVariant         = 'None' # CUDA, CPU, or UNKNOWN
        TorchLocalTag        = ''     # cu128, cpu, or none
        TorchvisionInstalled = $false
        TorchvisionVersion   = 'None'
        TorchvisionVariant   = 'None'
        TorchaudioInstalled  = $false
        TorchaudioVersion    = 'None'
        TorchaudioVariant    = 'None'
        HasOrtCpu            = $false
        OrtCpuVersion        = 'None'
        HasOrtGpu            = $false
        OrtGpuVersion        = 'None'
        HasOrtConflict       = $false
        HasTrtPlugin         = $false
        TrtPluginVersion     = 'None'
    }

    if (Test-Path -LiteralPath $sitePackagesDir) {
        $distInfos = Get-ChildItem -LiteralPath $sitePackagesDir -Directory -Filter '*.dist-info' -ErrorAction SilentlyContinue

        foreach ($di in $distInfos) {
            $folderName = $di.Name
            
            # Helper to inspect METADATA Version field
            $metaPath = Join-Path -Path $di.FullName -ChildPath 'METADATA'
            $versionInMeta = $null
            if (Test-Path -LiteralPath $metaPath) {
                $verLine = Get-Content -LiteralPath $metaPath -TotalCount 20 -ErrorAction SilentlyContinue | Where-Object { $_ -match '^Version:\s*(.+)$' }
                if ($verLine) {
                    $versionInMeta = $Matches[1].Trim()
                }
            }

            # Inspect torch
            if ($folderName -match '^torch-([0-9\.]+.*?)\.dist-info$') {
                $rawVer = if ($versionInMeta) { $versionInMeta } else { $Matches[1] }
                $pkgFacts.TorchInstalled = $true
                $pkgFacts.TorchVersion = $rawVer
                if ($rawVer -match '\+cu([0-9]+)') {
                    $pkgFacts.TorchVariant = 'CUDA'
                    $pkgFacts.TorchLocalTag = "cu$($Matches[1])"
                } elseif ($rawVer -match '\+cpu') {
                    $pkgFacts.TorchVariant = 'CPU'
                    $pkgFacts.TorchLocalTag = 'cpu'
                } else {
                    $pkgFacts.TorchVariant = 'CPU' # PyPI plain wheel without tag is CPU
                    $pkgFacts.TorchLocalTag = 'none'
                }
            }

            # Inspect torchvision
            if ($folderName -match '^torchvision-([0-9\.]+.*?)\.dist-info$') {
                $rawVer = if ($versionInMeta) { $versionInMeta } else { $Matches[1] }
                $pkgFacts.TorchvisionInstalled = $true
                $pkgFacts.TorchvisionVersion = $rawVer
                if ($rawVer -match '\+cu') {
                    $pkgFacts.TorchvisionVariant = 'CUDA'
                } else {
                    $pkgFacts.TorchvisionVariant = 'CPU'
                }
            }

            # Inspect torchaudio
            if ($folderName -match '^torchaudio-([0-9\.]+.*?)\.dist-info$') {
                $rawVer = if ($versionInMeta) { $versionInMeta } else { $Matches[1] }
                $pkgFacts.TorchaudioInstalled = $true
                $pkgFacts.TorchaudioVersion = $rawVer
                if ($rawVer -match '\+cu') {
                    $pkgFacts.TorchaudioVariant = 'CUDA'
                } else {
                    $pkgFacts.TorchaudioVariant = 'CPU'
                }
            }

            # Inspect onnxruntime (CPU)
            if ($folderName -match '^onnxruntime-([0-9\.]+.*?)\.dist-info$' -and $folderName -notmatch 'gpu' -and $folderName -notmatch 'ep') {
                $pkgFacts.HasOrtCpu = $true
                $pkgFacts.OrtCpuVersion = if ($versionInMeta) { $versionInMeta } else { $Matches[1] }
            }

            # Inspect onnxruntime-gpu
            if ($folderName -match '^onnxruntime_gpu-([0-9\.]+.*?)\.dist-info$') {
                $pkgFacts.HasOrtGpu = $true
                $pkgFacts.OrtGpuVersion = if ($versionInMeta) { $versionInMeta } else { $Matches[1] }
            }

            # Inspect TRT-RTX plugin
            if ($folderName -match '^onnxruntime_ep_nv_tensorrt_rtx_cu12-([0-9\.]+.*?)\.dist-info$') {
                $pkgFacts.HasTrtPlugin = $true
                $pkgFacts.TrtPluginVersion = if ($versionInMeta) { $versionInMeta } else { $Matches[1] }
            }
        }

        $pkgFacts.HasOrtConflict = ($pkgFacts.HasOrtCpu -and $pkgFacts.HasOrtGpu)
    }

    # 5. External Tools Facts
    # uv
    $uvInstalled = $false
    $uvVersion = 'None'
    $uvCmd = Get-Command 'uv.exe' -ErrorAction SilentlyContinue
    if ($uvCmd) {
        $uvRes = Invoke-Native -FilePath $uvCmd.Source -ArgumentList @('--version') -TimeoutSeconds 5
        if ($uvRes.Success) {
            $uvInstalled = $true
            $uvVersion = $uvRes.StdOut.Trim()
        }
    }

    # ffmpeg
    $ffmpegInstalled = $false
    $ffmpegPath = ''
    $ffmpegSource = 'None'
    $rootFfmpeg = Join-Path -Path $RootDirectory -ChildPath 'ffmpeg.exe'
    if (Test-Path -LiteralPath $rootFfmpeg) {
        $ffmpegInstalled = $true
        $ffmpegPath = $rootFfmpeg
        $ffmpegSource = 'Root'
    } else {
        $sysFfmpeg = Get-Command 'ffmpeg.exe' -ErrorAction SilentlyContinue
        if ($sysFfmpeg) {
            $ffmpegInstalled = $true
            $ffmpegPath = $sysFfmpeg.Source
            $ffmpegSource = 'SystemPath'
        }
    }

    # rubberband
    $rbInstalled = $false
    $rbPath = ''
    $rbSource = 'None'
    $rootRb = Join-Path -Path $RootDirectory -ChildPath 'rubberband.exe'
    if (Test-Path -LiteralPath $rootRb) {
        $rbInstalled = $true
        $rbPath = $rootRb
        $rbSource = 'Root'
    } else {
        $sysRb = Get-Command 'rubberband.exe' -ErrorAction SilentlyContinue
        if (-not $sysRb) {
            $sysRb = Get-Command 'rubberband-cli.exe' -ErrorAction SilentlyContinue
        }
        if ($sysRb) {
            $rbInstalled = $true
            $rbPath = $sysRb.Source
            $rbSource = 'SystemPath'
        }
    }

    # Visual C++ Redistributable
    $vcOk = $false
    $vcVersion = 'None'
    $vcMajor = 0
    $vcMinor = 0

    try {
        $vcKey = 'HKLM:\SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\X64'
        if (Test-Path -LiteralPath $vcKey) {
            $props = Get-ItemProperty -LiteralPath $vcKey -ErrorAction SilentlyContinue
            if ($props -and $props.Version) {
                $vcVersion = [string]$props.Version -replace '^v', ''
                $parts = $vcVersion -split '\.'
                if ($parts.Count -ge 2) {
                    [int]::TryParse($parts[0], [ref]$vcMajor) | Out-Null
                    [int]::TryParse($parts[1], [ref]$vcMinor) | Out-Null
                    if ($vcMajor -gt $Config.ExternalTools.VcRedistMinMajor) {
                        $vcOk = $true
                    } elseif ($vcMajor -eq $Config.ExternalTools.VcRedistMinMajor -and $vcMinor -ge $Config.ExternalTools.VcRedistMinMinor) {
                        $vcOk = $true
                    }
                }
            } elseif ($props -and $props.Installed -eq 1 -and $props.Minor -ge $Config.ExternalTools.VcRedistMinMinor) {
                $vcOk = $true
                $vcMajor = 14
                $vcMinor = [int]$props.Minor
            }
        }
    } catch { }

    $toolFacts = [PSCustomObject]@{
        UvInstalled       = $uvInstalled
        UvVersion         = $uvVersion
        FFmpegInstalled   = $ffmpegInstalled
        FFmpegPath        = $ffmpegPath
        FFmpegSource      = $ffmpegSource
        RubberbandInstalled = $rbInstalled
        RubberbandPath    = $rbPath
        RubberbandSource  = $rbSource
        VcRedistOk        = $vcOk
        VcRedistVersion   = $vcVersion
        VcRedistMajor     = $vcMajor
        VcRedistMinor     = $vcMinor
    }

    # 6. Resources & Connectivity Facts
    $installDrive = (Get-Item -LiteralPath $RootDirectory).PSDrive.Name
    $rootDriveObj = Get-PSDrive -Name $installDrive -ErrorAction SilentlyContinue
    $freeSpaceRoot = if ($rootDriveObj) { $rootDriveObj.Free } else { 0 }

    $tempPath = [System.IO.Path]::GetTempPath()
    $tempDriveName = (Get-Item -LiteralPath $tempPath).PSDrive.Name
    $tempDriveObj = Get-PSDrive -Name $tempDriveName -ErrorAction SilentlyContinue
    $freeSpaceTemp = if ($tempDriveObj) { $tempDriveObj.Free } else { 0 }

    $connectivity = @{}
    foreach ($h in $Config.Network.HostsToCheck) {
        $connected = $false
        try {
            $req = [System.Net.WebRequest]::Create($h)
            $req.Method = 'HEAD'
            $req.Timeout = 5000
            $resp = $req.GetResponse()
            $connected = $true
            $resp.Close()
        } catch {
            $connected = $false
        }
        $connectivity[$h] = $connected
    }

    $resourceFacts = [PSCustomObject]@{
        FreeDiskSpaceBytes = $freeSpaceRoot
        FreeTempSpaceBytes = $freeSpaceTemp
        Connectivity       = $connectivity
    }

    # 7. History Facts
    $recordFile = Join-Path -Path $installDir -ChildPath $Config.Files.InstallRecordJson
    $hasRecord = $false
    $prevOption = 'None'
    $prevMode = 'None'
    $recordData = $null

    if (Test-Path -LiteralPath $recordFile) {
        try {
            $jsonRaw = Get-Content -LiteralPath $recordFile -Raw -ErrorAction SilentlyContinue
            if ($jsonRaw) {
                $parsed = $jsonRaw | ConvertFrom-Json
                if ($parsed -and $parsed.install_mode) {
                    $hasRecord = $true
                    $prevMode = [string]$parsed.install_mode
                    $prevOption = [string]$parsed.selected_option
                    $recordData = $parsed
                }
            }
        } catch { }
    }

    $historyFacts = [PSCustomObject]@{
        HasRecord  = $hasRecord
        PrevOption = $prevOption
        PrevMode   = $prevMode
        RecordData = $recordData
    }

    # 8. Locking Process Facts
    $lockingProcesses = @()
    try {
        $pyProcs = Get-Process -Name 'python', 'pythonw' -ErrorAction SilentlyContinue
        if ($pyProcs) {
            foreach ($p in $pyProcs) {
                try {
                    $procPath = $p.Path
                    if ($procPath -and ($procPath -like "$venvDir*" -or $procPath -like "$RootDirectory*")) {
                        $lockingProcesses += [PSCustomObject]@{
                            Id   = $p.Id
                            Name = $p.ProcessName
                            Path = $procPath
                        }
                    }
                } catch { }
            }
        }
    } catch { }

    return [PSCustomObject]@{
        RootDirectory    = $RootDirectory
        InstallDirectory = $installDir
        OS               = $osFacts
        Gpu              = $gpuFacts
        Venv             = $venvFacts
        Packages         = $pkgFacts
        Tools            = $toolFacts
        Resources        = $resourceFacts
        History          = $historyFacts
        LockingProcesses = $lockingProcesses
    }
}

function Resolve-UserChoice {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)]
        [PSCustomObject]$Facts,

        [Parameter(Mandatory = $false)]
        [string]$ExplicitMode = '',

        [Parameter(Mandatory = $false)]
        [switch]$Yes = $false
    )

    # If explicit valid mode is provided (A, B, or C)
    if ($ExplicitMode -and ($ExplicitMode -match '^[ABC]$')) {
        return $ExplicitMode.ToUpperInvariant()
    }

    # Determine automated recommendation based on Facts
    $recOption = 'A'
    $recReason = 'Stable and safe option with NVIDIA CUDA acceleration.'

    if (-not $Facts.Gpu.HasNvidiaAdapter) {
        $recOption = 'C'
        if ($Facts.Gpu.HasAmd) {
            $recReason = "AMD GPU detected ($($Facts.Gpu.AmdName)), but AMD acceleration is not yet supported. Recommended: CPU."
        } else {
            $recReason = "No dedicated NVIDIA GPU detected. Recommended: CPU."
        }
    } elseif ($Facts.Gpu.ComputeCapability -ge 8.0 -and $Facts.Gpu.DriverMajor -ge 570) {
        $recOption = 'B'
        $recReason = "Ampere+ NVIDIA GPU detected ($($Facts.Gpu.Name), CC $($Facts.Gpu.ComputeCapability)). TensorRT-RTX EP available."
    } elseif ($Facts.Gpu.IsRtxSupported) {
        $recOption = 'A'
        $recReason = "RTX 20-series (Turing) detected. CUDA EP is recommended."
    }

    Write-Host ""
    Write-Host "Hardware recommendation: [$recOption] - $recReason" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Choose installation mode:"
    Write-Host "  [A] CUDA EP         - Stable and safe option (RTX 20+ / Turing or newer)"
    Write-Host "  [B] TensorRT-RTX EP - Experimental, fast, and optimized (Requires RTX 30+ / Ampere; CUDA EP fallback)"
    Write-Host "  [C] CPU             - Safest fallback (CPU inference)"
    Write-Host ""

    if ($Yes) {
        Write-Host "Auto-selecting recommended mode [$recOption] due to -Yes switch." -ForegroundColor Green
        return $recOption
    }

    $promptText = "Mode [A/B/C, default $recOption]: "
    $choice = Prompt-UserChoice -PromptText $promptText -ValidOptions @('A', 'B', 'C') -DefaultOption $recOption
    return $choice
}
