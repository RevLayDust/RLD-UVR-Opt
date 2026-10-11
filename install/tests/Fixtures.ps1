# Fixtures.ps1 - Factory generating test $Facts structures for Plan & Verify tests
Set-StrictMode -Version Latest

function New-TestFacts {
    [CmdletBinding()]
    param(
        [string]$GpuVendor = 'NVIDIA', # NVIDIA, AMD, INTEL, NONE
        [string]$GpuName = 'NVIDIA GeForce RTX 4070',
        [int]$DriverMajor = 570,
        [double]$ComputeCapability = 8.9,
        [bool]$HasNvidiaSmi = $true,
        [bool]$VenvExists = $true,
        [string]$TorchVariant = 'None', # CUDA, CPU, None
        [string]$TorchVersion = 'None',
        [string]$TorchLocalTag = '',
        [string]$TorchvisionVariant = 'None',
        [string]$TorchaudioVariant = 'None',
        [bool]$HasOrtCpu = $false,
        [string]$OrtCpuVer = 'None',
        [bool]$HasOrtGpu = $false,
        [string]$OrtGpuVer = 'None',
        [bool]$HasTrtPlugin = $false,
        [string]$TrtPluginVer = 'None',
        [long]$FreeDiskSpaceBytes = 10737418240, # 10 GB
        [bool]$IsOnline = $true,
        [int]$LockingProcessCount = 0
    )

    $hasNvidia = ($GpuVendor -eq 'NVIDIA')
    $hasAmd = ($GpuVendor -eq 'AMD')
    $hasIntel = ($GpuVendor -eq 'INTEL')

    $adapters = @()
    if ($hasNvidia) {
        $adapters += [PSCustomObject]@{ Name = $GpuName; DriverVersion = "$DriverMajor.00"; VramMB = 8192 }
    }
    if ($hasAmd) {
        $adapters += [PSCustomObject]@{ Name = 'AMD Radeon RX 6700 XT'; DriverVersion = '31.0.0'; VramMB = 12288 }
    }
    if ($hasIntel) {
        $adapters += [PSCustomObject]@{ Name = 'Intel Iris Xe Graphics'; DriverVersion = '31.0.0'; VramMB = 2048 }
    }

    $lockingProcs = @()
    if ($LockingProcessCount -gt 0) {
        1..$LockingProcessCount | ForEach-Object {
            $lockingProcs += [PSCustomObject]@{ Id = 1000 + $_; Name = 'python'; Path = "D:\UVR\venv\Scripts\python.exe" }
        }
    }

    $connectivity = @{
        'https://download.pytorch.org/whl/cu128' = $IsOnline
        'https://pypi.org/simple'               = $IsOnline
        'https://www.gyan.dev'                  = $IsOnline
        'https://breakfastquay.com'             = $IsOnline
        'https://aka.ms'                        = $IsOnline
    }

    return [PSCustomObject]@{
        RootDirectory    = 'D:\UVR'
        InstallDirectory = 'D:\UVR\install'
        OS = [PSCustomObject]@{
            VersionString = 'Microsoft Windows 11 Pro 10.0.22631'
            IsWindows     = $true
            Is64Bit       = $true
            IsArm64       = $false
            Architecture  = 'AMD64'
            PSVersion     = '5.1.22621.1'
            IsAdmin       = $false
        }
        Gpu = [PSCustomObject]@{
            AllAdapters       = $adapters
            HasAmd            = $hasAmd
            AmdName           = if ($hasAmd) { 'AMD Radeon RX 6700 XT' } else { '' }
            HasIntel          = $hasIntel
            IntelName         = if ($hasIntel) { 'Intel Iris Xe Graphics' } else { '' }
            HasNvidiaAdapter  = $hasNvidia
            HasNvidiaSmi      = ($hasNvidia -and $HasNvidiaSmi)
            HasNvidiaDriver   = ($hasNvidia -and $HasNvidiaSmi)
            Name              = if ($hasNvidia) { $GpuName } else { 'Unknown' }
            DriverVersion     = if ($hasNvidia) { "$DriverMajor.14" } else { 'Unknown' }
            DriverMajor       = if ($hasNvidia) { $DriverMajor } else { 0 }
            ComputeCapability = if ($hasNvidia) { $ComputeCapability } else { 0.0 }
            VramMB            = if ($hasNvidia) { 8192 } else { 0 }
            IsRtxSupported    = ($hasNvidia -and $GpuName -match 'RTX')
        }
        Venv = [PSCustomObject]@{
            Exists        = $VenvExists
            Path          = 'D:\UVR\venv'
            PythonExe     = 'D:\UVR\venv\Scripts\python.exe'
            PythonVersion = if ($VenvExists) { '3.11.9' } else { 'Unknown' }
            Is64Bit       = $true
        }
        Packages = [PSCustomObject]@{
            TorchInstalled       = ($TorchVariant -ne 'None')
            TorchVersion         = $TorchVersion
            TorchVariant         = $TorchVariant
            TorchLocalTag        = $TorchLocalTag
            TorchvisionInstalled = ($TorchvisionVariant -ne 'None')
            TorchvisionVersion   = '0.26.0'
            TorchvisionVariant   = $TorchvisionVariant
            TorchaudioInstalled  = ($TorchaudioVariant -ne 'None')
            TorchaudioVersion    = '2.11.0'
            TorchaudioVariant    = $TorchaudioVariant
            HasOrtCpu            = $HasOrtCpu
            OrtCpuVersion        = $OrtCpuVer
            HasOrtGpu            = $HasOrtGpu
            OrtGpuVersion        = $OrtGpuVer
            HasOrtConflict       = ($HasOrtCpu -and $HasOrtGpu)
            HasTrtPlugin         = $HasTrtPlugin
            TrtPluginVersion     = $TrtPluginVer
        }
        Tools = [PSCustomObject]@{
            UvInstalled         = $true
            UvVersion           = 'uv 0.6.5'
            FFmpegInstalled     = $true
            FFmpegPath          = 'D:\UVR\ffmpeg.exe'
            FFmpegSource        = 'Root'
            RubberbandInstalled = $true
            RubberbandPath      = 'D:\UVR\rubberband.exe'
            RubberbandSource    = 'Root'
            VcRedistOk          = $true
            VcRedistVersion     = '14.40.33810'
            VcRedistMajor       = 14
            VcRedistMinor       = 40
        }
        Resources = [PSCustomObject]@{
            FreeDiskSpaceBytes = $FreeDiskSpaceBytes
            FreeTempSpaceBytes = 10737418240
            Connectivity       = $connectivity
        }
        History = [PSCustomObject]@{
            HasRecord  = $false
            PrevOption = 'None'
            PrevMode   = 'None'
            RecordData = $null
        }
        LockingProcesses = $lockingProcs
    }
}
