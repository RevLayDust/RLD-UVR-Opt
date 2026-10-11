# Config.ps1 - Single source of truth for all installer configuration values
Set-StrictMode -Version Latest

function Get-InstallerConfig {
    [CmdletBinding()]
    param()

    return [PSCustomObject]@{
        App = [PSCustomObject]@{
            Name          = 'RLD UVR-Opt'
            PythonVersion = '3.11.9'
            MinDriverMajor = 570
        }

        Torch = [PSCustomObject]@{
            Version         = '2.11.0'
            VisionVersion   = '0.26.0'
            AudioVersion    = '2.11.0'
            Packages        = @('torch==2.11.0', 'torchvision==0.26.0', 'torchaudio==2.11.0')
            Cu128IndexUrl   = 'https://download.pytorch.org/whl/cu128'
            CpuIndexUrl     = 'https://download.pytorch.org/whl/cpu'
            TargetCuda      = '12.8'
        }

        Ort = [PSCustomObject]@{
            CudaVersion      = '1.22.0'
            CpuVersion       = '1.22.0'
            TrtGpuVersion    = '1.24.4'
            TrtPluginVersion = '0.4.0'
            TrtPluginPkg     = 'onnxruntime-ep-nv-tensorrt-rtx-cu12==0.4.0'
        }

        Hardware = [PSCustomObject]@{
            MinTrtComputeCapability = 8.0
            MinFreeDiskSpaceBytes   = 5368709120 # 5 GB
            MinDriverMajor          = 570
        }

        ExternalTools = [PSCustomObject]@{
            FFmpegUrl     = 'https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip'
            RubberbandUrl = 'https://breakfastquay.com/files/releases/rubberband-3.1.2-gpl-executable-windows.zip'
            VcRedistUrl   = 'https://aka.ms/vs/17/release/vc_redist.x64.exe'
            VcRedistMinMajor = 14
            VcRedistMinMinor = 40
        }

        Network = [PSCustomObject]@{
            TimeoutSeconds = 10
            HostsToCheck   = @(
                'https://download.pytorch.org/whl/cu128',
                'https://pypi.org/simple',
                'https://www.gyan.dev',
                'https://breakfastquay.com',
                'https://aka.ms'
            )
        }

        Files = [PSCustomObject]@{
            RequirementsMain   = 'reqs/requirements.txt'
            RequirementsTrtRtx = 'reqs/requirements_trt_rtx.txt'
            RequirementsCpu    = 'reqs/requirements_cpu.txt'
            InstallRecordJson  = 'install_record.json'
            InstallRecordPy    = 'install_record.py'
            InstallStateJson   = 'install_state.json'
            RollbackSnapshot   = 'rollback_snapshot.txt'
        }
    }
}
