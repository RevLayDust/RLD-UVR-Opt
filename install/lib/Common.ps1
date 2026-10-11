# Common.ps1 - Core logging, native process runner, download, and UI helpers
Set-StrictMode -Version Latest

# Global logger state container
$script:InstallerLogger = @{
    LogPath = $null
}

function Initialize-InstallerLogger {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)]
        [string]$LogDirectory
    )

    if (-not (Test-Path -LiteralPath $LogDirectory)) {
        New-Item -ItemType Directory -Path $LogDirectory -Force | Out-Null
    }

    $now = Get-Date
    $fileName = "install_{0:yyyy-MM-dd_HH-mm-ss}.log" -f $now
    $script:InstallerLogger.LogPath = Join-Path -Path $LogDirectory -ChildPath $fileName
}

function Write-InstallerLog {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)]
        [string]$Message,

        [Parameter(Mandatory = $false)]
        [ValidateSet('INFO', 'WARN', 'ERROR', 'DEBUG', 'STEP')]
        [string]$Level = 'INFO'
    )

    $timestamp = (Get-Date).ToString('yyyy-MM-dd HH:mm:ss')
    $formatted = "[{0}] [{1}] {2}" -f $timestamp, $Level, $Message

    # Console display with suitable colors
    switch ($Level) {
        'WARN'  { Write-Host $formatted -ForegroundColor Yellow }
        'ERROR' { Write-Host $formatted -ForegroundColor Red }
        'STEP'  { Write-Host "`n=== $Message ===" -ForegroundColor Cyan }
        default { Write-Host $formatted }
    }

    if ($script:InstallerLogger.LogPath) {
        try {
            [System.IO.File]::AppendAllText(
                $script:InstallerLogger.LogPath,
                $formatted + "`r`n",
                (New-Object System.Text.UTF8Encoding($false))
            )
        } catch {
            # Ignore file logging failure to prevent crashing installer
        }
    }
}

function Invoke-Native {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)]
        [string]$FilePath,

        [Parameter(Mandatory = $false)]
        [string[]]$ArgumentList = @(),

        [Parameter(Mandatory = $false)]
        [string]$WorkingDirectory = $null,

        [Parameter(Mandatory = $false)]
        [int]$TimeoutSeconds = 300
    )

    $cmdLine = "$FilePath $($ArgumentList -join ' ')"
    Write-InstallerLog -Message "Executing: $cmdLine" -Level DEBUG

    $prevEAP = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'

    $stdoutLines = @()
    $stderrLines = @()
    $exitCode = 0

    $prevLocation = Get-Location
    if ($WorkingDirectory -and (Test-Path -LiteralPath $WorkingDirectory)) {
        Set-Location -LiteralPath $WorkingDirectory
    }

    try {
        if ($ArgumentList.Count -gt 0) {
            $rawOutput = & $FilePath @ArgumentList 2>&1
        } else {
            $rawOutput = & $FilePath 2>&1
        }
        $exitCode = if ($null -ne $LASTEXITCODE) { $LASTEXITCODE } else { 0 }

        if ($rawOutput) {
            foreach ($item in $rawOutput) {
                if ($item -is [System.Management.Automation.ErrorRecord]) {
                    $stderrLines += $item.ToString()
                } else {
                    $stdoutLines += $item.ToString()
                }
            }
        }

        $stdoutStr = $stdoutLines -join "`r`n"
        $stderrStr = $stderrLines -join "`r`n"

        Write-InstallerLog -Message "Completed ($exitCode): $cmdLine" -Level DEBUG

        return [PSCustomObject]@{
            ExitCode = $exitCode
            StdOut   = $stdoutStr
            StdErr   = $stderrStr
            Success  = ($exitCode -eq 0)
        }
    } catch {
        Write-InstallerLog -Message "Execution failed ($cmdLine): $($_.Exception.Message)" -Level ERROR
        return [PSCustomObject]@{
            ExitCode = -1
            StdOut   = ''
            StdErr   = $_.Exception.Message
            Success  = $false
        }
    } finally {
        if ($WorkingDirectory -and (Test-Path -LiteralPath $WorkingDirectory)) {
            Set-Location -LiteralPath $prevLocation
        }
        $ErrorActionPreference = $prevEAP
    }
}

function Write-FileUtf8NoBom {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,

        [Parameter(Mandatory = $true)]
        [string]$Content
    )

    $dir = Split-Path -Path $Path -Parent
    if ($dir -and (-not (Test-Path -LiteralPath $dir))) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }

    [System.IO.File]::WriteAllText(
        $Path,
        $Content,
        (New-Object System.Text.UTF8Encoding($false))
    )
}

function Invoke-DownloadWithRetry {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)]
        [string]$Uri,

        [Parameter(Mandatory = $true)]
        [string]$OutFile,

        [Parameter(Mandatory = $false)]
        [int]$MaxAttempts = 3,

        [Parameter(Mandatory = $false)]
        [int]$DelaySeconds = 2
    )

    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    $prevProgress = $ProgressPreference
    $ProgressPreference = 'SilentlyContinue'

    $hasCurl = $null -ne (Get-Command 'curl.exe' -ErrorAction SilentlyContinue)

    $attempt = 1
    $downloadSuccess = $false

    while ($attempt -le $MaxAttempts) {
        Write-InstallerLog -Message "Downloading from $Uri (attempt $attempt/$MaxAttempts)..." -Level INFO

        try {
            if ($hasCurl) {
                $res = Invoke-Native -FilePath 'curl.exe' -ArgumentList @('-f', '-L', '--retry', '2', '-o', "`"$OutFile`"", "`"$Uri`"")
                if ($res.Success -and (Test-Path -LiteralPath $OutFile) -and ((Get-Item -LiteralPath $OutFile).Length -gt 0)) {
                    $downloadSuccess = $true
                    break
                }
            } else {
                $webClient = New-Object System.Net.WebClient
                try {
                    $webClient.DownloadFile($Uri, $OutFile)
                    if ((Test-Path -LiteralPath $OutFile) -and ((Get-Item -LiteralPath $OutFile).Length -gt 0)) {
                        $downloadSuccess = $true
                        break
                    }
                } finally {
                    $webClient.Dispose()
                }
            }
        } catch {
            Write-InstallerLog -Message "Download attempt $attempt failed: $($_.Exception.Message)" -Level WARN
        }

        if (Test-Path -LiteralPath $OutFile) {
            Remove-Item -LiteralPath $OutFile -Force -ErrorAction SilentlyContinue
        }

        Start-Sleep -Seconds $DelaySeconds
        $attempt++
    }

    $ProgressPreference = $prevProgress
    return $downloadSuccess
}

function Prompt-UserChoice {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)]
        [string]$PromptText,

        [Parameter(Mandatory = $true)]
        [string[]]$ValidOptions,

        [Parameter(Mandatory = $false)]
        [string]$DefaultOption = ''
    )

    $validLower = @($ValidOptions | ForEach-Object { $_.ToLowerInvariant() })

    while ($true) {
        Write-Host -NoNewline $PromptText
        $inputVal = Read-Host

        if ([string]::IsNullOrWhiteSpace($inputVal)) {
            if (-not [string]::IsNullOrEmpty($DefaultOption)) {
                return $DefaultOption.ToUpperInvariant()
            }
            Write-Host "Input is required. Options: ($($ValidOptions -join '/'))" -ForegroundColor Yellow
            continue
        }

        $trimmed = $inputVal.Trim().ToLowerInvariant()
        if ($validLower -contains $trimmed) {
            return $trimmed.ToUpperInvariant()
        }

        Write-Host "Invalid choice '$inputVal'. Please choose from: ($($ValidOptions -join '/'))" -ForegroundColor Yellow
    }
}

function Show-InstallerBanner {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)]
        [string]$AppName,

        [Parameter(Mandatory = $true)]
        [string]$PythonVersion,

        [Parameter(Mandatory = $true)]
        [string]$TorchVersion
    )

    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host " $AppName - Modern Installer" -ForegroundColor Cyan
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host " Target : NVIDIA RTX 20-series or newer, with CPU fallback"
    Write-Host " Python : $PythonVersion"
    Write-Host " PyTorch: $TorchVersion"
    Write-Host " CUDA   : cu128 primary"
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host ""
}
