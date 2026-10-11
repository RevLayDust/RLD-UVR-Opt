# Record.ps1 - Phase 5: Atomic recording of installation state with additive schema
Set-StrictMode -Version Latest

function Write-InstallRecord {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory = $true)]
        [PSCustomObject]$Facts,

        [Parameter(Mandatory = $true)]
        [string]$Choice,

        [Parameter(Mandatory = $true)]
        [PSCustomObject]$Plan,

        [Parameter(Mandatory = $true)]
        [PSCustomObject]$Result,

        [Parameter(Mandatory = $true)]
        [PSCustomObject]$Report,

        [Parameter(Mandatory = $false)]
        [PSCustomObject]$Config = $null
    )

    if ($null -eq $Config) {
        $Config = Get-InstallerConfig
    }

    $installDir = $Facts.InstallDirectory
    $recordFile = Join-Path -Path $installDir -ChildPath $Config.Files.InstallRecordJson
    $recordPy = Join-Path -Path $installDir -ChildPath $Config.Files.InstallRecordPy
    $venvPython = $Facts.Venv.PythonExe

    $effectiveOpt = $Report.EffectiveOption
    $effectiveMode = switch ($effectiveOpt) {
        'A' { 'NVIDIA' }
        'B' { 'TENSORRT_RTX' }
        'C' { 'CPU' }
    }

    $hasFfmpegStr = if ($Facts.Tools.FFmpegInstalled) { '1' } else { '0' }
    $hasRbStr = if ($Facts.Tools.RubberbandInstalled) { '1' } else { '0' }
    $hasVcStr = if ($Facts.Tools.VcRedistOk) { '1' } else { '0' }
    $reuseVenvStr = if ($Plan.VenvStrategy -eq 'Reuse') { 'Y' } else { 'N' }

    Write-InstallerLog -Message "Recording installation state to $recordFile..." -Level INFO

    # Step 1: Call install_record.py write for backwards-compatible base schema
    if (Test-Path -LiteralPath $venvPython) {
        $pyArgs = @(
            "`"$recordPy`"", 'write',
            '--option', $effectiveOpt,
            '--mode', $effectiveMode,
            '--reuse-venv', $reuseVenvStr,
            '--has-uv', '1',
            '--has-ffmpeg', $hasFfmpegStr,
            '--has-rubberband', $hasRbStr,
            '--has-vc-redist', $hasVcStr
        )
        Invoke-Native -FilePath $venvPython -ArgumentList $pyArgs | Out-Null
    }

    # Step 2: Read written JSON, add additive enhancements, and write atomically (UTF-8 without BOM)
    $recordObj = $null
    if (Test-Path -LiteralPath $recordFile) {
        try {
            $rawJson = Get-Content -LiteralPath $recordFile -Raw -ErrorAction SilentlyContinue
            if ($rawJson) {
                $recordObj = $rawJson | ConvertFrom-Json
            }
        } catch { }
    }

    if ($null -eq $recordObj) {
        $recordObj = [PSCustomObject]@{
            schema_version  = 1
            timestamp_utc   = (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ')
            selected_option = $effectiveOpt
            install_mode    = $effectiveMode
        }
    }

    # Additive fields
    $recordObj | Add-Member -MemberType NoteProperty -Name 'requested_option' -Value $Choice -Force
    $recordObj | Add-Member -MemberType NoteProperty -Name 'effective_option' -Value $effectiveOpt -Force
    $recordObj | Add-Member -MemberType NoteProperty -Name 'fallback_reason' -Value $Report.FallbackReason -Force

    $detectedObj = [PSCustomObject]@{
        gpu_name           = $Facts.Gpu.Name
        driver_version     = $Facts.Gpu.DriverVersion
        compute_capability = $Facts.Gpu.ComputeCapability
        vram_mb            = $Facts.Gpu.VramMB
        os_version         = $Facts.OS.VersionString
        os_arch            = $Facts.OS.Architecture
        vc_redist_ver      = $Facts.Tools.VcRedistVersion
    }
    $recordObj | Add-Member -MemberType NoteProperty -Name 'detected' -Value $detectedObj -Force

    $recordObj | Add-Member -MemberType NoteProperty -Name 'validation' -Value $Report.Checks -Force

    # Write atomically via temp file
    $tempFile = Join-Path -Path $installDir -ChildPath "record_tmp_$([guid]::NewGuid().ToString('N')).json"
    $jsonOutput = $recordObj | ConvertTo-Json -Depth 6

    Write-FileUtf8NoBom -Path $tempFile -Content $jsonOutput
    Move-Item -LiteralPath $tempFile -Destination $recordFile -Force

    Write-InstallerLog -Message "Successfully written installation record." -Level INFO
}
