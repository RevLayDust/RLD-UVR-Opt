# Install-UVR.ps1 - Main entry point orchestrating the 5-phase installer pipeline
[CmdletBinding()]
param(
    [Parameter(Mandatory = $false)]
    [ValidateSet('A', 'B', 'C')]
    [string]$Mode = '',

    [Parameter(Mandatory = $false)]
    [switch]$Yes = $false,

    [Parameter(Mandatory = $false)]
    [switch]$DryRun = $false,

    [Parameter(Mandatory = $false)]
    [ValidateSet('Y', 'N')]
    [string]$ReuseVenv = 'Y',

    [Parameter(Mandatory = $false)]
    [string]$LogPath = ''
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$rootDir = Split-Path -Parent $scriptDir
$libDir = Join-Path -Path $scriptDir -ChildPath 'lib'

# Import library modules
. (Join-Path -Path $libDir -ChildPath 'Config.ps1')
. (Join-Path -Path $libDir -ChildPath 'Common.ps1')
. (Join-Path -Path $libDir -ChildPath 'Detect.ps1')
. (Join-Path -Path $libDir -ChildPath 'Plan.ps1')
. (Join-Path -Path $libDir -ChildPath 'Verify.ps1')
. (Join-Path -Path $libDir -ChildPath 'Install.ps1')
. (Join-Path -Path $libDir -ChildPath 'Validate.ps1')
. (Join-Path -Path $libDir -ChildPath 'Record.ps1')

$config = Get-InstallerConfig

$logDir = if ($LogPath) { Split-Path -Parent $LogPath } else { Join-Path -Path $rootDir -ChildPath 'install_logs' }
Initialize-InstallerLogger -LogDirectory $logDir

Show-InstallerBanner -AppName $config.App.Name -PythonVersion $config.App.PythonVersion -TorchVersion $config.Torch.Version
Write-InstallerLog -Message "Starting $($config.App.Name) installer pipeline..." -Level INFO

try {
    # ----------------------------------------------------
    # Phase 1: Detect (Read-Only)
    # ----------------------------------------------------
    Write-InstallerLog -Message "Phase 1: Detecting system, hardware, and environment..." -Level STEP
    $facts = Get-InstallFacts -RootDirectory $rootDir -Config $config

    Write-InstallerLog -Message "Detected OS: $($facts.OS.VersionString) ($($facts.OS.Architecture))" -Level INFO
    if ($facts.Gpu.HasNvidiaAdapter) {
        Write-InstallerLog -Message "Detected GPU: $($facts.Gpu.Name) (Driver: $($facts.Gpu.DriverVersion), CC: $($facts.Gpu.ComputeCapability))" -Level INFO
    } elseif ($facts.Gpu.HasAmd) {
        Write-InstallerLog -Message "Detected AMD GPU: $($facts.Gpu.AmdName)" -Level INFO
    } else {
        Write-InstallerLog -Message "No dedicated GPU detected; using CPU." -Level INFO
    }

    # ----------------------------------------------------
    # User Choice Resolution
    # ----------------------------------------------------
    $choice = Resolve-UserChoice -Facts $facts -ExplicitMode $Mode -Yes:$Yes
    Write-InstallerLog -Message "Selected installation option: [$choice]" -Level INFO

    # ----------------------------------------------------
    # Phase 2: Plan (Pure Function)
    # ----------------------------------------------------
    Write-InstallerLog -Message "Phase 2: Generating installation plan..." -Level STEP
    $plan = New-InstallPlan -Facts $facts -Choice $choice -Config $config -ReuseVenv $ReuseVenv

    # ----------------------------------------------------
    # Phase 3: Verification (Read-Only Gate)
    # ----------------------------------------------------
    Write-InstallerLog -Message "Phase 3: Verifying plan against hardware invariants..." -Level STEP
    $verdict = Test-InstallPlan -Facts $facts -Choice $choice -Plan $plan -Config $config

    if ($verdict.Status -eq 'Blocked') {
        Write-Host ""
        Write-Host "============================================================" -ForegroundColor Red
        Write-Host " [BLOCKED] Installation cannot proceed on current system:" -ForegroundColor Red
        Write-Host "============================================================" -ForegroundColor Red
        foreach ($c in ($verdict.Checks | Where-Object { $_.Result -eq 'Fail' })) {
            Write-Host " - $($c.Message)" -ForegroundColor Red
            if ($c.Remediation) {
                Write-Host "   Remediation: $($c.Remediation)" -ForegroundColor Yellow
            }
        }
        Write-Host ""
        Write-InstallerLog -Message "Installation blocked at verification gate. No changes made." -Level ERROR
        exit 2
    }

    if ($verdict.Status -eq 'Adjusted') {
        Write-Host ""
        Write-Host "============================================================" -ForegroundColor Yellow
        Write-Host " [ADJUSTED] System requirements adjusted your selection:" -ForegroundColor Yellow
        Write-Host "============================================================" -ForegroundColor Yellow
        Write-Host " Reason: $($verdict.RecommendedReason)" -ForegroundColor Yellow
        Write-Host " Target mode adjusted from [$choice] to [$($verdict.RecommendedOption)]." -ForegroundColor Cyan
        Write-Host ""

        if (-not $Yes) {
            $confirmAdjust = Prompt-UserChoice -PromptText "Proceed with adjusted mode [$($verdict.RecommendedOption)]? [Y/N, default Y]: " -ValidOptions @('Y', 'N') -DefaultOption 'Y'
            if ($confirmAdjust -eq 'N') {
                Write-InstallerLog -Message "Installation cancelled by user on adjustment prompt." -Level WARN
                exit 10
            }
        }
        $choice = $verdict.RecommendedOption
        $plan = New-InstallPlan -Facts $facts -Choice $choice -Config $config -ReuseVenv $ReuseVenv
    }

    # Print Plan Table
    Write-Host ""
    Write-Host "Installation Plan for Profile [$choice]:" -ForegroundColor Cyan
    $plan.Actions | Format-Table -Property Order, Component, Action, Current, Target, Reason -AutoSize

    if ($DryRun) {
        Write-Host ""
        Write-Host "============================================================" -ForegroundColor Cyan
        Write-Host " [DRY-RUN] Verification status: $($verdict.Status)" -ForegroundColor Green
        Write-Host " No changes were made to the system." -ForegroundColor Cyan
        Write-Host "============================================================" -ForegroundColor Cyan
        Write-InstallerLog -Message "Dry-run execution completed successfully." -Level INFO
        exit 0
    }

    # ----------------------------------------------------
    # Phase 4: Install (Mutation)
    # ----------------------------------------------------
    Write-InstallerLog -Message "Phase 4: Executing installation plan..." -Level STEP
    $installResult = Invoke-InstallPlan -Plan $plan -Facts $facts -Config $config

    if (-not $installResult.Success) {
        Write-Host ""
        Write-Host "============================================================" -ForegroundColor Red
        Write-Host " [ERROR] Installation failed during execution:" -ForegroundColor Red
        Write-Host " $($installResult.ErrorMessage)" -ForegroundColor Red
        if ($installResult.SnapshotPath) {
            Write-Host " Environment rollback snapshot: $($installResult.SnapshotPath)" -ForegroundColor Yellow
        }
        Write-Host "============================================================" -ForegroundColor Red
        Write-InstallerLog -Message "Installation failed: $($installResult.ErrorMessage)" -Level ERROR
        exit 3
    }

    # ----------------------------------------------------
    # Phase 5: Validate (+ Fallback Ladder + Record)
    # ----------------------------------------------------
    Write-InstallerLog -Message "Phase 5: Validating installed runtime..." -Level STEP
    $validationReport = Test-InstalledStack -Plan $plan -Facts $facts -Config $config

    if (-not $validationReport.Passed) {
        Write-Host ""
        Write-Host "============================================================" -ForegroundColor Red
        Write-Host " [ERROR] Runtime validation failed after installation." -ForegroundColor Red
        Write-Host "============================================================" -ForegroundColor Red
        foreach ($chk in ($validationReport.Checks | Where-Object { $_.Result -eq 'Fail' })) {
            Write-Host " - $($chk.Message)" -ForegroundColor Red
        }
        Write-InstallerLog -Message "Runtime validation failed." -Level ERROR
        exit 4
    }

    # Atomic Record
    Write-InstallRecord -Facts $facts -Choice $choice -Plan $plan -Result $installResult -Report $validationReport -Config $config

    # Final Summary
    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Green
    Write-Host " Installation complete successfully!" -ForegroundColor Green
    Write-Host "============================================================" -ForegroundColor Green
    Write-Host " Effective Mode : $($validationReport.EffectiveOption)" -ForegroundColor Cyan
    Write-Host " Log File       : $($script:InstallerLogger.LogPath)"
    Write-Host ""
    Write-Host " Run UVR using  : .\run_uvr.bat" -ForegroundColor Cyan
    Write-Host ""

    if (-not $Yes) {
        $runNow = Prompt-UserChoice -PromptText "Run UVR application now? [Y/N, default N]: " -ValidOptions @('Y', 'N') -DefaultOption 'N'
        if ($runNow -eq 'Y') {
            $runBat = Join-Path -Path $rootDir -ChildPath 'run_uvr.bat'
            if (Test-Path -LiteralPath $runBat) {
                Start-Process -FilePath 'cmd.exe' -ArgumentList @('/c', "`"$runBat`"")
            }
        }
    }

    Write-InstallerLog -Message "Installer finished successfully with exit code 0." -Level INFO
    exit 0

} catch {
    Write-Host ""
    Write-Host "[FATAL ERROR] An unexpected exception occurred: $($_.Exception.Message)" -ForegroundColor Red
    Write-InstallerLog -Message "Fatal unexpected exception: $($_.Exception.ToString())" -Level ERROR
    exit 1
}
