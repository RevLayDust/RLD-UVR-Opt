# Verify.Tests.ps1 - Pester tests for Phase 3: Plan Verification and Gates
Set-StrictMode -Version Latest

$testDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$libDir = Join-Path -Path (Split-Path -Parent $testDir) -ChildPath 'lib'

. (Join-Path -Path $libDir -ChildPath 'Config.ps1')
. (Join-Path -Path $libDir -ChildPath 'Plan.ps1')
. (Join-Path -Path $libDir -ChildPath 'Verify.ps1')
. (Join-Path -Path $testDir -ChildPath 'Fixtures.ps1')

Describe 'Phase 3: Verification & Gate Tests' {

    It 'Scenario 1: No GPU, user picks Option A -> Status is Adjusted, recommends Option C' {
        $facts = New-TestFacts -GpuVendor 'NONE'
        $plan = New-InstallPlan -Facts $facts -Choice 'A'
        $verdict = Test-InstallPlan -Facts $facts -Choice 'A' -Plan $plan

        $verdict.Status | Should Be 'Adjusted'
        $verdict.RecommendedOption | Should Be 'C'
    }

    It 'Scenario 2: AMD GPU only, user picks Option A or B -> Status is Adjusted, recommends Option C' {
        $facts = New-TestFacts -GpuVendor 'AMD'
        $plan = New-InstallPlan -Facts $facts -Choice 'B'
        $verdict = Test-InstallPlan -Facts $facts -Choice 'B' -Plan $plan

        $verdict.Status | Should Be 'Adjusted'
        $verdict.RecommendedOption | Should Be 'C'
    }

    It 'Scenario 3: NVIDIA Turing (CC 7.5), user picks Option B -> Status is Adjusted, recommends Option A' {
        $facts = New-TestFacts -GpuVendor 'NVIDIA' -GpuName 'NVIDIA GeForce RTX 2080 Ti' -DriverMajor 570 -ComputeCapability 7.5
        $plan = New-InstallPlan -Facts $facts -Choice 'B'
        $verdict = Test-InstallPlan -Facts $facts -Choice 'B' -Plan $plan

        $verdict.Status | Should Be 'Adjusted'
        $verdict.RecommendedOption | Should Be 'A'
    }

    It 'Scenario 4: RTX 50-series (CC 12.0, driver 617), picks Option B -> Status is Approved' {
        $facts = New-TestFacts -GpuVendor 'NVIDIA' -GpuName 'NVIDIA GeForce RTX 5060 Ti' -DriverMajor 617 -ComputeCapability 12.0
        $plan = New-InstallPlan -Facts $facts -Choice 'B'
        $verdict = Test-InstallPlan -Facts $facts -Choice 'B' -Plan $plan

        $verdict.Status | Should Be 'Approved'
    }

    It 'Scenario 5: NVIDIA GPU present but driver is too old (< 570) -> Status is Blocked' {
        $facts = New-TestFacts -GpuVendor 'NVIDIA' -DriverMajor 535
        $plan = New-InstallPlan -Facts $facts -Choice 'A'
        $verdict = Test-InstallPlan -Facts $facts -Choice 'A' -Plan $plan

        $verdict.Status | Should Be 'Blocked'
        $failedCheck = $verdict.Checks | Where-Object { $_.Id -eq 'GPU_DRIVER_VERSION' }
        $failedCheck.Result | Should Be 'Fail'
    }

    It 'Scenario 11: Offline (no connection to repositories) -> Status is Blocked' {
        $facts = New-TestFacts -IsOnline $false
        $plan = New-InstallPlan -Facts $facts -Choice 'C'
        $verdict = Test-InstallPlan -Facts $facts -Choice 'C' -Plan $plan

        $verdict.Status | Should Be 'Blocked'
        $netCheck = $verdict.Checks | Where-Object { $_.Id -eq 'NETWORK' }
        $netCheck.Result | Should Be 'Fail'
    }

    It 'Scenario 12: Insufficient disk space (< 5GB) -> Status is Blocked' {
        $facts = New-TestFacts -FreeDiskSpaceBytes 1048576000 # 1 GB
        $plan = New-InstallPlan -Facts $facts -Choice 'C'
        $verdict = Test-InstallPlan -Facts $facts -Choice 'C' -Plan $plan

        $verdict.Status | Should Be 'Blocked'
        $diskCheck = $verdict.Checks | Where-Object { $_.Id -eq 'DISK_SPACE' }
        $diskCheck.Result | Should Be 'Fail'
    }

    It 'Scenario 13: UVR or Python is currently running and holding files -> Status is Blocked' {
        $facts = New-TestFacts -LockingProcessCount 2
        $plan = New-InstallPlan -Facts $facts -Choice 'C'
        $verdict = Test-InstallPlan -Facts $facts -Choice 'C' -Plan $plan

        $verdict.Status | Should Be 'Blocked'
        $lockCheck = $verdict.Checks | Where-Object { $_.Id -eq 'PROCESS_LOCK' }
        $lockCheck.Result | Should Be 'Fail'
    }
}
