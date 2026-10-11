# Record.Tests.ps1 - Pester tests for Phase 5: Install Record writing
Set-StrictMode -Version Latest

$testDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$libDir = Join-Path -Path (Split-Path -Parent $testDir) -ChildPath 'lib'

. (Join-Path -Path $libDir -ChildPath 'Config.ps1')
. (Join-Path -Path $libDir -ChildPath 'Common.ps1')
. (Join-Path -Path $libDir -ChildPath 'Plan.ps1')
. (Join-Path -Path $libDir -ChildPath 'Record.ps1')
. (Join-Path -Path $testDir -ChildPath 'Fixtures.ps1')

Describe 'Phase 5: Record Writing Tests' {
    It 'Writes install record without syntax or runtime error' {
        $tempDir = Join-Path -Path $env:TEMP -ChildPath "uvr_test_rec_$([guid]::NewGuid().ToString('N'))"
        New-Item -ItemType Directory -Path $tempDir -Force | Out-Null

        try {
            $facts = New-TestFacts
            $facts.InstallDirectory = $tempDir

            $plan = New-InstallPlan -Facts $facts -Choice 'B'
            $result = [PSCustomObject]@{ Success = $true; ErrorMessage = ''; SnapshotPath = '' }
            $report = [PSCustomObject]@{
                Passed          = $true
                EffectiveOption = 'B'
                FallbackReason  = 'None'
                Checks          = @()
            }

            { Write-InstallRecord -Facts $facts -Choice 'B' -Plan $plan -Result $result -Report $report } | Should Not Throw

            $recordPath = Join-Path -Path $tempDir -ChildPath 'install_record.json'
            Test-Path -LiteralPath $recordPath | Should Be $true

            $content = Get-Content -LiteralPath $recordPath -Raw | ConvertFrom-Json
            $content.selected_option | Should Be 'B'
            $content.effective_option | Should Be 'B'
            $content.requested_option | Should Be 'B'
        }
        finally {
            if (Test-Path -LiteralPath $tempDir) {
                Remove-Item -LiteralPath $tempDir -Recurse -Force -ErrorAction SilentlyContinue
            }
        }
    }
}
