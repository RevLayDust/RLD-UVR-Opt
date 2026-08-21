@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "UVR_ROOT=%~dp0"
set "UVR_PYTHON=%UVR_ROOT%venv\Scripts\python.exe"
set "UVR_LOG=%UVR_ROOT%uvr_run.log"
if not exist "%UVR_PYTHON%" (
    echo [ERROR] The UVR virtual environment was not found.
    echo Run install/install_packages.bat first.
    pause
    exit /b 1
)

echo Starting Ultimate Vocal Remover...
powershell.exe -NoLogo -NoProfile -Command "& { & $env:UVR_PYTHON -u (Join-Path $env:UVR_ROOT 'UVR.py') 2>&1 | Tee-Object -FilePath $env:UVR_LOG; $uvrExitCode = $LASTEXITCODE; if ($null -eq $uvrExitCode) { $uvrExitCode = 1 }; exit $uvrExitCode }"
set "UVR_EXIT_CODE=%ERRORLEVEL%"
if not "%UVR_EXIT_CODE%"=="0" echo [ERROR] UVR exited with code %UVR_EXIT_CODE%. See uvr_run.log for details.
pause
exit /b %UVR_EXIT_CODE%
