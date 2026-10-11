@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "UVR_ROOT=%~dp0"
set "UVR_VENV=%UVR_ROOT%venv\Scripts\activate.bat"
set "UVR_LOG=%UVR_ROOT%uvr_run.log"
if not exist "%UVR_VENV%" (
    echo [ERROR] The UVR virtual environment was not found.
    echo Run install/install_packages.bat first.
    pause
    exit /b 1
)

echo Starting Ultimate Vocal Remover...
call "%UVR_VENV%"
call uv run python UVR.py
set "UVR_EXIT_CODE=%ERRORLEVEL%"
if not "%UVR_EXIT_CODE%"=="0" echo [ERROR] UVR exited with code %UVR_EXIT_CODE%. See uvr_run.log for details.
pause
exit /b %UVR_EXIT_CODE%
