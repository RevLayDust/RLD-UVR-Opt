@echo off
setlocal EnableExtensions EnableDelayedExpansion

cd /d "%~dp0"

set "INSTALLER_FILE=%~f0"
powershell -NoProfile -Command "$previous = -1; foreach ($byte in [IO.File]::ReadAllBytes($env:INSTALLER_FILE)) { if ($byte -eq 10 -and $previous -ne 13) { exit 1 }; $previous = $byte }; exit 0" >nul 2>&1
if errorlevel 1 (
    echo.
    echo [ERROR] install_packages.bat has invalid Unix LF line endings.
    echo         Convert this file to Windows CRLF or restore it from Git before running it.
    pause
    exit /b 1
)
set "INSTALLER_FILE="

set "APP_NAME=Ultimate Vocal Remover"
set "PYTHON_VERSION=3.11.9"
set "VENV_DIR=..\venv"
set "VENV_PYTHON=%VENV_DIR%\Scripts\python.exe"
set "REQ_FILE=requirements.txt"
set "LOG_DIR=..\install_logs"
set "LOG_FILE=%LOG_DIR%\install_%DATE:/=-%_%TIME::=-%.log"
set "LOG_FILE=%LOG_FILE: =0%"
set "CPU_REQ_FILE=%LOG_DIR%\requirements_cpu.generated.txt"
set "UV_LINK_MODE=copy"
set "ROOT_DIR=.."
set "FFMPEG_URL=https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
set "RUBBERBAND_URL=https://breakfastquay.com/files/releases/rubberband-3.1.2-gpl-executable-windows.zip"

set "TORCH_VERSION=2.11.0"
set "TORCHVISION_VERSION=0.26.0"
set "TORCHAUDIO_VERSION=2.11.0"
set "TORCH_PACKAGES=torch==%TORCH_VERSION% torchvision==%TORCHVISION_VERSION% torchaudio==%TORCHAUDIO_VERSION%"
set "TORCH_CU128=https://download.pytorch.org/whl/cu128"
set "TORCH_CPU=https://download.pytorch.org/whl/cpu"
set "MIN_DRIVER_MAJOR=570"

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%" >nul 2>&1

call :banner
call :log "Starting installer in %CD%"

if not exist "%REQ_FILE%" (
    call :fail "requirements.txt was not found in this folder."
    exit /b 1
)

call :check_windows
call :detect_gpu
call :choose_mode
call :ensure_uv
if errorlevel 1 exit /b 1
call :create_venv
if errorlevel 1 exit /b 1
call :verify_venv
if errorlevel 1 exit /b 1

if /i "%INSTALL_MODE%"=="CPU" (
    call :install_cpu_stack
) else (
    call :install_nvidia_stack
)
if errorlevel 1 exit /b 1

call :install_base_deps
if errorlevel 1 exit /b 1

call :ensure_external_tools
if errorlevel 1 exit /b 1

call :smoke_test
if errorlevel 1 exit /b 1
call :done
exit /b 0

:banner
echo.
echo ============================================================
echo  %APP_NAME% - Modern Installer
echo ============================================================
echo  Target : NVIDIA RTX 20-series or newer, with CPU fallback
echo  Python : %PYTHON_VERSION%
echo  PyTorch: %TORCH_VERSION%
echo  CUDA   : cu128 primary
echo ============================================================
echo.
exit /b 0

:check_windows
if /i not "%OS%"=="Windows_NT" (
    call :warn "This installer is designed for Windows. Use install_packages.sh on other operating systems."
)
exit /b 0

:verify_venv
if not exist "%VENV_PYTHON%" (
    call :fail "The virtual environment Python executable was not found: %VENV_PYTHON%"
    exit /b 1
)

if defined CONDA_PREFIX (
    echo Active Conda environment detected: %CONDA_PREFIX%
    call :warn "Conda is active, but it will be bypassed. Every command is pinned to %VENV_PYTHON%."
)

set "VENV_CHECK_LOG=%TEMP%\uvr_venv_check_%RANDOM%%RANDOM%.log"
"%VENV_PYTHON%" -c "import sys; print(sys.executable); raise SystemExit(sys.prefix == sys.base_prefix)" >"!VENV_CHECK_LOG!" 2>&1
if errorlevel 1 (
    call :warn "Virtual environment diagnostic output:"
    if exist "!VENV_CHECK_LOG!" type "!VENV_CHECK_LOG!"
    if exist "!VENV_CHECK_LOG!" del "!VENV_CHECK_LOG!" >nul 2>&1
    call :fail "The existing virtual environment is invalid or could not be started. Choose N when asked to reuse it so the installer can rebuild it."
    exit /b 1
)

set /p "RESOLVED_VENV_PYTHON=" <"!VENV_CHECK_LOG!"
del "!VENV_CHECK_LOG!" >nul 2>&1
echo Using isolated Python : !RESOLVED_VENV_PYTHON!
call :log "Using isolated Python: !RESOLVED_VENV_PYTHON!"
exit /b 0

:detect_gpu
set "HAS_NVIDIA=0"
set "HAS_AMD=0"
set "GPU_NAME=Not detected"
set "AMD_GPU_NAME=Not detected"
set "GPU_DRIVER=Unknown"
set "RTX_SUPPORTED=0"
set "DRIVER_SUPPORTED=0"

for /f "usebackq delims=" %%A in (`powershell -NoProfile -Command "$gpu = Get-CimInstance Win32_VideoController | Where-Object { $_.Name -match 'AMD|Radeon' } | Select-Object -First 1 -ExpandProperty Name; if ($gpu) { $gpu }" 2^>nul`) do (
    set "HAS_AMD=1"
    set "AMD_GPU_NAME=%%A"
)

where nvidia-smi >nul 2>&1
if errorlevel 1 (
    call :warn "nvidia-smi was not found. The NVIDIA driver may be missing, or PATH may not be configured correctly."
    if "!HAS_AMD!"=="1" (
        echo Detected AMD GPU    : !AMD_GPU_NAME!
        call :warn "AMD GPUs are not supported by this UVR installer yet. CPU fallback will be used."
        call :log "Detected AMD GPU: !AMD_GPU_NAME!. AMD is unsupported; using CPU fallback."
    )
    exit /b 0
)

for /f "tokens=1,* delims=," %%A in ('nvidia-smi --query-gpu=name^,driver_version --format=csv^,noheader 2^>nul') do (
    set "HAS_NVIDIA=1"
    set "GPU_NAME=%%A"
    set "GPU_DRIVER=%%B"
    goto :gpu_detected
)

:gpu_detected
if "%HAS_NVIDIA%"=="0" (
    call :warn "No NVIDIA GPU was detected."
    if "!HAS_AMD!"=="1" (
        echo Detected AMD GPU    : !AMD_GPU_NAME!
        call :warn "AMD GPUs are not supported by this UVR installer yet. CPU fallback will be used."
        call :log "Detected AMD GPU: !AMD_GPU_NAME!. AMD is unsupported; using CPU fallback."
    )
    exit /b 0
)

for /f "tokens=* delims= " %%A in ("!GPU_NAME!") do set "GPU_NAME=%%A"
for /f "tokens=* delims= " %%A in ("!GPU_DRIVER!") do set "GPU_DRIVER=%%A"

echo Detected NVIDIA GPU : !GPU_NAME!
echo NVIDIA Driver      : !GPU_DRIVER!
call :log "Detected NVIDIA GPU: !GPU_NAME!, driver: !GPU_DRIVER!"

echo !GPU_NAME! | findstr /i /r "RTX.*20 RTX.*30 RTX.*40 RTX.*50 RTX A RTX Ada RTX PRO" >nul && set "RTX_SUPPORTED=1"

for /f "tokens=1 delims=." %%A in ("!GPU_DRIVER!") do set "DRIVER_MAJOR=%%A"
set "DRIVER_CHECK=0"
set /a DRIVER_CHECK=1!DRIVER_MAJOR!-1000 >nul 2>&1
if !DRIVER_CHECK! GEQ %MIN_DRIVER_MAJOR% set "DRIVER_SUPPORTED=1"

if "!RTX_SUPPORTED!"=="0" (
    call :warn "This GPU does not match the official RTX 20-series-or-newer target. GPU installation can still be attempted, but it is not recommended."
)

if "!DRIVER_SUPPORTED!"=="0" (
    call :warn "The NVIDIA driver appears older than the CUDA 12.8 recommendation. Updating the driver is recommended before using NVIDIA mode."
)
exit /b 0

:choose_mode
set "INSTALL_MODE=AUTO"
set "USER_FORCED_NVIDIA=0"
echo.
echo Choose installation mode:
echo   [A] Auto       - choose NVIDIA for detected RTX GPUs; otherwise CPU
echo   [N] NVIDIA GPU - force PyTorch CUDA 12.8
echo   [C] CPU        - safest fallback, including AMD systems because AMD is not supported yet
if "%HAS_NVIDIA%"=="0" if "%HAS_AMD%"=="1" echo   Note: AMD GPU detected, but AMD acceleration is not supported yet. Auto will use CPU.
echo.
call :prompt_choice MODE_CHOICE "Mode [A/N/C, default A]: " "A N C" "A" "A, N, or C"
if /i "!MODE_CHOICE!"=="N" (
    set "INSTALL_MODE=NVIDIA"
    set "USER_FORCED_NVIDIA=1"
)
if /i "!MODE_CHOICE!"=="C" set "INSTALL_MODE=CPU"

if /i "%INSTALL_MODE%"=="AUTO" (
    if "%HAS_NVIDIA%"=="1" if "%RTX_SUPPORTED%"=="1" set "INSTALL_MODE=NVIDIA"
    if /i "!INSTALL_MODE!"=="AUTO" set "INSTALL_MODE=CPU"
)

if /i "!INSTALL_MODE!"=="NVIDIA" if "%USER_FORCED_NVIDIA%"=="1" (
    if "%HAS_NVIDIA%"=="0" call :confirm_risky "NVIDIA was not detected. Force GPU stack installation anyway?"
    if /i "!INSTALL_MODE!"=="CPU" goto :choose_mode_done
    if "%RTX_SUPPORTED%"=="0" call :confirm_risky "The GPU is outside the RTX 20+ target. Force GPU stack installation anyway?"
    if /i "!INSTALL_MODE!"=="CPU" goto :choose_mode_done
    if "%DRIVER_SUPPORTED%"=="0" call :confirm_risky "The driver is older than the CUDA 12.8 recommendation. Force GPU stack installation anyway?"
)

:choose_mode_done
echo Selected mode: !INSTALL_MODE!
call :log "Selected install mode: !INSTALL_MODE!"
exit /b 0

:ensure_uv
where uv >nul 2>&1
if not errorlevel 1 (
    echo uv already installed.
    uv --version
    exit /b 0
)

echo Installing uv...
call :log "Installing uv"
powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
if errorlevel 1 (
    call :fail "Failed to install uv. Check your internet connection or PowerShell policy."
    exit /b 1
)

where uv >nul 2>&1
if errorlevel 1 (
    set "PATH=%USERPROFILE%\.local\bin;%USERPROFILE%\.cargo\bin;%PATH%"
)

where uv >nul 2>&1
if errorlevel 1 (
    call :fail "uv was installed, but it was not found in PATH. Close this terminal and try again."
    exit /b 1
)
uv --version
exit /b 0

:create_venv
set "VENV_CLEAR="
if exist "%VENV_DIR%\Scripts\python.exe" (
    echo Existing virtual environment found: %VENV_DIR%
    call :prompt_choice REUSE_VENV "Reuse existing venv? [Y/N, default Y]: " "Y N" "Y" "Y or N"
    if /i "!REUSE_VENV!"=="Y" exit /b 0
    set "VENV_CLEAR=--clear"
)

echo Creating virtual environment with Python %PYTHON_VERSION%...
call :log "Creating venv with Python %PYTHON_VERSION%"
uv venv %VENV_CLEAR% "%VENV_DIR%" --python %PYTHON_VERSION%
if errorlevel 1 (
    call :warn "Python %PYTHON_VERSION% was not available automatically. Trying any available Python 3.11 installation."
    uv venv %VENV_CLEAR% "%VENV_DIR%" --python 3.11
    if errorlevel 1 (
        call :fail "Failed to create the venv. Install Python 3.11 first or check uv connectivity."
        exit /b 1
    )
)
exit /b 0

:install_base_deps
echo.
echo Preparing base dependencies...
set "INSTALL_REQ=%REQ_FILE%"
if /i "%INSTALL_MODE%"=="CPU" (
    call :make_cpu_requirements
    if errorlevel 1 exit /b 1
    set "INSTALL_REQ=%CPU_REQ_FILE%"
)

echo Installing base dependencies...
call :log "Installing requirements from !INSTALL_REQ!"
uv pip install --python "%VENV_PYTHON%" -r "!INSTALL_REQ!"
set "BASE_INSTALL_EXIT=!ERRORLEVEL!"
if /i "%INSTALL_MODE%"=="CPU" if exist "%CPU_REQ_FILE%" del "%CPU_REQ_FILE%" >nul 2>&1
if not "!BASE_INSTALL_EXIT!"=="0" (
    call :fail "Failed to install base dependencies. See %LOG_FILE% for details."
    exit /b 1
)
exit /b 0

:make_cpu_requirements
if exist "%CPU_REQ_FILE%" del "%CPU_REQ_FILE%" >nul 2>&1
type nul >"%CPU_REQ_FILE%"
if errorlevel 1 (
    call :fail "Failed to create the CPU requirements file in %LOG_DIR%."
    exit /b 1
)

for /f "usebackq delims=" %%L in ("%REQ_FILE%") do (
    set "LINE=%%L"
    set "SKIP=0"
    echo !LINE! | findstr /i /b "torch== torchvision== torchaudio==" >nul && set "SKIP=1"
    echo !LINE! | findstr /i /b "cupy-cuda cuda-pathfinder onnxruntime-gpu nvidia-" >nul && set "SKIP=1"
    echo !LINE! | findstr /i /b "polygraphy nvidia-modelopt onnx-graphsurgeon" >nul && set "SKIP=1"
    if "!SKIP!"=="0" echo !LINE!>>"%CPU_REQ_FILE%"
)

for %%A in ("%CPU_REQ_FILE%") do if %%~zA EQU 0 (
    call :fail "The generated CPU requirements file is empty."
    exit /b 1
)
exit /b 0

:install_nvidia_stack
echo.
echo Installing NVIDIA CUDA PyTorch stack...
call :log "Installing PyTorch CUDA from %TORCH_CU128%"
uv pip install --python "%VENV_PYTHON%" %TORCH_PACKAGES% --index-url "%TORCH_CU128%"
if errorlevel 1 (
    call :warn "PyTorch CUDA installation failed. Offering CPU fallback."
    call :prompt_choice CPU_FALLBACK "Install CPU stack instead? [Y/N, default Y]: " "Y N" "Y" "Y or N"
    if /i "!CPU_FALLBACK!"=="N" (
        call :fail "PyTorch CUDA installation failed and CPU fallback was cancelled."
        exit /b 1
    )
    set "INSTALL_MODE=CPU"
    call :install_cpu_stack
    exit /b 0
)

echo Installing ONNX Runtime GPU compatibility package...
uv pip install --python "%VENV_PYTHON%" onnxruntime-gpu==1.22.0
if errorlevel 1 call :warn "Failed to install onnxruntime-gpu. UVR can still use PyTorch CUDA, but ONNX models may fall back or fail."
exit /b 0

:install_cpu_stack
echo.
echo Installing CPU fallback stack...
call :log "Installing PyTorch CPU from %TORCH_CPU%"
uv pip install --python "%VENV_PYTHON%" %TORCH_PACKAGES% --index-url "%TORCH_CPU%"
if errorlevel 1 (
    call :fail "Failed to install PyTorch CPU."
    exit /b 1
)

uv pip install --python "%VENV_PYTHON%" onnxruntime==1.22.0
if errorlevel 1 call :warn "Failed to install ONNX Runtime CPU. PyTorch models can still run, but ONNX models may fail."
exit /b 0

:ensure_external_tools
echo.
echo ============================================================
echo  Checking External Media Binaries (FFmpeg ^& Rubber Band)
echo ============================================================
call :log "Checking external media binaries"

call :ensure_ffmpeg
call :ensure_rubberband
echo.
exit /b 0

:ensure_ffmpeg
set "HAS_FFMPEG=0"
set "DETECTED_FFMPEG="

if exist "%ROOT_DIR%\ffmpeg.exe" (
    set "HAS_FFMPEG=1"
    set "DETECTED_FFMPEG=%ROOT_DIR%\ffmpeg.exe"
    echo [OK] FFmpeg found in root directory: !DETECTED_FFMPEG!
    call :log "FFmpeg found in root: !DETECTED_FFMPEG!"
    exit /b 0
)

where ffmpeg >nul 2>&1
if not errorlevel 1 (
    for /f "delims=" %%A in ('where ffmpeg 2^>nul') do (
        set "HAS_FFMPEG=1"
        set "DETECTED_FFMPEG=%%A"
        goto :ffmpeg_path_found
    )
)

:ffmpeg_path_found
if "!HAS_FFMPEG!"=="1" (
    echo [OK] FFmpeg detected in system PATH: !DETECTED_FFMPEG!
    call :log "FFmpeg detected in system PATH: !DETECTED_FFMPEG!"
    exit /b 0
)

echo FFmpeg was not detected in system PATH or root folder.
echo Downloading portable FFmpeg to application root...
call :log "Downloading portable FFmpeg from %FFMPEG_URL%"
set "FFMPEG_ZIP=%TEMP%\uvr_ffmpeg_%RANDOM%.zip"
powershell -NoProfile -ExecutionPolicy Bypass -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; if (Get-Command curl.exe -ErrorAction SilentlyContinue) { curl.exe -f -L --progress-bar -o '%FFMPEG_ZIP%' '%FFMPEG_URL%' } else { (New-Object System.Net.WebClient).DownloadFile('%FFMPEG_URL%', '%FFMPEG_ZIP%') }"
if errorlevel 1 (
    if exist "%FFMPEG_ZIP%" del "%FFMPEG_ZIP%" >nul 2>&1
    call :warn "Failed to download portable FFmpeg. Non-WAV audio conversion may be unavailable."
    exit /b 0
)

echo Extracting ffmpeg.exe to %ROOT_DIR%...
powershell -NoProfile -ExecutionPolicy Bypass -Command "Add-Type -AssemblyName System.IO.Compression.FileSystem; $archive = [System.IO.Compression.ZipFile]::OpenRead('%FFMPEG_ZIP%'); foreach ($entry in $archive.Entries) { if ($entry.Name -eq 'ffmpeg.exe' -or $entry.Name -eq 'ffprobe.exe') { [System.IO.Compression.ZipFileExtensions]::ExtractToFile($entry, (Join-Path '%ROOT_DIR%' $entry.Name), $true) } }; $archive.Dispose()"
if exist "%FFMPEG_ZIP%" del "%FFMPEG_ZIP%" >nul 2>&1

if exist "%ROOT_DIR%\ffmpeg.exe" (
    echo [OK] FFmpeg portable installed successfully: %ROOT_DIR%\ffmpeg.exe
    call :log "FFmpeg portable installed to %ROOT_DIR%\ffmpeg.exe"
) else (
    call :warn "FFmpeg archive downloaded, but ffmpeg.exe could not be extracted."
)
exit /b 0

:ensure_rubberband
set "HAS_RB=0"
set "DETECTED_RB="

if exist "%ROOT_DIR%\rubberband.exe" (
    set "HAS_RB=1"
    set "DETECTED_RB=%ROOT_DIR%\rubberband.exe"
    echo [OK] rubberband-cli found in root directory: !DETECTED_RB!
    call :log "rubberband-cli found in root: !DETECTED_RB!"
    exit /b 0
)

where rubberband >nul 2>&1
if not errorlevel 1 (
    for /f "delims=" %%A in ('where rubberband 2^>nul') do (
        set "HAS_RB=1"
        set "DETECTED_RB=%%A"
        goto :rb_path_found
    )
)

where rubberband-cli >nul 2>&1
if not errorlevel 1 (
    for /f "delims=" %%A in ('where rubberband-cli 2^>nul') do (
        set "HAS_RB=1"
        set "DETECTED_RB=%%A"
        goto :rb_path_found
    )
)

:rb_path_found
if "!HAS_RB!"=="1" (
    echo [OK] rubberband-cli detected in system PATH: !DETECTED_RB!
    call :log "rubberband-cli detected in system PATH: !DETECTED_RB!"
    exit /b 0
)

echo rubberband-cli was not detected in system PATH or root folder.
echo Downloading portable Rubber Band to application root...
call :log "Downloading portable Rubber Band from %RUBBERBAND_URL%"
set "RB_ZIP=%TEMP%\uvr_rubberband_%RANDOM%.zip"
powershell -NoProfile -ExecutionPolicy Bypass -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; if (Get-Command curl.exe -ErrorAction SilentlyContinue) { curl.exe -f -L --progress-bar -o '%RB_ZIP%' '%RUBBERBAND_URL%' } else { (New-Object System.Net.WebClient).DownloadFile('%RUBBERBAND_URL%', '%RB_ZIP%') }"
if errorlevel 1 (
    if exist "%RB_ZIP%" del "%RB_ZIP%" >nul 2>&1
    call :warn "Failed to download portable Rubber Band. Time-stretch and pitch-shift tools will be unavailable."
    exit /b 0
)

echo Extracting rubberband.exe and sndfile.dll to %ROOT_DIR%...
powershell -NoProfile -ExecutionPolicy Bypass -Command "Add-Type -AssemblyName System.IO.Compression.FileSystem; $archive = [System.IO.Compression.ZipFile]::OpenRead('%RB_ZIP%'); foreach ($entry in $archive.Entries) { if ($entry.Name -eq 'rubberband.exe' -or $entry.Name -eq 'sndfile.dll') { [System.IO.Compression.ZipFileExtensions]::ExtractToFile($entry, (Join-Path '%ROOT_DIR%' $entry.Name), $true) } }; $archive.Dispose()"
if exist "%RB_ZIP%" del "%RB_ZIP%" >nul 2>&1

if exist "%ROOT_DIR%\rubberband.exe" (
    echo [OK] rubberband-cli portable installed successfully: %ROOT_DIR%\rubberband.exe
    call :log "rubberband-cli portable installed to %ROOT_DIR%\rubberband.exe"
) else (
    call :warn "Rubber Band archive downloaded, but rubberband.exe could not be extracted."
)
exit /b 0

:smoke_test
echo.
echo Running smoke test...
"%VENV_PYTHON%" -c "import sys, torch; print('Python executable:', sys.executable); print('Python:', sys.version.split()[0]); print('Torch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA build:', torch.version.cuda); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU mode')"
if errorlevel 1 (
    call :fail "PyTorch smoke test failed."
    exit /b 1
)

"%VENV_PYTHON%" -c "import onnxruntime as ort; print('ONNX Runtime providers:', ort.get_available_providers())"
if errorlevel 1 call :warn "ONNX Runtime smoke test failed. UVR can still run non-ONNX models."

if /i "%INSTALL_MODE%"=="NVIDIA" (
    "%VENV_PYTHON%" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 2)"
    if errorlevel 1 (
        call :warn "NVIDIA mode is installed, but torch.cuda.is_available() is still False."
        call :warn "Likely causes: outdated driver, DLL conflict, or incompatible PyTorch wheel."
    )
)
exit /b 0

:done
echo.
echo ============================================================
echo  Installation complete.
echo ============================================================
echo  Mode : %INSTALL_MODE%
echo  Log  : %LOG_FILE%
echo.
echo Run UVR with: run_uvr.bat
echo.
call :prompt_choice RUN_NOW "Run run_uvr.bat now? [Y/N, default N]: " "Y N" "N" "Y or N"
if /i "!RUN_NOW!"=="Y" call ../run_uvr.bat
pause
exit /b 0

:confirm_risky
echo.
call :warn "%~1"
call :prompt_choice RISK_OK "Continue anyway? [Y/N, default N]: " "Y N" "N" "Y or N"
if /i "!RISK_OK!"=="N" (
    echo Switching to CPU fallback.
    set "INSTALL_MODE=CPU"
)
exit /b 0

:prompt_choice
setlocal EnableDelayedExpansion
set "PROMPT_OPTIONS=%~3"
set "PROMPT_DEFAULT=%~4"
set "PROMPT_HINT=%~5"

:prompt_choice_retry
set "PROMPT_VALUE="
set /p "PROMPT_VALUE=%~2"
if not defined PROMPT_VALUE (
    if defined PROMPT_DEFAULT (
        set "PROMPT_VALUE=!PROMPT_DEFAULT!"
    ) else (
        call :warn "Input is required. Enter !PROMPT_HINT! exactly."
        goto :prompt_choice_retry
    )
)

set "PROMPT_MATCH="
for %%O in (!PROMPT_OPTIONS!) do (
    if /i "!PROMPT_VALUE!"=="%%O" set "PROMPT_MATCH=%%O"
)

if not defined PROMPT_MATCH (
    if defined PROMPT_DEFAULT (
        call :warn "Invalid input. Enter !PROMPT_HINT! exactly, or press Enter for default [!PROMPT_DEFAULT!]."
    ) else (
        call :warn "Invalid input. Enter !PROMPT_HINT! exactly."
    )
    goto :prompt_choice_retry
)

endlocal & set "%~1=%PROMPT_MATCH%"
exit /b 0

:warn
echo [WARNING] %~1
call :log "[WARNING] %~1"
exit /b 0

:fail
echo.
echo [ERROR] %~1
call :log "[ERROR] %~1"
echo.
echo Installer stopped. Log: %LOG_FILE%
pause
exit /b 1

:log
>>"%LOG_FILE%" echo [%DATE% %TIME%] %~1
exit /b 0
