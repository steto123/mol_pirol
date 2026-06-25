@echo off
setlocal enabledelayedexpansion

:: Configuration
set "INNO_SETUP_PATH=C:\Program Files (x86)\Inno Setup 6\ISCC.exe"
set "ISS_FILE=build_installer.iss"
set "OUTPUT_DIR=Output"

echo ======================================================
echo NMR Predictor - Inno Setup Builder
echo ======================================================

:: Check if Inno Setup is installed
if not exist "%INNO_SETUP_PATH%" (
    echo [ERROR] Inno Setup compiler not found at:
    echo "%INNO_SETUP_PATH%"
    echo Please install Inno Setup 6 or update the path in this script.
    pause
    exit /b 1
)

:: Check if .iss file exists
if not exist "%ISS_FILE%" (
    echo [ERROR] Inno Setup script not found: %ISS_FILE%
    pause
    exit /b 1
)

:: Create Output directory if it doesn't exist
if not exist "%OUTPUT_DIR%" (
    mkdir "%OUTPUT_DIR%"
)

echo [INFO] Starting compilation of %ISS_FILE%...
"%INNO_SETUP_PATH%" "%ISS_FILE%"

if %ERRORLEVEL% EQU 0 (
    echo.
    echo [SUCCESS] Installer has been created successfully!
    echo [INFO] You can find the setup file in the "%OUTPUT_DIR%" folder.
) else (
    echo.
    echo [ERROR] Inno Setup compilation failed.
)

pause
