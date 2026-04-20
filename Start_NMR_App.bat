@echo off
cd /d "%~dp0"

:: Versuche zuerst die lokale VENV Umgebung
IF EXIST "venv\Scripts\pythonw.exe" (
    start "" "venv\Scripts\pythonw.exe" "nmr_app.py"
    exit /b
)

:: Wenn VENV nicht da ist, versuche Portable Python (für Nutzer, die das Github Release laden)
IF EXIST "portable_python\WPy64-31180\python-3.11.8.amd64\pythonw.exe" (
    start "" "portable_python\WPy64-31180\python-3.11.8.amd64\pythonw.exe" "nmr_app.py"
    exit /b
)

echo Weder "venv" noch "portable_python" wurden gefunden!
pause
