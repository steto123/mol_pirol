@echo off
echo ==============================================================
echo [EN] Starting NMR Calculator...
echo [DE] Starte NMR Calculator...
echo ==============================================================

if not exist venv\Scripts\activate.bat (
    echo [EN] Error: Virtual environment not found. Please run install.bat first.
    echo [DE] Fehler: Virtuelle Umgebung nicht gefunden. Bitte zuerst install.bat ausfuehren.
    pause
    exit /b
)

call venv\Scripts\activate.bat
python nmr_app.py

pause
