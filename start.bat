@echo off
echo ==============================================================
echo [EN] Starting NMR Calculator...
echo [DE] Starte NMR Calculator...
echo ==============================================================

if not exist venv\Scripts\python.exe (
    echo [EN] Error: Virtual environment not found. Please run install.bat first.
    echo [DE] Fehler: Virtuelle Umgebung nicht gefunden. Bitte zuerst install.bat ausfuehren.
    pause
    exit /b
)

venv\Scripts\python.exe nmr_app.py

pause
