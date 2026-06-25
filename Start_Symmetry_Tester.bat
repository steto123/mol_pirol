@echo off
REM Start the NMR Symmetry Tester App
SET ROOT_DIR=%~dp0
CD /D "%ROOT_DIR%"
IF EXIST venv\Scripts\pythonw.exe (
    START "" venv\Scripts\pythonw.exe symmetry_tester.py
) ELSE (
    START "" pythonw symmetry_tester.py
)
EXIT
