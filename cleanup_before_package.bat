@echo off
echo ======================================================
echo Cleaning project for packaging...
echo ======================================================

:: Remove __pycache__ directories
echo [INFO] Removing __pycache__...
for /d /r . %%d in (__pycache__) do (
    if exist "%%d" (
        echo Deleting "%%d"
        rd /s /q "%%d"
    )
)

:: Remove Jupyter checkpoints
echo [INFO] Removing .ipynb_checkpoints...
for /d /r . %%d in (.ipynb_checkpoints) do (
    if exist "%%d" (
        echo Deleting "%%d"
        rd /s /q "%%d"
    )
)

:: Remove previous Output folder
if exist "Output" (
    echo [INFO] Removing previous Output folder...
    rd /s /q "Output"
)

:: Remove log files or temp html files if any
del /s /q *.log 2>nul
del /s /q *.tmp 2>nul

echo.
echo [SUCCESS] Project cleaned!
echo.
