@echo off
echo ======================================================
echo FULL REBUILD: NMR Predictor Installer
echo ======================================================

call cleanup_before_package.bat
echo.
call build_installer.bat

echo.
echo ======================================================
echo Rebuild Process Finished.
echo ======================================================
pause
