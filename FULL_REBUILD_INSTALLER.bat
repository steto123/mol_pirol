@echo off
setlocal enabledelayedexpansion

:: Immer im Verzeichnis des Scripts ausfuehren
cd /d "%~dp0"

echo ======================================================
echo  FULL REBUILD: NMR 13C Predictor Installer
echo ======================================================
echo.

:: -------------------------------------------------------
:: 1. Icons pruefen
:: -------------------------------------------------------
echo [CHECK] Icons...

if exist "app_icon.ico" goto :ico_ok
echo [ERROR] app_icon.ico nicht gefunden!
echo         Bitte zuerst das Icon erstellen:
echo         venv\Scripts\python.exe scratch\convert_icon.py
pause
exit /b 1
:ico_ok
echo        app_icon.ico    OK

if exist "app_icon.png" goto :png_ok
echo [ERROR] app_icon.png nicht gefunden!
pause
exit /b 1
:png_ok
echo        app_icon.png    OK
echo.

:: -------------------------------------------------------
:: 2. Quelldateien pruefen
:: -------------------------------------------------------
echo [CHECK] Quelldateien...

if exist "nmr_app.py"           goto :f1
echo [ERROR] nmr_app.py nicht gefunden & pause & exit /b 1
:f1
echo        nmr_app.py            OK

if exist "symmetry_ranking.py"  goto :f2
echo [ERROR] symmetry_ranking.py nicht gefunden & pause & exit /b 1
:f2
echo        symmetry_ranking.py   OK

if exist "symmetry_tester.py"   goto :f3
echo [ERROR] symmetry_tester.py nicht gefunden & pause & exit /b 1
:f3
echo        symmetry_tester.py    OK

if exist "Start_NMR_App.bat"    goto :f4
echo [ERROR] Start_NMR_App.bat nicht gefunden & pause & exit /b 1
:f4
echo        Start_NMR_App.bat     OK

if exist "build_installer.iss"  goto :f5
echo [ERROR] build_installer.iss nicht gefunden & pause & exit /b 1
:f5
echo        build_installer.iss   OK
echo.

:: -------------------------------------------------------
:: 3. Inno Setup pruefen
::    Hinweis: Pfad mit (x86) darf NICHT in if (...)-Bloecken
::    stehen -> goto-Muster verwenden
:: -------------------------------------------------------
set ISCC="C:\Program Files (x86)\Inno Setup 6\ISCC.exe"
echo [CHECK] Inno Setup 6...

if exist %ISCC% goto :inno_ok
echo [ERROR] Inno Setup 6 nicht gefunden!
echo         Erwartet: C:\Program Files (x86)\Inno Setup 6\ISCC.exe
echo         Download: https://jrsoftware.org/isdl.php
pause
exit /b 1
:inno_ok
echo        Inno Setup 6    OK
echo.

:: -------------------------------------------------------
:: 4. Cleanup
:: -------------------------------------------------------
echo [STEP 1/2] Cleanup laeuft...
call cleanup_before_package.bat
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Cleanup fehlgeschlagen.
    pause
    exit /b 1
)
echo.

:: -------------------------------------------------------
:: 5. Installer bauen
:: -------------------------------------------------------
echo [STEP 2/2] Inno Setup kompiliert build_installer.iss...
echo.
%ISCC% "build_installer.iss"
set BUILD_ERR=%ERRORLEVEL%
echo.

if %BUILD_ERR% EQU 0 goto :build_ok
echo ======================================================
echo  FEHLER: Inno Setup Kompilierung fehlgeschlagen
echo  Exit-Code: %BUILD_ERR%
echo ======================================================
echo.
pause
exit /b 1

:build_ok
echo ======================================================
echo  ERFOLG: Installer wurde erstellt!
echo  Ausgabepfad: Output\
echo ======================================================
for %%F in ("Output\*.exe") do (
    echo  ^> %%~nxF  [%%~zF Bytes]
)
echo.
pause
