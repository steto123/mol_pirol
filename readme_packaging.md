# Packaging Instructions for NMR Predictor

This directory contains scripts to automate the creation of the Windows Installer using Inno Setup.

## Prerequisites
1. **Inno Setup 6** must be installed on your system.
   - Default path expected: `C:\Program Files (x86)\Inno Setup 6\ISCC.exe`
   - If installed elsewhere, update the path in `build_installer.bat`.

## Included Scripts

| Script | Description |
| :--- | :--- |
| `cleanup_before_package.bat` | Removes `__pycache__`, temporary files, and previous build output. Use this to keep the installer size small. |
| `build_installer.bat` | Compiles the `build_installer.iss` script to generate the setup EXE. |
| `FULL_REBUILD_INSTALLER.bat` | **Recommended.** Runs cleanup followed by the build process in one go. |

## Important Configuration Notes
- **Non-Admin Rights:** The installer is configured with `PrivilegesRequired=lowest`. This means:
  - If run as a normal user, it installs to `%LOCALAPPDATA%\Programs\NMR_Predictor` (no admin password required).
  - If run as admin, it installs to `C:\Program Files\NMR_Predictor`.
- **Portable Python:** The script assumes the `portable_python` folder is present in the root directory and contains the necessary environment.

## Troubleshooting
If the build fails, check if all source files (like `nmr_app.py`, `models/`, `ketcher/`, etc.) are in the same folder as the `.iss` file.
