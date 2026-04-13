@echo off
echo ==============================================================
echo [EN] Starting Installation of NMR Calculator Requirements
echo [DE] Starte Installation der NMR Calculator Abhaengigkeiten
echo ==============================================================
echo.

echo [EN] Creating virtual environment (venv)...
echo [DE] Erstelle virtuelle Umgebung (venv)...
python -m venv venv

echo.
echo [EN] Activating virtual environment...
echo [DE] Aktiviere virtuelle Umgebung...
call venv\Scripts\activate.bat

echo.
echo [EN] Installing required packages. This may take a while...
echo [DE] Installiere benoetigte Pakete. Dies kann eine Weile dauern...
python -m pip install --upgrade pip
pip install pandas numpy torch PyQt5 rdkit tf-keras nfp scipy scikit-learn

echo.
echo ==============================================================
echo [EN] Installation finished successfully! You can now run start.bat
echo [DE] Installation erfolgreich abgeschlossen! Sie koennen nun start.bat ausfuehren
echo ==============================================================
pause
