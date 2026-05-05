# NMR 13C Prediction App - Installation & Usage / Installation & Nutzung

## 🇬🇧 English

### Prerequisites
- Python 3.8 to 3.11 installed. (Make sure Python is added to your PATH).
- Windows OS (Optimized for Windows 11).

### Installation (Automatic)
1. Double-click the `install.bat` file.
2. This will create a virtual environment (`venv`) and install all required packages using pip.
3. Wait for the installation to finish.

### Installation (Manual via Command Prompt)
1. Open up a terminal/cmd in the application directory.
2. Create a virtual environment: `python -m venv venv`
3. Activate the environment: `venv\Scripts\activate`
4. Install all dependencies: `pip install pandas numpy torch PyQt5 rdkit tf-keras nfp scipy scikit-learn`
5. (Note: Make sure the `models` folder is located in the application directory or its parent directory, as expected by the script).

### Usage
1. Simply double-click `start.bat`.
2. This will automatically activate the Python environment and start the Graphic User Interface.

---

## 🇩🇪 Deutsch

### Voraussetzungen
- Python 3.8 bis 3.11 installiert. (Stellen Sie sicher, dass Python bei der Installation zu den Umgebungsvariablen (PATH) hinzugefügt wurde).
- Windows Betriebssystem (Optimiert für Windows 11).

### Installation (Automatisch)
1. Doppelklicken Sie auf die Datei `install.bat`.
2. Dies erstellt eine virtuelle Umgebung (`venv`) und installiert alle benötigten Pakete über die Paketverwaltung `pip`.
3. Warten Sie, bis die Installation abgeschlossen ist. Das Fenster pausiert am Ende.

### Installation (Manuell via Eingabeaufforderung)
1. Öffnen Sie ein Terminal (CMD/PowerShell) im Verzeichnis der Anwendung.
2. Erstellen Sie eine virtuelle Umgebung: `python -m venv venv`
3. Aktivieren Sie die Umgebung: `venv\Scripts\activate`
4. Installieren Sie alle Abhängigkeiten: `pip install pandas numpy torch PyQt5 rdkit tf-keras nfp scipy scikit-learn`
5. (Hinweis: Stellen Sie sicher, dass sich der Ordner `models` im Verzeichnis der Anwendung oder dem übergeordneten Ordner befindet, damit die Modelle geladen werden können).

### Nutzung
1. Doppelklicken Sie einfach auf `start.bat`.
2. Dies aktiviert automatisch die virtuelle Umgebung und startet die grafische Benutzeroberfläche der Anwendung.
