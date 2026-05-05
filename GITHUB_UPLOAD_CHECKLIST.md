# 🚀 GitHub Releasing Checklist (Gedächtnisstütze)

Wenn du das Projekt auf GitHub als Open Source veröffentlichst, halte dich strikt an diese Trennung, um Serverprobleme (z.B. >1 GB Repositories) bei GitHub zu vermeiden!

## 1. Das Code-Repository (git push)
Das Repository (Source Code) sollte federleicht bleiben. Hier lädst du **NUR den reinen Code und die Dokumentation** hoch (**KEINE** ML-Modelle oder Portable Python!).

**✅ Diese Dateien gehören ins GitHub-Repository:**
* `nmr_app.py`
* `README.md`
* `RELEASE_NOTES.md`
* `LICENSE`
* `create_portable_python.bat`
* `Start_NMR_App.bat`
* `build_installer.iss`
* Den leeren Ordner `models/` (Lege am besten eine winzige Datei namens `.gitkeep` hinein).
* Den Ordner `ketcher/` (Sicher im Repo, da es nur Textdateien sind).

**❌ Diese Dateien dürfen NIEMALS in die Quellcode-Synchronisation (mache ggf. eine .gitignore):**
* Die dicken Machine Learning-Dateien aus `models/` (besonders die riesige Dateigrößen wie `best_model.hdf5` und `DLNMR1.pt`). GitHub blockiert Commits über 100MB ohnehin!
* Der Ordner `portable_python/` (Der sprengt dein Repository komplett!).
* Eventuell vorhandene lokale Umgebungen (`venv/`).

---

## 2. Das "Release" (Der Bereich für Endnutzer-Downloads)
Wenn Nutzer auf deiner GitHub-Seite ganz rechts auf **"Releases"** klicken, sollen sie das kompilierte Produkt herunterladen können. Dort gibt es keine Platzangst für den Dateiupload.

**✅ Diese Dateien machst du als Release-Asset zum direkten Download verfügbar:**
1. **`NMR_App_Setup.exe`:** (Sehr empfehlenswert!) Diese installierbare Exe wird über `build_installer.iss` (Inno Setup) generiert. Sie packt Code, Portable Python und die gigantischen ML-Modelle in _eine dicke, bequeme Windows-Setup-Datei_.
2. (Optional) **`NMR_App_Portable.zip`:** Ein einfaches manuell erstelltes ZIP-Archiv deines Projektordners (inkl. `portable_python` und aller Modelle).

### Workflow vor einem großen Release
Immer wenn du neue Features (z.B. Multithreading) gepusht hast und ein neues Update freigeben willst:
1. `create_portable_python.bat` einmal lokal ausführen (falls nicht ohnehin `portable_python/` noch da ist).
2. Sicherstellen, dass die ML-Gewichte in `models/` vollständig bereitliegen.
3. Rechtsklick auf `build_installer.iss` -> *Compile* (Erfordert vorherige Inno Setup Installation).
4. Das Resultat `NMR_App_Setup.exe` laden und bei GitHub unter einem neuen Release (z.B. "Version 1.0") anhängen!
