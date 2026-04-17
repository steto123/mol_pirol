# Release Notes - NMR 13C Prediction App

## Version 0.2 (2026-04-16)

### Localization / Translations
* **English GUI**: All user interface elements including buttons, input fields, labels, and table headers ("Atom Index", "CASCADE", "EST-NMR", "DCode", "Range") have been translated to English.
* **System Messages**: Console outputs, error dialogs, and internal status updates are now completely in English.

### New Features & Enhancements
* **Boltzmann Weighting for EST-NMR**: Analogous to the DCode topology algorithm, the EST-NMR prediction engine now utilizes a full Boltzmann-weighting approach. The algorithm optionally generates 10 conformers, minimizes their energies via the MMFF94 force field, and aggregates the PyTorch predictions into a final Boltzmann-weighted result. The UI table now displays both `EST-NMR` (Single) and `EST-NMR (Boltz)` side-by-side.
* **Interactive 2D Molecule View**: Upgraded the static molecule image to a dynamic, interactive canvas. Users can now zoom in/out with the mouse wheel and pan the structure by clicking and dragging.
* **True 3D Conformer Visualization**: Integrated a toggleable "3D View" mode. Checking the box switches the 2D view to a fully interactive 3D model (using `3Dmol.js`), allowing users to freely rotate, zoom, and inspect the calculated 3D conformer directly in the app. *(Requires `PyQtWebEngine`)*.
* **CASCADE Element Warning**: Implemented a dynamic warning label below the results table. The warning ("Warning: CASCADE is only trained for elements C, H, N, O, S, P, F, Cl.") automatically appears if the inputted molecule contains any unsupported elements, safeguarding users against potentially inaccurate CASCADE model predictions.
* **Status Bar Integration**: Added a status bar at the bottom of the application window. It provides real-time, non-intrusive feedback regarding the application's current state (e.g., "Ready", "Calculation running...", "Calculation finished.").
* **Seamless Model Loading**: The initial delay caused by loading ML models (which originally triggered an interruptive pop-up dialog requiring the user to click "OK") has been moved to the new status bar.

### UX Improvements
* **Cleaner Tables**: The redundant consecutive row numbering (vertical table header) has been hidden from the output table to provide a cleaner and more focused layout.
