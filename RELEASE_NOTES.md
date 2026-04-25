# Release Notes - NMR 13C Prediction App

## Version 0.3 (2026-04-24)

### New Experimental Feature
* **Hybrid Symmetry Averaging**: Introduced a sophisticated atom-ranking algorithm that improves chemical consistency in predictions.
    * **3D Refinement**: Unlike standard topological ranking, this algorithm analyzes the 3D geometry (distance sets) to distinguish between cis/trans isomers and axial/equatorial positions.
    * **Tolerant Grouping**: Uses a 0.4 Å spatial tolerance to ensure that rotating groups (like phenyl rings or methyl groups) are still correctly identified as equivalent despite minor conformational fluctuations.
    * **User Control**: A new "Symmetry Average" checkbox in the UI allows users to toggle this behavior. When enabled, it averages the predictions of all models across equivalent atoms.

## Version 0.2 (2026-04-16)

### Localization / Translations
* **English GUI**: All user interface elements including buttons, input fields, labels, and table headers ("Atom Index", "CASCADE", "EST-NMR", "DCode", "Range") have been translated to English.
* **System Messages**: Console outputs, error dialogs, and internal status updates are now completely in English.

### New Features & Enhancements
* **Boltzmann Weighting for EST-NMR**: Analogous to the DCode topology algorithm, the EST-NMR prediction engine now utilizes a full Boltzmann-weighting approach. The algorithm optionally generates 10 conformers, minimizes their energies via the MMFF94 force field, and aggregates the PyTorch predictions into a final Boltzmann-weighted result. The UI table now displays both `EST-NMR` (Single) and `EST-NMR (Boltz)` side-by-side.
* **Interactive 2D Molecule View**: Upgraded the static molecule image to a dynamic, interactive canvas. Users can now zoom in/out with the mouse wheel and pan the structure by clicking and dragging.
* **Simulated Analytical Spectrum**: Added an interactive "Spectrum" tab powered by matplotlib that synthesizes a 1D 13C-NMR spectrum diagram. Peak picking allows users to click directly on graphical Lorentzian peaks to highlight the corresponding carbon atoms in the table and viewer.
* **Auto-Assignment & Experimental Validation**: Shipped an "Exp. Data" input field. The application processes user-supplied laboratory peaks and performs automatic assignment via a **Consensus-Ranking Algorithm** (it averages the predictions across CASCADE, EST-NMR and DCode to rank the carbon atoms, and iteratively assigns the sorted experimental values to them). It renders a dedicated Mean Absolute Error (MAE) benchmarking metric for the models directly into the UI.
* **Smart Session Caching**: An intelligently managed 'History' dropdown caches entire calculated chemical structures and their tensors into RAM during a session. Reviewing previous predictive molecules happens instantaneously without invoking recalculations.
* **Responsive Background Calculus**: Reprogrammed execution to operate non-blockingly. Deep learning logic now natively functions out of `QThread` workers, allowing the UI to remain entirely responsive and providing granular progress feedback.
* **Offline Ketcher Tool**: Packaged EPAM's [Ketcher](https://github.com/epam/ketcher) engine directly into the QtWebEngine. The GUI now features a "🖌 Draw" button enabling full drawing and automated SMILES code conversion without relying on any external internet access.
* **Report Export Capabilities**: Introduced an "Export Report" function. Users can export their analysis into an off-line CSV file or dump a highly-formatted, standalone HTML report stringing together the drawn 2D atomic SVG, smiles, metadata, and numerical results.
* **True 3D Conformer Visualization**: Integrated a toggleable "3D View" mode. Checking the box switches the 2D view to a fully interactive 3D model (using `3Dmol.js`), allowing users to freely rotate, zoom, and inspect the calculated 3D conformer directly in the app. *(Requires `PyQtWebEngine`)*.
* **CASCADE Element Warning**: Implemented a dynamic warning label below the results table. The warning ("Warning: CASCADE is only trained for elements C, H, N, O, S, P, F, Cl.") automatically appears if the inputted molecule contains any unsupported elements, safeguarding users against potentially inaccurate CASCADE model predictions.
* **Status Bar Integration**: Added a status bar at the bottom of the application window. It provides real-time, non-intrusive feedback regarding the application's current state (e.g., "Ready", "Calculation running...", "Calculation finished.").
* **Seamless Model Loading**: The initial delay caused by loading ML models (which originally triggered an interruptive pop-up dialog requiring the user to click "OK") has been moved to the new status bar.

### UX Improvements
* **Cleaner Tables**: The redundant consecutive row numbering (vertical table header) has been hidden from the output table to provide a cleaner and more focused layout.
