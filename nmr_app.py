"""
NMR 13C Prediction App
Diese GUI-Anwendung kombiniert drei ML/Topologie-Modelle (CASCADE, EST-NMR, DCode),
um die chemischen C-13 NMR-Verschiebungen aus einem SMILES Code zu prognostizieren und zu vergleichen.
"""
import os
import sys

_NOTEBOOK_DIR = os.path.dirname(os.path.abspath(__file__))
if _NOTEBOOK_DIR not in sys.path:
    sys.path.insert(0, _NOTEBOOK_DIR)

import logging
import warnings
import pandas as pd
import numpy as np
import math
import pickle
import torch
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QLineEdit, QPushButton, 
                             QTableWidget, QTableWidgetItem, QMessageBox, QHeaderView, QSplitter,
                             QGraphicsView, QGraphicsScene, QStackedWidget, QCheckBox)
from PyQt5.QtCore import Qt
try:
    from PyQt5.QtWebEngineWidgets import QWebEngineView
    WEB_ENGINE_AVAILABLE = True
except ImportError:
    QWebEngineView = QWidget
    WEB_ENGINE_AVAILABLE = False

import json

HTML_3DMOL = """
<!DOCTYPE html>
<html>
<head>
  <script src="https://3Dmol.csb.pitt.edu/build/3Dmol-min.js"></script>
  <style>
    body { margin: 0; padding: 0; overflow: hidden; background-color: white;}
    #container { width: 100vw; height: 100vh; position: relative;}
  </style>
</head>
<body>
  <div id="container"></div>
  <script>
    var viewer = $3Dmol.createViewer("container", {backgroundColor: "white"});
    function loadMolecule(molBlock) {
        viewer.clear();
        viewer.addModel(molBlock, "mol");
        viewer.setStyle({}, {stick: {radius: 0.15}, sphere: {scale: 0.3}});
        viewer.zoomTo();
        viewer.render();
    }
  </script>
</body>
</html>
"""
from PyQt5.QtSvg import QSvgWidget, QGraphicsSvgItem, QSvgRenderer
from PyQt5.QtGui import QFont, QColor, QPainter

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D

try:
    from dcode.geometry import DCodeName
    from dcode.tools import DCodeMol
    from dcode.calcshift import calcshift
except ImportError as e:
    print(f"Error importing DCode libraries: {e}")


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning, module="tf_keras")
warnings.filterwarnings("ignore", category=FutureWarning)

_NOTEBOOK_DIR = os.path.dirname(os.path.abspath(__file__))
_models_dir = os.path.join(_NOTEBOOK_DIR, "models")
if not os.path.exists(_models_dir):
    _models_dir = os.path.join(os.path.dirname(_NOTEBOOK_DIR), "models")

if os.path.exists(_models_dir):
    if _models_dir not in sys.path:
        sys.path.insert(0, _models_dir)

try:
    import tf_keras as keras
    from tf_keras.models import load_model
    from nfp.layers import (
        MessageLayer, GRUStep, Squeeze, EdgeNetwork,
        ReduceBondToPro, ReduceBondToAtom, GatherAtomToBond, ReduceAtomToPro,
    )
    from nfp.models import GraphModel
    from cascade.apply import preprocess_C, evaluate_C
except ImportError as e:
    print(f"Error importing ML libraries: {e}")
    sys.exit(1)

_CUSTOM_OBJECTS = {
    "GraphModel": GraphModel,
    "ReduceAtomToPro": ReduceAtomToPro,
    "Squeeze": Squeeze,
    "GatherAtomToBond": GatherAtomToBond,
    "ReduceBondToAtom": ReduceBondToAtom,
}

# Globale Variablen für Modelle (werden beim Start geladen)
NMR_model_C = None
NMR_model_E = None
codes_df = None

def init_models():
    """
    Lädt alle benötigten Modelle und Datenbanken verzögert (lazy loading) beim ersten Klick auf 'Berechnen'.
    - Lade Tensorflow/Keras Cascade-Model.
    - Lade EST-NMR Torch-Model.
    - Längste Ladezeit: Einlesen der extrem großen DCode CSV in pandas.
    """
    global NMR_model_C, NMR_model_E, codes_df
    try:
        print("Loading models...")
        
        # 1. CASCADE Lade-Routinen
        modelpath_C = os.path.join(_models_dir, "cascade", "trained_model", "best_model.hdf5")
        NMR_model_C = load_model(modelpath_C, custom_objects=_CUSTOM_OBJECTS)
        
        # 2. EST-NMR PyTorch-Model Laden
        modelpath_E = os.path.join(_models_dir, "DLNMR1.pt")
        NMR_model_E = torch.jit.load(modelpath_E)
        
        # 3. DCode Pandas DataFrame importieren (> 40MB Datensatz)
        codefile = os.path.join(_NOTEBOOK_DIR, "codes", "v3_update_23_10_2025.csv")
        if os.path.exists(codefile):
            codes_df = pd.read_csv(codefile, dtype={0: str})
        else:
            print(f"WARNING: DCode code file not found: {codefile}")
            
        print("Models loaded successfully!")
    except Exception as e:
        print(f"Error loading models: {e}")

def predict_cascade(mol, model, models_dir):
    """
    Graph Neural Network Vorhersage über CASCADE.
    Verwendet den internen 'preprocessor' um den molekularen Graphen aufzubauen
    und berechnet das Ergebnis konformer-gewichtet.
    """
    _CASCADE_DIR = os.path.join(models_dir, "cascade")
    preprocessor_path = os.path.join(_CASCADE_DIR, 'preprocessor.p')
    with open(preprocessor_path, 'rb') as ft:
        preprocessor = pickle.load(ft)['preprocessor']
    
    # Preprocessing baut Konformere und Kanten-Verbindungen auf
    m = Chem.AddHs(mol, addCoords=True)
    inputs, df, mols = preprocess_C([m], preprocessor, keep_all_cf=True)
    if not inputs: return {}
    
    # NN Prediction auslösen
    predicted_values = evaluate_C(inputs, preprocessor, model)
    chunks = []
    for _, r in df.iterrows():
        df_mol = pd.DataFrame({
            'mol_id': [r.mol_id]*len(r.atom_index), 
            'atom_index': r.atom_index, 
            'relative_E': [r.relative_E]*len(r.atom_index), 
            'cf_id': [r.cf_id]*len(r.atom_index)
        })
        chunks.append(df_mol)
    
    if not chunks: return {}
    
    # Daten zusammenführen und Boltzmann-Gewichtung der Kraftfeld Energie errechnen
    spread_df = pd.concat(chunks)
    spread_df['predicted'] = predicted_values
    spread_df['b_weight'] = spread_df.relative_E.apply(lambda x: math.exp(-x/(0.001987*298.15)))
    
    df_group = spread_df.set_index(['mol_id', 'atom_index', 'cf_id']).groupby(level=1)
    # Rückgabe gemittelter Werte
    return {int(a_idx): round(group.apply(lambda x: x['b_weight']*x['predicted'], axis=1).sum()/group.b_weight.sum(), 2) 
            for a_idx, group in df_group}

def predict_est_nmr(mol, model):
    """
    Führt Vorhersage mit EST-NMR basierend auf Thomas Hehre et al. (PyTorch) durch.
    Dieses Tool braucht als primären Input die echten Raumkoordinaten.
    """
    m = Chem.AddHs(mol, addCoords=True)
    # Bettet das Mol ein und holt Geometrie, wenn noch keine generiert wurde
    if m.GetNumConformers() == 0:
        if AllChem.EmbedMolecule(m, randomSeed=42) == -1:
            AllChem.Compute2DCoords(m)
        AllChem.MMFFOptimizeMolecule(m)
        
    conf = m.GetConformer()
    species = [atom.GetAtomicNum() for atom in m.GetAtoms()]
    coords = [[pos.x, pos.y, pos.z] for pos in [conf.GetAtomPosition(i) for i in range(m.GetNumAtoms())]]
    
    # In Tensoren wandeln
    Z = torch.tensor(species, dtype=torch.int64)
    R = torch.tensor(coords, dtype=torch.float32).unsqueeze(0)
    
    with torch.no_grad():
        res = model(Z, R)
    all_shifts = res[2].tolist()
    
    # Rückgabe nur für Kohlenstoff (C = 6)
    return {i: round(all_shifts[i], 2) for i, atom in enumerate(m.GetAtoms()) if atom.GetAtomicNum() == 6}

def predict_est_nmr_boltzmann(mol, model):
    """
    Führt Vorhersage mit EST-NMR basierend auf Thomas Hehre et al. (PyTorch) durch,
    und gewichtet die Ergebnisse über mehrere Konformere (Boltzmann-Verteilung).
    """
    m = Chem.AddHs(mol, addCoords=True)
    if m.GetNumConformers() == 0:
        if AllChem.EmbedMolecule(m, randomSeed=42) == -1:
            AllChem.Compute2DCoords(m)
        AllChem.MMFFOptimizeMolecule(m)
        
    cids = AllChem.EmbedMultipleConfs(m, numConfs=10, randomSeed=42, pruneRmsThresh=0.5)
    
    energies = []
    for cid in cids:
        ff = AllChem.MMFFGetMoleculeForceField(m, AllChem.MMFFGetMoleculeProperties(m), confId=cid)
        if ff:
            ff.Initialize()
            ff.Minimize()
            energies.append((cid, ff.CalcEnergy()))
            
    if not energies:
        return predict_est_nmr(mol, model) # Fallback
        
    min_e = min([e for c, e in energies])
    RT = 0.001987 * 298.15
    b_weights = {cid: math.exp(-(e - min_e)/RT) for cid, e in energies}
    sum_w = sum(b_weights.values())
    b_weights = {cid: w/sum_w for cid, w in b_weights.items()}

    est_sums = {atom.GetIdx(): 0.0 for atom in m.GetAtoms() if atom.GetAtomicNum() == 6}
    weight_sums = {atom.GetIdx(): 0.0 for atom in m.GetAtoms() if atom.GetAtomicNum() == 6}
    
    species = [atom.GetAtomicNum() for atom in m.GetAtoms()]
    Z = torch.tensor(species, dtype=torch.int64)

    for cid, w in b_weights.items():
        conf = m.GetConformer(cid)
        coords = [[pos.x, pos.y, pos.z] for pos in [conf.GetAtomPosition(i) for i in range(m.GetNumAtoms())]]
        R = torch.tensor(coords, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            res = model(Z, R)
        all_shifts = res[2].tolist()
        
        for i, atom in enumerate(m.GetAtoms()):
            if atom.GetAtomicNum() == 6:
                est_sums[i] += w * all_shifts[i]
                weight_sums[i] += w
                
    final_results = {}
    for atom_idx in est_sums:
        if weight_sums[atom_idx] > 0:
            final_results[atom_idx] = round(est_sums[atom_idx] / weight_sums[atom_idx], 2)
        else:
            final_results[atom_idx] = np.nan
            
    return final_results

def predict_dcode_boltzmann(mol, codes_df_input):
    """
    Zieht die Distanz- und Topologie-Logik (DCode) ab und gewichtet
    diese über die Konformere.
    """
    if codes_df_input is None:
        return {}
    
    m = Chem.AddHs(mol, addCoords=True)
    # Um eine sinnvolle DCode Statistik zu erhalten, brauchen wir MMFF Startwerte
    if m.GetNumConformers() == 0:
        if AllChem.EmbedMolecule(m, randomSeed=42) == -1:
            AllChem.Compute2DCoords(m)
        AllChem.MMFFOptimizeMolecule(m)
        
    # Generiere 10 Konformere für den Boltzmann Test
    cids = AllChem.EmbedMultipleConfs(m, numConfs=10, randomSeed=42, pruneRmsThresh=0.5)
    
    energies = []
    # Berechne die absolute Energie aller Konformere im MMFF-Kraftfeld
    for cid in cids:
        ff = AllChem.MMFFGetMoleculeForceField(m, AllChem.MMFFGetMoleculeProperties(m), confId=cid)
        if ff:
            ff.Initialize()
            ff.Minimize()
            energies.append((cid, ff.CalcEnergy()))
            
    if not energies:
        return {}
        
    # Berechne relative Energien (Delta E) zur Min-Energie
    min_e = min([e for c, e in energies])
    # RT berechnen - T = 298.15 K, R = 0.001987 kcal/(mol*K)
    RT = 0.001987 * 298.15
    b_weights = {cid: math.exp(-(e - min_e)/RT) for cid, e in energies}
    sum_w = sum(b_weights.values())
    b_weights = {cid: w/sum_w for cid, w in b_weights.items()}

    # Initiale Summen-Dictionaries
    dcode_sums = {atom.GetIdx(): 0.0 for atom in m.GetAtoms() if atom.GetAtomicNum() == 6}
    weight_sums = {atom.GetIdx(): 0.0 for atom in m.GetAtoms() if atom.GetAtomicNum() == 6}
    
    # Berechne den DCode Ansatz für jedes der 10 Konformere
    for cid, w in b_weights.items():
        single_conf_m = Chem.Mol(m)
        single_conf_m.RemoveAllConformers()
        single_conf_m.AddConformer(m.GetConformer(cid), assignId=True)
        
        # Weise dem Molekül seine topologischen Distanz-Namen zu
        single_conf_m = DCodeName(single_conf_m)
        single_conf_m = DCodeMol(single_conf_m)
        
        for atom in single_conf_m.GetAtoms():
            if atom.GetAtomicNum() == 6 and atom.HasProp('DCode'):
                codestring = atom.GetProp('DCode')
                # Suchstring Abgleich via dcode/calcshift.py (ausgelagert in Pandas)
                verschiebung, treffer, _, _, _ = calcshift(codes_df_input, codestring, atom.GetIdx())
                
                # Ignoriere -999 (nicht gefundene Shifts)
                if verschiebung != -999 and verschiebung != -999.0:
                    dcode_sums[atom.GetIdx()] += w * verschiebung
                    weight_sums[atom.GetIdx()] += w
                    
    final_results = {}
    # Durch den Gewichtungsserver teilen
    for atom_idx in dcode_sums:
        if weight_sums[atom_idx] > 0:
            final_results[atom_idx] = round(dcode_sums[atom_idx] / weight_sums[atom_idx], 2)
        else:
            final_results[atom_idx] = np.nan
            
    return final_results

def draw_annotated_mol(mol):
    """
    Rendert ein Molekül als SVG, wobei die Original-Atom-Indizes
    auch nach dem Entfernen der Wasserstoffatome korrekt erhalten bleiben.
    """
    tm_full = Chem.Mol(mol)
    for i, atom in enumerate(tm_full.GetAtoms()):
        atom.SetIntProp("orig_idx", i)
    
    tm = Chem.RemoveHs(tm_full)
    for atom in tm.GetAtoms():
        atom.SetProp("atomNote", str(atom.GetIntProp("orig_idx")))
    
    AllChem.Compute2DCoords(tm)
    drawer = rdMolDraw2D.MolDraw2DSVG(500, 400)
    opts = drawer.drawOptions()
    opts.addAtomIndices = False
    opts.annotationFontScale = 0.6
    
    drawer.DrawMolecule(tm)
    drawer.FinishDrawing()
    return drawer.GetDrawingText().encode('utf-8')

class InteractiveSvgView(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.svg_item = None
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setRenderHint(QPainter.Antialiasing)
        
    def load(self, svg_bytes):
        self.scene.clear()
        self.svg_item = QGraphicsSvgItem()
        renderer = QSvgRenderer(svg_bytes)
        self.svg_item.setSharedRenderer(renderer)
        self.scene.addItem(self.svg_item)
        # Reset scale to ensure standard view before fitting
        self.resetTransform()
        self.fitInView(self.scene.itemsBoundingRect(), Qt.KeepAspectRatio)

    def wheelEvent(self, event):
        zoom_in_factor = 1.15
        zoom_out_factor = 1 / zoom_in_factor
        if event.angleDelta().y() > 0:
            zoom_factor = zoom_in_factor
        else:
            zoom_factor = zoom_out_factor
        self.scale(zoom_factor, zoom_factor)

class NMRApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NMR 13C Prediction App (Windows 11)")
        self.resize(1000, 600)
        self.setup_ui()

    def setup_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        
        # Eingabe-Bereich
        input_layout = QHBoxLayout()
        self.smiles_label = QLabel("SMILES:")
        self.smiles_label.setFont(QFont("Segoe UI", 11))
        
        self.smiles_input = QLineEdit()
        self.smiles_input.setFont(QFont("Segoe UI", 11))
        self.smiles_input.setPlaceholderText("Enter SMILES code here (e.g. c1ccccc1)")
        
        self.calc_button = QPushButton("Calculate")
        self.calc_button.setFont(QFont("Segoe UI", 11, QFont.Bold))
        self.calc_button.setStyleSheet("background-color: #0078D4; color: white; padding: 5px 15px; border-radius: 4px;")
        self.calc_button.clicked.connect(self.run_analysis)
        
        self.toggle_3d_cb = QCheckBox("3D View")
        self.toggle_3d_cb.setFont(QFont("Segoe UI", 11))
        self.toggle_3d_cb.toggled.connect(self.toggle_view)
        
        input_layout.addWidget(self.smiles_label)
        input_layout.addWidget(self.smiles_input)
        input_layout.addWidget(self.calc_button)
        input_layout.addWidget(self.toggle_3d_cb)
        
        main_layout.addLayout(input_layout)
        
        # Splitter für Bild und Tabelle
        splitter = QSplitter(Qt.Horizontal)
        
        self.stacked_widget = QStackedWidget()
        
        # Interaktive 2D SVG Bildanzeige (Index 0)
        self.svg_widget = InteractiveSvgView()
        self.svg_widget.setMinimumWidth(400)
        self.stacked_widget.addWidget(self.svg_widget)
        
        # 3D View (Index 1) falls verfügbar
        if WEB_ENGINE_AVAILABLE:
            self.web_view = QWebEngineView()
            self.web_view.setHtml(HTML_3DMOL)
            self.stacked_widget.addWidget(self.web_view)
            
        splitter.addWidget(self.stacked_widget)
        
        # Tabelle für Ergebnisse
        self.table = QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels(["Atom Index", "CASCADE", "EST-NMR", "EST-NMR (Boltz)", "DCode", "Range"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setFont(QFont("Segoe UI", 10))
        splitter.addWidget(self.table)
        
        main_layout.addWidget(splitter)
        
        self.warning_label = QLabel("Warning: CASCADE is only trained for elements C, H, N, O, S, P, F, Cl.")
        self.warning_label.setFont(QFont("Segoe UI", 10, QFont.Bold))
        self.warning_label.setStyleSheet("color: red;")
        self.warning_label.setVisible(False)
        main_layout.addWidget(self.warning_label)
        
        self.statusBar().showMessage("Ready")
        
    def toggle_view(self, state):
        if state:
            if WEB_ENGINE_AVAILABLE:
                self.stacked_widget.setCurrentIndex(1)
            else:
                QMessageBox.warning(self, "Missing Dependency", "PyQtWebEngine is not installed.\nPlease run 'pip install PyQtWebEngine' to use the 3D view.")
                self.toggle_3d_cb.setChecked(False)
        else:
            self.stacked_widget.setCurrentIndex(0)
        
    def run_analysis(self):
        smiles = self.smiles_input.text().strip()
        if not smiles:
            QMessageBox.warning(self, "Error", "Please enter a SMILES code!")
            return
            
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            QMessageBox.warning(self, "Error", "Invalid SMILES code!")
            return
            
        allowed_elements = {6, 1, 7, 8, 16, 15, 9, 17} # C, H, N, O, S, P, F, Cl
        molecule_elements = {atom.GetAtomicNum() for atom in mol.GetAtoms()}
        if molecule_elements - allowed_elements:
            self.warning_label.setVisible(True)
        else:
            self.warning_label.setVisible(False)
            
        self.calc_button.setEnabled(False)
        self.calc_button.setText("Calculating...")
        self.statusBar().showMessage("Calculation running...")
        QApplication.processEvents()
        
        try:
            # SVG zeichnen
            svg_bytes = draw_annotated_mol(mol)
            self.svg_widget.load(svg_bytes)
            
            # 3D Daten für Web Engine übergeben, falls WebEngine verfügbar
            if getattr(self, 'web_view', None) is not None:
                m_3d = Chem.AddHs(mol, addCoords=True)
                if AllChem.EmbedMolecule(m_3d, randomSeed=42) == -1:
                    AllChem.Compute2DCoords(m_3d)
                AllChem.MMFFOptimizeMolecule(m_3d)
                mol_block = Chem.MolToMolBlock(m_3d)
                js_code = f"if(typeof loadMolecule !== 'undefined') loadMolecule({json.dumps(mol_block)});"
                self.web_view.page().runJavaScript(js_code)
            
            # Modelle laden falls noch nicht geschehen
            if NMR_model_C is None or NMR_model_E is None:
                self.statusBar().showMessage("Loading models (one-time). This may take a moment...")
                QApplication.processEvents()
                # QMessageBox.information(self, "Info", "Modelle werden geladen (einmalig). Dies kann einen Moment dauern...")
                init_models()
                self.statusBar().showMessage("Calculation running...")
                QApplication.processEvents()
                
            # Vorhersagen
            pred_cascade = predict_cascade(mol, NMR_model_C, _models_dir)
            pred_est_nmr = predict_est_nmr(mol, NMR_model_E)
            pred_est_nmr_boltz = predict_est_nmr_boltzmann(mol, NMR_model_E)
            pred_dcode = predict_dcode_boltzmann(mol, codes_df) if codes_df is not None else {}
            
            # Ergebnisse sammeln
            results = []
            for atom in mol.GetAtoms():
                if atom.GetAtomicNum() == 6:
                    idx = atom.GetIdx()
                    pc = pred_cascade.get(idx, np.nan)
                    pe = pred_est_nmr.get(idx, np.nan)
                    peb = pred_est_nmr_boltz.get(idx, np.nan)
                    pd_val = pred_dcode.get(idx, np.nan)
                    
                    valid_shifts = [x for x in [pc, pe, peb, pd_val] if not np.isnan(x)]
                    if len(valid_shifts) > 0:
                        spannweite = max(valid_shifts) - min(valid_shifts)
                    else:
                        spannweite = np.nan
                    
                    results.append({
                        'idx': idx,
                        'cascade': pc,
                        'est_nmr': pe,
                        'est_nmr_boltz': peb,
                        'dcode': pd_val,
                        'spannweite': round(spannweite, 2) if not np.isnan(spannweite) else '-'
                    })
            
            # Tabelle updaten
            self.table.setRowCount(len(results))
            for row, res in enumerate(results):
                item_idx = QTableWidgetItem(str(res['idx']))
                item_idx.setTextAlignment(Qt.AlignCenter)
                
                c_val = QTableWidgetItem(str(res['cascade']) if not np.isnan(res['cascade']) else "N/A")
                c_val.setTextAlignment(Qt.AlignCenter)
                
                e_val = QTableWidgetItem(str(res['est_nmr']) if not np.isnan(res['est_nmr']) else "N/A")
                e_val.setTextAlignment(Qt.AlignCenter)
                
                eb_val = QTableWidgetItem(str(res['est_nmr_boltz']) if not np.isnan(res['est_nmr_boltz']) else "N/A")
                eb_val.setTextAlignment(Qt.AlignCenter)
                
                dc_val = QTableWidgetItem(str(res['dcode']) if not np.isnan(res['dcode']) else "N/A")
                dc_val.setTextAlignment(Qt.AlignCenter)
                
                s_val = QTableWidgetItem(str(res['spannweite']))
                s_val.setTextAlignment(Qt.AlignCenter)
                
                # Highlight große Abweichungen zwischen den Modellen
                if res['spannweite'] != '-' and float(res['spannweite']) > 5.0:
                    s_val.setBackground(QColor(255, 200, 200))
                
                self.table.setItem(row, 0, item_idx)
                self.table.setItem(row, 1, c_val)
                self.table.setItem(row, 2, e_val)
                self.table.setItem(row, 3, eb_val)
                self.table.setItem(row, 4, dc_val)
                self.table.setItem(row, 5, s_val)
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Error", f"An error occurred:\n{str(e)}")
        finally:
            self.calc_button.setEnabled(True)
            self.calc_button.setText("Calculate")
            self.statusBar().showMessage("Calculation finished.")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle("Fusion") # Sieht moderner aus auf Windows
    window = NMRApp()
    window.show()
    sys.exit(app.exec_())
