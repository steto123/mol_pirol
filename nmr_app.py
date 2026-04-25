"""
NMR 13C Prediction App
Diese GUI-Anwendung kombiniert drei ML/Topologie-Modelle (CASCADE, EST-NMR, DCode),
um die chemischen C-13 NMR-Verschiebungen aus einem SMILES Code zu prognostizieren und zu vergleichen.
"""
import os
import sys

# Fix for pythonw.exe: Redirect stdout/stderr if they are None to avoid AttributeError
if sys.stdout is None:
    class NullWriter:
        def write(self, data): pass
        def flush(self): pass
    sys.stdout = NullWriter()
if sys.stderr is None:
    class NullWriter:
        def write(self, data): pass
        def flush(self): pass
    sys.stderr = NullWriter()

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
import colorsys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QLineEdit, QPushButton, 
                             QTableWidget, QTableWidgetItem, QMessageBox, QHeaderView, QSplitter,
                             QGraphicsView, QGraphicsScene, QStackedWidget, QCheckBox, QTabWidget,
                             QFileDialog, QComboBox, QDialog)
from PyQt5.QtCore import Qt, QDateTime, QUrl, QThread, pyqtSignal
try:
    import matplotlib
    matplotlib.use('Qt5Agg')
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    FigureCanvas = object

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
    function highlightAtoms(indices) {
        viewer.setStyle({}, {stick: {radius: 0.15}, sphere: {scale: 0.3}});
        if (indices && indices.length > 0) {
            indices.forEach(function(idx) {
                if (idx !== -1) {
                    viewer.addStyle({serial: idx + 1}, {sphere: {scale: 0.5, color: '#00FFFF'}});
                }
            });
        }
        viewer.render();
    }
  </script>
</body>
</html>
"""
from PyQt5.QtSvg import QSvgWidget, QGraphicsSvgItem, QSvgRenderer
from PyQt5.QtGui import QFont, QColor, QPainter, QIcon

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
os.environ['QTWEBENGINE_CHROMIUM_FLAGS'] = '--disable-logging --log-level=3'
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

def generate_boltzmann_conformers(mol):
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
        ff = AllChem.MMFFGetMoleculeForceField(m, AllChem.MMFFGetMoleculeProperties(m))
        if ff:
            ff.Initialize()
            ff.Minimize()
            e = ff.CalcEnergy()
        else:
            e = 0.0
        return m, {0: 1.0}, {0: e}
        
    min_e = min([e for c, e in energies])
    RT = 0.001987 * 298.15
    b_weights = {cid: math.exp(-(e - min_e)/RT) for cid, e in energies}
    sum_w = sum(b_weights.values())
    b_weights = {cid: w/sum_w for cid, w in b_weights.items()}
    energies_dict = {cid: e for cid, e in energies}
    
    return m, b_weights, energies_dict

def predict_est_nmr_single(m_3d, cid, model):
    species = [atom.GetAtomicNum() for atom in m_3d.GetAtoms()]
    conf = m_3d.GetConformer(cid)
    coords = [[pos.x, pos.y, pos.z] for pos in [conf.GetAtomPosition(i) for i in range(m_3d.GetNumAtoms())]]
    
    Z = torch.tensor(species, dtype=torch.int64)
    R = torch.tensor(coords, dtype=torch.float32).unsqueeze(0)
    
    with torch.no_grad():
        res = model(Z, R)
    all_shifts = res[2].tolist()
    
    # Rückgabe nur für Kohlenstoff (C = 6)
    return {i: round(all_shifts[i], 2) for i, atom in enumerate(m_3d.GetAtoms()) if atom.GetAtomicNum() == 6}

def predict_est_nmr_boltzmann(m, b_weights, model):
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

def predict_dcode_boltzmann(m, b_weights, codes_df_input):
    if codes_df_input is None:
        return {}
    
    dcode_sums = {atom.GetIdx(): 0.0 for atom in m.GetAtoms() if atom.GetAtomicNum() == 6}
    weight_sums = {atom.GetIdx(): 0.0 for atom in m.GetAtoms() if atom.GetAtomicNum() == 6}
    
    for cid, w in b_weights.items():
        single_conf_m = Chem.Mol(m)
        single_conf_m.RemoveAllConformers()
        single_conf_m.AddConformer(m.GetConformer(cid), assignId=True)
        
        single_conf_m = DCodeName(single_conf_m)
        single_conf_m = DCodeMol(single_conf_m)
        
        for atom in single_conf_m.GetAtoms():
            if atom.GetAtomicNum() == 6 and atom.HasProp('DCode'):
                codestring = atom.GetProp('DCode')
                verschiebung, treffer, _, _, _ = calcshift(codes_df_input, codestring, atom.GetIdx())
                
                if verschiebung != -999 and verschiebung != -999.0:
                    dcode_sums[atom.GetIdx()] += w * verschiebung
                    weight_sums[atom.GetIdx()] += w
                    
    final_results = {}
    for atom_idx in dcode_sums:
        if weight_sums[atom_idx] > 0:
            final_results[atom_idx] = round(dcode_sums[atom_idx] / weight_sums[atom_idx], 2)
        else:
            final_results[atom_idx] = np.nan
            
    return final_results

def draw_annotated_mol(mol, highlight_orig_indices=None):
    """
    Rendert ein Molekül als SVG, wobei die Original-Atom-Indizes
    auch nach dem Entfernen der Wasserstoffatome korrekt erhalten bleiben.
    Optionale Hervorhebung von Atomen anhand ihrer Original-Indizes.
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
    
    highlightAtoms = []
    if highlight_orig_indices:
        for idx_tm, atom in enumerate(tm.GetAtoms()):
            if atom.HasProp("orig_idx") and atom.GetIntProp("orig_idx") in highlight_orig_indices:
                highlightAtoms.append(idx_tm)
    
    if highlightAtoms:
        # Hervorhebung anwenden
        drawer.DrawMolecule(tm, highlightAtoms=highlightAtoms)
    else:
        drawer.DrawMolecule(tm)
        
    drawer.FinishDrawing()
    return drawer.GetDrawingText().encode('utf-8')

class SpectrumCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)
        self.peak_to_atom = []

    def plot_spectrum(self, shifts_dict, molecule_smiles, model_name):
        self.ax.clear()
        self.peak_to_atom = []
        
        if not shifts_dict:
            self.ax.set_title("No data to plot")
            self.draw()
            return

        valid_shifts = [v for v in shifts_dict.values() if not np.isnan(v)]
        if not valid_shifts:
            self.draw()
            return
            
        min_s = min(valid_shifts) - 20
        max_s = max(valid_shifts) + 20
        if min_s < 0: min_s = -10
        if max_s > 250: max_s = max(250, max_s + 20)
        
        x = np.linspace(max_s, min_s, 2000)
        y = np.zeros_like(x)
        gamma = 0.3
        
        peak_x = []
        peak_y = np.zeros(len(valid_shifts))
        
        for atom_idx, shift in shifts_dict.items():
            if np.isnan(shift): continue
            y += 1.0 / (1.0 + ((x - shift) / gamma)**2)
            peak_x.append(shift)
            self.peak_to_atom.append(atom_idx)
            
        for i, (atom_idx, shift) in enumerate(zip(self.peak_to_atom, peak_x)):
            idx_x = np.argmin(np.abs(x - shift))
            peak_y[i] = y[idx_x]
            self.ax.text(shift, y[idx_x] + 0.05, str(atom_idx), horizontalalignment='center', fontsize=8, color='blue')
            
        self.ax.plot(x, y, color='#0078D4', linewidth=1.5)
        self.ax.scatter(peak_x, peak_y, picker=True, pickradius=10, color='red', alpha=0.0)
        
        self.ax.set_xlim(max_s, min_s)
        self.ax.set_yticks([])
        self.ax.set_xlabel("13C Chemical Shift (ppm)")
        self.ax.set_title(f"Simulated 13C-NMR Spectrum - {model_name}")
        self.ax.grid(True, axis='x', linestyle='--', alpha=0.6)
        
        self.fig.tight_layout()
        self.draw()

class InteractiveSvgView(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.svg_item = None
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setRenderHint(QPainter.Antialiasing)
        
    def load(self, svg_bytes, reset_view=True):
        self.scene.clear()
        self.svg_item = QGraphicsSvgItem()
        renderer = QSvgRenderer(svg_bytes)
        self.svg_item.setSharedRenderer(renderer)
        self.scene.addItem(self.svg_item)
        if reset_view:
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

class CalculationWorker(QThread):
    progress_status = pyqtSignal(str)
    calculation_done = pyqtSignal(object)
    calculation_error = pyqtSignal(str)

    def __init__(self, smiles, use_symmetry=False):
        super().__init__()
        self.smiles = smiles
        self.use_symmetry = use_symmetry

    def run(self):
        try:
            self.progress_status.emit("Parsing SMILES...")
            mol = Chem.MolFromSmiles(self.smiles)
            if mol is None:
                self.calculation_error.emit("Invalid SMILES code!")
                return
                
            self.progress_status.emit("Loading models (one-time). This may take a moment...")
            if NMR_model_C is None or NMR_model_E is None:
                init_models()
                
            self.progress_status.emit("Generating conformers & optimizing...")
            m_3d, b_weights, energies_dict = generate_boltzmann_conformers(mol)
            
            self.progress_status.emit("Predicting via ML Models...")
            pred_cascade = predict_cascade(mol, NMR_model_C, _models_dir)
            if energies_dict:
                sorted_confs = sorted(energies_dict.items(), key=lambda x: x[1])
                pred_est_nmr = predict_est_nmr_single(m_3d, sorted_confs[0][0], NMR_model_E)
            else:
                sorted_confs = []
                pred_est_nmr = {}
            pred_est_nmr_boltz = predict_est_nmr_boltzmann(m_3d, b_weights, NMR_model_E)
            pred_dcode = predict_dcode_boltzmann(m_3d, b_weights, codes_df) if codes_df is not None else {}
            
            self.progress_status.emit("Calculating symmetry ranks...")
            Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
            base_ranks = list(Chem.CanonicalRankAtoms(mol, breakTies=False, includeChirality=True))
            
            # Refine ranks using spatial distances from the 3D structure to distinguish 
            # cis/trans or axial/equatorial positions that are topologically identical.
            # We use a tolerant check (0.4A) to avoid over-sensitivity to conformational tilts.
            if m_3d.GetNumConformers() > 0:
                best_cid = sorted_confs[0][0] if sorted_confs else 0
                conf = m_3d.GetConformer(best_cid)
                num_at = mol.GetNumAtoms()
                
                # Pre-calculate distance sets for all atoms
                all_dist_sets = []
                for i in range(num_at):
                    d_set = sorted([conf.GetAtomPosition(i).Distance(conf.GetAtomPosition(j)) 
                                   for j in range(num_at) if i != j])
                    all_dist_sets.append(d_set)
                
                # Group atoms that are topologically equivalent AND spatially similar
                sym_ranks = [-1] * num_at
                next_rank = 0
                for i in range(num_at):
                    if sym_ranks[i] == -1:
                        sym_ranks[i] = next_rank
                        for j in range(i + 1, num_at):
                            if sym_ranks[j] == -1 and base_ranks[i] == base_ranks[j]:
                                # Check spatial similarity with a 0.4A tolerance
                                is_similar = True
                                for d1, d2 in zip(all_dist_sets[i], all_dist_sets[j]):
                                    if abs(d1 - d2) > 0.4:
                                        is_similar = False
                                        break
                                if is_similar:
                                    sym_ranks[j] = next_rank
                        next_rank += 1
            else:
                sym_ranks = base_ranks
            
            if self.use_symmetry:
                self.progress_status.emit("Applying symmetry averaging...")
                for pred_dict in [pred_cascade, pred_est_nmr, pred_est_nmr_boltz, pred_dcode]:
                    if not pred_dict: continue
                    # Group by rank
                    rank_to_vals = {}
                    for atom_idx, val in pred_dict.items():
                        if np.isnan(val): continue
                        rank = sym_ranks[atom_idx]
                        if rank not in rank_to_vals: rank_to_vals[rank] = []
                        rank_to_vals[rank].append(val)
                    
                    # Apply average back to all atoms of the same rank
                    for atom_idx in list(pred_dict.keys()):
                        rank = sym_ranks[atom_idx]
                        if rank in rank_to_vals:
                            avg_val = sum(rank_to_vals[rank]) / len(rank_to_vals[rank])
                            pred_dict[atom_idx] = round(avg_val, 2)

            result = {
                'mol': mol,
                'm_3d': m_3d,
                'b_weights': b_weights,
                'energies_dict': energies_dict,
                'sorted_confs': sorted_confs,
                'pred_cascade': pred_cascade,
                'pred_est_nmr': pred_est_nmr,
                'pred_est_nmr_boltz': pred_est_nmr_boltz,
                'pred_dcode': pred_dcode,
                'sym_ranks': sym_ranks,
                'smiles': self.smiles,
                'use_symmetry': self.use_symmetry
            }
            self.calculation_done.emit(result)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.calculation_error.emit(str(e))

class KetcherDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Draw Structure - Ketcher (Offline)")
        self.resize(1000, 700)
        self.layout = QVBoxLayout(self)
        
        self.web_view = QWebEngineView()
        self.web_view.titleChanged.connect(self.on_title)
        pth = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ketcher", "standalone", "index.html").replace("\\", "/")
        self.web_view.setUrl(QUrl(f"file:///{pth}"))
        self.layout.addWidget(self.web_view)
        
        self.button_box = QHBoxLayout()
        self.ok_btn = QPushButton("Use Drawn Structure")
        self.ok_btn.setStyleSheet("background-color: #0078D4; color: white; padding: 5px 15px; font-weight: bold;")
        self.ok_btn.clicked.connect(self.request_smiles)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        
        self.button_box.addStretch()
        self.button_box.addWidget(self.cancel_btn)
        self.button_box.addWidget(self.ok_btn)
        self.layout.addLayout(self.button_box)
        
        self.smiles = ""

    def request_smiles(self):
        js = "window.ketcher.getSmiles().then(s => { document.title = 'SMILES_' + s; }).catch(e => { alert(e); });"
        self.web_view.page().runJavaScript(js)
            
    def on_title(self, title):
        if title.startswith("SMILES_"):
            self.smiles = title[7:]
            self.accept()

class NMRApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NMR 13C Prediction App (Windows 11)")
        self.resize(1400, 900)
        
        # Set Application Icon
        icon_path = os.path.join(os.path.dirname(__file__), "app_icon.png")
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
            
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
        
        self.draw_button = QPushButton("🖌 Draw")
        self.draw_button.setFont(QFont("Segoe UI", 11, QFont.Bold))
        self.draw_button.clicked.connect(self.open_ketcher)
        
        self.calc_button = QPushButton("Calculate")
        self.calc_button.setFont(QFont("Segoe UI", 11, QFont.Bold))
        self.calc_button.setStyleSheet("background-color: #0078D4; color: white; padding: 5px 15px; border-radius: 4px;")
        self.calc_button.clicked.connect(self.run_analysis)
        
        self.toggle_3d_cb = QCheckBox("3D View")
        self.toggle_3d_cb.setFont(QFont("Segoe UI", 11))
        self.toggle_3d_cb.toggled.connect(self.toggle_view)
        
        self.export_button = QPushButton("Export Report")
        self.export_button.setFont(QFont("Segoe UI", 11))
        self.export_button.clicked.connect(self.export_report)
        
        self.sym_avg_cb = QCheckBox("Symmetry Average (Exp.)")
        self.sym_avg_cb.setFont(QFont("Segoe UI", 11))
        self.sym_avg_cb.setToolTip("Experimental: Average predicted shifts for chemically equivalent atoms using 3D-spatial analysis")
        self.sym_avg_cb.setChecked(True)
        
        input_layout.addWidget(self.smiles_label)
        input_layout.addWidget(self.smiles_input)
        input_layout.addWidget(self.draw_button)
        input_layout.addWidget(self.calc_button)
        input_layout.addWidget(self.toggle_3d_cb)
        input_layout.addWidget(self.sym_avg_cb)
        input_layout.addWidget(self.export_button)
        
        main_layout.addLayout(input_layout)
        
        # Experimental Data & History
        aux_layout = QHBoxLayout()
        self.history_combo = QComboBox()
        self.history_combo.addItem("-- History --")
        self.history_combo.currentIndexChanged.connect(self.load_from_history)
        
        self.exp_input = QLineEdit()
        self.exp_input.setPlaceholderText("Optional Exp. Shifts (comma sep. e.g. 15.2, 128.4) for Auto-MAE")
        
        aux_layout.addWidget(QLabel("History:"))
        aux_layout.addWidget(self.history_combo)
        aux_layout.addSpacing(20)
        aux_layout.addWidget(QLabel("Exp. Data:"))
        aux_layout.addWidget(self.exp_input)
        
        main_layout.addLayout(aux_layout)
        
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
        
        tabs = QTabWidget()
        
        # Tab 1: Tabelle für Ergebnisse
        self.table = QTableWidget()
        self.table.setColumnCount(8)
        self.table.setHorizontalHeaderLabels(["Atom Index", "Sym. Rank", "Exp. Data", "CASCADE", "EST-NMR", "EST-NMR (Boltz)", "DCode", "Range"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Interactive)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Interactive)
        self.table.setColumnWidth(0, 80)
        self.table.setColumnWidth(1, 80)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.setFont(QFont("Segoe UI", 10))
        self.table.itemSelectionChanged.connect(self.on_table_selection)
        tabs.addTab(self.table, "Results")
        
        # Tab 2: Konformere
        self.conf_table = QTableWidget()
        self.conf_table.setColumnCount(4)
        self.conf_table.setHorizontalHeaderLabels(["Conformer ID", "Abs. Energy (kcal/mol)", "Rel. Energy (kcal/mol)", "Weight (%)"])
        self.conf_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.conf_table.verticalHeader().setVisible(False)
        self.conf_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.conf_table.setSelectionMode(QTableWidget.SingleSelection)
        self.conf_table.setFont(QFont("Segoe UI", 10))
        self.conf_table.itemSelectionChanged.connect(self.on_conf_table_selection)
        tabs.addTab(self.conf_table, "Conformers")
        
        # Tab 3: Spectrum
        if HAS_MATPLOTLIB:
            self.spectrum_widget = QWidget()
            val_layout = QVBoxLayout(self.spectrum_widget)
            
            sel_layout = QHBoxLayout()
            sel_layout.addWidget(QLabel("Model for Spectrum:"))
            self.model_combo = QComboBox()
            self.model_combo.addItems(["EST-NMR (Boltz)", "CASCADE", "EST-NMR", "DCode"])
            self.model_combo.currentIndexChanged.connect(self.update_spectrum)
            sel_layout.addWidget(self.model_combo)
            sel_layout.addStretch()
            val_layout.addLayout(sel_layout)
            
            self.spectrum_canvas = SpectrumCanvas(self)
            self.spectrum_canvas.mpl_connect('pick_event', self.on_spectrum_pick)
            self.spectrum_toolbar = NavigationToolbar(self.spectrum_canvas, self)
            val_layout.addWidget(self.spectrum_toolbar)
            val_layout.addWidget(self.spectrum_canvas)
            
            tabs.addTab(self.spectrum_widget, "Spectrum")
            
        splitter.addWidget(tabs)
        
        main_layout.addWidget(splitter)
        
        self.warning_label = QLabel("Warning: CASCADE is only trained for elements C, H, N, O, S, P, F, Cl.")
        self.warning_label.setFont(QFont("Segoe UI", 10, QFont.Bold))
        self.warning_label.setStyleSheet("color: red;")
        self.warning_label.setVisible(False)
        main_layout.addWidget(self.warning_label)
        
        self.mae_label = QLabel("")
        self.mae_label.setFont(QFont("Segoe UI", 10, QFont.Bold))
        self.mae_label.setStyleSheet("color: green;")
        main_layout.addWidget(self.mae_label)
        
        self.statusBar().showMessage("Ready")
        
        self.session_cache = {}
        
    def open_ketcher(self):
        if not WEB_ENGINE_AVAILABLE:
            QMessageBox.warning(self, "Missing Dependency", "PyQtWebEngine is not installed. Ketcher requires the WebEngine.")
            return
            
        dialog = KetcherDialog(self)
        if dialog.exec_() == QDialog.Accepted and dialog.smiles:
            self.smiles_input.setText(dialog.smiles)
        
    def update_spectrum(self):
        if not hasattr(self, 'current_preds') or not HAS_MATPLOTLIB: return
        model_name = self.model_combo.currentText()
        shifts = self.current_preds.get(model_name, {})
        smiles = self.smiles_input.text().strip()
        self.spectrum_canvas.plot_spectrum(shifts, smiles, model_name)
        
    def on_spectrum_pick(self, event):
        if not event.ind.size: return
        idx = event.ind[0]
        atom_idx = self.spectrum_canvas.peak_to_atom[idx]
        for row in range(self.table.rowCount()):
            if self.table.item(row, 0).text() == str(atom_idx):
                self.table.selectRow(row)
                break
        
        # Highlight all equivalent atoms
        highlight_indices = [atom_idx]
        if hasattr(self, 'current_sym_ranks') and atom_idx < len(self.current_sym_ranks):
            rank = self.current_sym_ranks[atom_idx]
            highlight_indices = [i for i, r in enumerate(self.current_sym_ranks) if r == rank]
        
        self.update_highlight(highlight_indices)
        
    def export_report(self):
        if self.table.rowCount() == 0:
            QMessageBox.warning(self, "Export", "No results to export. Calculate first.")
            return
            
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Report", "", "CSV Files (*.csv);;HTML Report (*.html)")
        if not file_path:
            return
            
        try:
            if file_path.endswith('.csv'):
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write("Atom Index,Sym. Rank,Exp. Data,CASCADE,EST-NMR,EST-NMR (Boltz),DCode,Range\n")
                    for row in range(self.table.rowCount()):
                        row_data = [self.table.item(row, col).text() for col in range(8)]
                        f.write(','.join(row_data) + "\n")
                QMessageBox.information(self, "Export", f"CSV Exported to {file_path}")
                
            elif file_path.endswith('.html'):
                import base64
                svg_data = ""
                if hasattr(self, 'current_mol') and self.current_mol is not None:
                    svg_bytes = draw_annotated_mol(self.current_mol)
                    svg_data = base64.b64encode(svg_bytes).decode('utf-8')
                    
                html = f'''<html><head><style>
                    body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; padding: 20px; color: #333; }}
                    table {{ border-collapse: collapse; width: 100%; margin-bottom: 30px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
                    th, td {{ border: 1px solid #ddd; padding: 10px; text-align: center; }}
                    th {{ background-color: #f8f9fa; font-weight: bold; color: #0078D4; }}
                    h1, h2, h3 {{ color: #0078D4; border-bottom: 2px solid #0078D4; padding-bottom: 5px; }}
                    .molecule {{ text-align: center; margin: 30px 0; background: #fff; padding: 20px; border-radius: 8px; }}
                    .summary {{ background: #e7f3ff; padding: 15px; border-radius: 8px; margin-bottom: 20px; border-left: 5px solid #0078D4; }}
                    .mae {{ color: #1e7e34; font-weight: bold; font-size: 1.1em; }}
                </style></head><body>
                    <h1>NMR 13C Prediction Report</h1>
                    <div class="summary">
                        <p><b>SMILES:</b> {self.smiles_input.text()}</p>
                        <p><b>Date:</b> {QDateTime.currentDateTime().toString("yyyy-MM-dd HH:mm:ss")}</p>'''
                
                if self.mae_label.text():
                    html += f'<p class="mae"><b>{self.mae_label.text()}</b></p>'
                
                html += '</div>'
                
                if svg_data:
                    html += f'<div class="molecule"><h2>Molecular Structure</h2><img src="data:image/svg+xml;base64,{svg_data}" alt="Molecule" width="500"/></div>'
                    
                html += '''<h2>Prediction Results</h2>
                <table>
                    <tr>
                        <th>Atom Index</th>
                        <th>Sym. Rank</th>
                        <th>Exp. Data</th>
                        <th>CASCADE</th>
                        <th>EST-NMR</th>
                        <th>EST-NMR (Boltz)</th>
                        <th>DCode</th>
                        <th>Range (ppm)</th>
                    </tr>'''
                
                for row in range(self.table.rowCount()):
                    html += "<tr>"
                    # Symmetry color
                    bg_color_sym = self.table.item(row, 1).background().color().name()
                    
                    for col in range(8):
                        bg_style = f' style="background-color: {bg_color_sym};"'
                        item = self.table.item(row, col)
                        
                        # High range highlight
                        if col == 7 and item.background().color() == QColor(255, 200, 200):
                            bg_style = ' style="background-color: #ffc8c8;"'
                        
                        # Exp data bold
                        cell_content = item.text()
                        if col == 2 and cell_content != "-":
                            cell_content = f"<b>{cell_content}</b>"
                            
                        html += f"<td{bg_style}>{cell_content}</td>"
                    html += "</tr>"
                html += "</table>"
                
                # Conformers Table
                if self.conf_table.rowCount() > 1:
                    html += "<h2>Conformer Ensemble (Boltzmann)</h2>"
                    html += "<table><tr><th>Conformer ID</th><th>Abs. Energy (kcal/mol)</th><th>Rel. Energy (kcal/mol)</th><th>Weight (%)</th></tr>"
                    for row in range(self.conf_table.rowCount()):
                        html += "<tr>"
                        for col in range(4):
                            html += f"<td>{self.conf_table.item(row, col).text()}</td>"
                        html += "</tr>"
                    html += "</table>"
                
                html += "</body></html>"
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(html)
                    
                QMessageBox.information(self, "Export", f"HTML Report Exported to {file_path}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Error", f"Could not export file:\n{str(e)}")
            
    def on_conf_table_selection(self):
        if not hasattr(self, 'current_m_3d') or self.current_m_3d is None: return
        selected_items = self.conf_table.selectedItems()
        if not selected_items: return
            
        row = selected_items[0].row()
        cid_str = self.conf_table.item(row, 0).text()
        try:
            cid = int(cid_str)
            if getattr(self, 'web_view', None) is not None:
                mol_block = Chem.MolToMolBlock(self.current_m_3d, confId=cid)
                js_code = f"if(typeof loadMolecule !== 'undefined') loadMolecule({json.dumps(mol_block)});"
                self.web_view.page().runJavaScript(js_code)
                
                # Re-highlight if atoms are selected in main table
                main_sel = self.table.selectedItems()
                if main_sel:
                    atom_idx_str = self.table.item(main_sel[0].row(), 0).text()
                    try:
                        atom_idx = int(atom_idx_str)
                        highlight_indices = [atom_idx]
                        if hasattr(self, 'current_sym_ranks') and atom_idx < len(self.current_sym_ranks):
                            rank = self.current_sym_ranks[atom_idx]
                            highlight_indices = [i for i, r in enumerate(self.current_sym_ranks) if r == rank]
                        
                        js_highlight = f"setTimeout(function(){{ if(typeof highlightAtoms !== 'undefined') highlightAtoms({json.dumps(highlight_indices)}); }}, 100);"
                        self.web_view.page().runJavaScript(js_highlight)
                    except ValueError:
                        pass
        except ValueError:
            pass

    def on_table_selection(self):
        selected_items = self.table.selectedItems()
        if not selected_items:
            self.update_highlight([])
            return
            
        row = selected_items[0].row()
        idx_str = self.table.item(row, 0).text()
        try:
            idx = int(idx_str)
            # Find all atoms with same symmetry rank
            highlight_indices = [idx]
            if hasattr(self, 'current_sym_ranks') and idx < len(self.current_sym_ranks):
                rank = self.current_sym_ranks[idx]
                highlight_indices = [i for i, r in enumerate(self.current_sym_ranks) if r == rank]
            
            self.update_highlight(highlight_indices)
        except ValueError:
            self.update_highlight([])
            
    def update_highlight(self, atom_indices):
        """
        Updates highlights in both 2D and 3D views.
        atom_indices can be a single int or a list of ints.
        """
        if not hasattr(self, 'current_mol') or self.current_mol is None:
            return
            
        if isinstance(atom_indices, int):
            if atom_indices == -1:
                highlight_list = []
            else:
                highlight_list = [atom_indices]
        else:
            highlight_list = atom_indices

        # 1. Update 2D SVG
        svg_bytes = draw_annotated_mol(self.current_mol, highlight_list)
        self.svg_widget.load(svg_bytes, reset_view=False)
        
        # 2. Update 3D Viewer (if available)
        if WEB_ENGINE_AVAILABLE and getattr(self, 'web_view', None) is not None:
            js_code = f"if(typeof highlightAtoms !== 'undefined') highlightAtoms({json.dumps(highlight_list)});"
            self.web_view.page().runJavaScript(js_code)

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
            
        if self.history_combo.currentIndex() > 0 and self.history_combo.currentText() == smiles:
            if smiles in self.session_cache:
                self.calculation_success(self.session_cache[smiles])
                return

        self.calc_button.setEnabled(False)
        self.calc_button.setText("Calculating...")
        self.mae_label.setText("")
        self.table.clearSelection()
        
        self.worker = CalculationWorker(smiles, use_symmetry=self.sym_avg_cb.isChecked())
        self.worker.progress_status.connect(self.statusBar().showMessage)
        self.worker.calculation_done.connect(self.calculation_success)
        self.worker.calculation_error.connect(self.calculation_err)
        self.worker.start()
        
    def calculation_err(self, err_msg):
        self.calc_button.setEnabled(True)
        self.calc_button.setText("Calculate")
        self.statusBar().showMessage("Calculation failed.")
        QMessageBox.critical(self, "Error", f"An error occurred:\n{err_msg}")
        
    def load_from_history(self):
        if self.history_combo.currentIndex() > 0:
            smiles = self.history_combo.currentText()
            if smiles in self.session_cache:
                self.smiles_input.setText(smiles)
                self.calculation_success(self.session_cache[smiles])
                
    def calculation_success(self, result):
        self.calc_button.setEnabled(True)
        self.calc_button.setText("Calculate")
        self.statusBar().showMessage("Calculation finished.")
        
        mol = result['mol']
        self.current_mol = mol
        self.current_m_3d = result['m_3d']
        self.current_sym_ranks = result.get('sym_ranks', [])
        
        # Save to history
        smiles = result['smiles']
        if smiles not in self.session_cache:
            self.session_cache[smiles] = result
            if self.history_combo.findText(smiles) == -1:
                self.history_combo.addItem(smiles)
            # Find and set the index without triggering load
            idx = self.history_combo.findText(smiles)
            self.history_combo.blockSignals(True)
            self.history_combo.setCurrentIndex(idx)
            self.history_combo.blockSignals(False)
            
        allowed_elements = {6, 1, 7, 8, 16, 15, 9, 17}
        molecule_elements = {atom.GetAtomicNum() for atom in mol.GetAtoms()}
        if molecule_elements - allowed_elements:
            self.warning_label.setVisible(True)
        else:
            self.warning_label.setVisible(False)
            
        try:
            svg_bytes = draw_annotated_mol(mol)
            self.svg_widget.load(svg_bytes)
            
            sorted_confs = result['sorted_confs']
            energies_dict = result['energies_dict']
            b_weights = result['b_weights']
            
            if energies_dict:
                min_e = min(energies_dict.values())
                self.conf_table.setRowCount(len(sorted_confs))
                for row, (cid, energy) in enumerate(sorted_confs):
                    cid_item = QTableWidgetItem(str(cid))
                    cid_item.setTextAlignment(Qt.AlignCenter)
                    abs_e_item = QTableWidgetItem(f"{energy:.3f}")
                    abs_e_item.setTextAlignment(Qt.AlignCenter)
                    rel_e_item = QTableWidgetItem(f"{energy - min_e:.3f}")
                    rel_e_item.setTextAlignment(Qt.AlignCenter)
                    w_item = QTableWidgetItem(f"{b_weights[cid]*100:.1f}")
                    w_item.setTextAlignment(Qt.AlignCenter)
                    
                    self.conf_table.setItem(row, 0, cid_item)
                    self.conf_table.setItem(row, 1, abs_e_item)
                    self.conf_table.setItem(row, 2, rel_e_item)
                    self.conf_table.setItem(row, 3, w_item)
                
                if WEB_ENGINE_AVAILABLE and getattr(self, 'web_view', None) is not None:
                    first_cid = sorted_confs[0][0]
                    mol_block = Chem.MolToMolBlock(self.current_m_3d, confId=first_cid)
                    js_code = f"if(typeof loadMolecule !== 'undefined') loadMolecule({json.dumps(mol_block)});"
                    self.web_view.page().runJavaScript(js_code)
                    
            pred_cascade = result['pred_cascade']
            pred_est_nmr = result['pred_est_nmr']
            pred_est_nmr_boltz = result['pred_est_nmr_boltz']
            pred_dcode = result['pred_dcode']
            sym_ranks = result.get('sym_ranks', [])
            
            results_list = []
            for atom in mol.GetAtoms():
                if atom.GetAtomicNum() == 6:
                    idx = atom.GetIdx()
                    s_rank = sym_ranks[idx] if idx < len(sym_ranks) else -1
                    pc = pred_cascade.get(idx, np.nan)
                    pe = pred_est_nmr.get(idx, np.nan)
                    peb = pred_est_nmr_boltz.get(idx, np.nan)
                    pd_val = pred_dcode.get(idx, np.nan)
                    
                    valid_shifts = [x for x in [pc, pe, peb, pd_val] if not np.isnan(x)]
                    if len(valid_shifts) > 0:
                        spannweite = max(valid_shifts) - min(valid_shifts)
                    else:
                        spannweite = np.nan
                    
                    results_list.append({
                        'idx': idx,
                        'sym_rank': s_rank,
                        'cascade': pc,
                        'est_nmr': pe,
                        'est_nmr_boltz': peb,
                        'dcode': pd_val,
                        'avg': sum(valid_shifts)/len(valid_shifts) if valid_shifts else 0.0,
                        'spannweite': round(spannweite, 2) if not np.isnan(spannweite) else '-',
                        'exp': ''
                    })
            
            exp_text = self.exp_input.text().strip()
            exp_shifts = []
            if exp_text:
                try:
                    exp_shifts = [float(x.strip()) for x in exp_text.replace(';', ',').split(',') if x.strip()]
                    exp_shifts.sort(reverse=True)
                except ValueError:
                    self.mae_label.setText("Invalid format in Exp. Data!")
                    exp_shifts = []
            
            if exp_shifts:
                unique_c_ranks = sorted(list(set([res['sym_rank'] for res in results_list])), reverse=True)
                
                # If we have exactly as many or fewer signals than unique symmetry groups, group them
                if len(exp_shifts) <= len(unique_c_ranks):
                    rank_to_avg = {}
                    for res in results_list:
                        r = res['sym_rank']
                        if r not in rank_to_avg: rank_to_avg[r] = []
                        rank_to_avg[r].append(res['avg'])
                    
                    group_shifts = []
                    for r, vals in rank_to_avg.items():
                        group_shifts.append({'rank': r, 'avg_pred': sum(vals)/len(vals)})
                    group_shifts.sort(key=lambda x: x['avg_pred'], reverse=True)
                    
                    rank_to_exp = {}
                    for i in range(min(len(exp_shifts), len(group_shifts))):
                        rank_to_exp[group_shifts[i]['rank']] = str(exp_shifts[i])
                    
                    for res in results_list:
                        res['exp'] = rank_to_exp.get(res['sym_rank'], '')
                else:
                    # Fallback to standard 1-to-1 matching
                    results_list.sort(key=lambda x: x['avg'], reverse=True)
                    for i in range(min(len(exp_shifts), len(results_list))):
                        results_list[i]['exp'] = str(exp_shifts[i])
                    results_list.sort(key=lambda x: x['idx'])
                    
            self.table.setRowCount(len(results_list))
            
            # Create a color map for symmetry ranks
            unique_all_ranks = sorted(list(set([res['sym_rank'] for res in results_list])))
            rank_colors = {}
            for i, r in enumerate(unique_all_ranks):
                # Using a slightly higher saturation (0.12 instead of 0.05) to make it visible
                h = (i * 0.618033988749895) % 1.0
                r_col, g_col, b_col = colorsys.hsv_to_rgb(h, 0.12, 0.98)
                rank_colors[r] = QColor(int(r_col*255), int(g_col*255), int(b_col*255))

            for row, res in enumerate(results_list):
                item_idx = QTableWidgetItem(str(res['idx']))
                item_idx.setTextAlignment(Qt.AlignCenter)
                
                item_sym = QTableWidgetItem(str(res['sym_rank']))
                item_sym.setTextAlignment(Qt.AlignCenter)
                
                exp_val = QTableWidgetItem(res['exp'] if res['exp'] else "-")
                exp_val.setTextAlignment(Qt.AlignCenter)
                font = QFont()
                font.setBold(True)
                exp_val.setFont(font)
                exp_val.setForeground(QColor("#1e7e34"))
                
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
                
                self.table.setItem(row, 0, item_idx)
                self.table.setItem(row, 1, item_sym)
                self.table.setItem(row, 2, exp_val)
                self.table.setItem(row, 3, c_val)
                self.table.setItem(row, 4, e_val)
                self.table.setItem(row, 5, eb_val)
                self.table.setItem(row, 6, dc_val)
                self.table.setItem(row, 7, s_val)
                
                # Apply symmetry background color
                bg_color = rank_colors.get(res['sym_rank'], QColor(255, 255, 255))
                for col in range(8):
                    item = self.table.item(row, col)
                    if item: item.setBackground(bg_color)
                
                # Overwrite range background if high
                if res['spannweite'] != '-' and float(res['spannweite']) > 5.0:
                    s_val.setBackground(QColor(255, 200, 200))
                
            self.current_preds = {
                "CASCADE": pred_cascade,
                "EST-NMR": pred_est_nmr,
                "EST-NMR (Boltz)": pred_est_nmr_boltz,
                "DCode": pred_dcode
            }
            if HAS_MATPLOTLIB:
                self.update_spectrum()
                
            # Perform Auto-MAE Assignment if exp data provided
            if exp_text and exp_shifts:
                try:
                    mae_texts = []
                    for m_name, p_dict in self.current_preds.items():
                        # Get predictions for all C atoms that have a prediction
                        atom_preds = []
                        for res in results_list:
                            val = p_dict.get(res['idx'], np.nan)
                            if not np.isnan(val):
                                atom_preds.append({'val': val, 'rank': res['sym_rank']})
                        
                        if not atom_preds: continue
                        
                        unique_ranks_in_model = sorted(list(set([p['rank'] for p in atom_preds])), reverse=True)
                        
                        if len(exp_shifts) <= len(unique_ranks_in_model):
                            # Symmetry-aware grouping
                            rank_groups = {}
                            for p in atom_preds:
                                if p['rank'] not in rank_groups: rank_groups[p['rank']] = []
                                rank_groups[p['rank']].append(p['val'])
                            
                            group_avg_preds = sorted([sum(vals)/len(vals) for vals in rank_groups.values()], reverse=True)
                            pairs = min(len(exp_shifts), len(group_avg_preds))
                            if pairs > 0:
                                errors = [abs(exp_shifts[i] - group_avg_preds[i]) for i in range(pairs)]
                                mae = sum(errors)/pairs
                                mae_texts.append(f"{m_name}: {mae:.2f} ppm")
                        else:
                            # 1-to-1 matching
                            p_vals = sorted([p['val'] for p in atom_preds], reverse=True)
                            pairs = min(len(exp_shifts), len(p_vals))
                            if pairs > 0:
                                errors = [abs(exp_shifts[i] - p_vals[i]) for i in range(pairs)]
                                mae = sum(errors)/pairs
                                mae_texts.append(f"{m_name}: {mae:.2f} ppm")
                                
                    if mae_texts:
                        self.mae_label.setText("Exp. MAE: " + " | ".join(mae_texts))
                    else:
                        self.mae_label.setText("")
                except Exception:
                    pass
                    
        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Error", f"Error displaying results:\n{str(e)}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle("Fusion") # Sieht moderner aus auf Windows
    window = NMRApp()
    window.show()
    sys.exit(app.exec_())
