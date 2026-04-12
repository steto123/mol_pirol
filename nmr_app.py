import os
import sys
import logging
import warnings
import pandas as pd
import numpy as np
import math
import pickle
import torch
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QLineEdit, QPushButton, 
                             QTableWidget, QTableWidgetItem, QMessageBox, QHeaderView, QSplitter)
from PyQt5.QtCore import Qt
from PyQt5.QtSvg import QSvgWidget
from PyQt5.QtGui import QFont, QColor

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D

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

def init_models():
    global NMR_model_C, NMR_model_E
    try:
        print("Lade Modelle...")
        modelpath_C = os.path.join(_models_dir, "cascade", "trained_model", "best_model.hdf5")
        NMR_model_C = load_model(modelpath_C, custom_objects=_CUSTOM_OBJECTS)
        
        modelpath_E = os.path.join(_models_dir, "DLNMR1.pt")
        NMR_model_E = torch.jit.load(modelpath_E)
        print("Modelle erfolgreich geladen!")
    except Exception as e:
        print(f"Fehler beim Laden der Modelle: {e}")

def predict_cascade(mol, model, models_dir):
    _CASCADE_DIR = os.path.join(models_dir, "cascade")
    preprocessor_path = os.path.join(_CASCADE_DIR, 'preprocessor.p')
    with open(preprocessor_path, 'rb') as ft:
        preprocessor = pickle.load(ft)['preprocessor']
    
    m = Chem.AddHs(mol, addCoords=True)
    inputs, df, mols = preprocess_C([m], preprocessor, keep_all_cf=True)
    if not inputs: return {}
    
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
    spread_df = pd.concat(chunks)
    spread_df['predicted'] = predicted_values
    spread_df['b_weight'] = spread_df.relative_E.apply(lambda x: math.exp(-x/(0.001987*298.15)))
    
    df_group = spread_df.set_index(['mol_id', 'atom_index', 'cf_id']).groupby(level=1)
    return {int(a_idx): round(group.apply(lambda x: x['b_weight']*x['predicted'], axis=1).sum()/group.b_weight.sum(), 2) 
            for a_idx, group in df_group}

def predict_est_nmr(mol, model):
    m = Chem.AddHs(mol, addCoords=True)
    if m.GetNumConformers() == 0:
        if AllChem.EmbedMolecule(m, randomSeed=42) == -1:
            AllChem.Compute2DCoords(m)
        AllChem.MMFFOptimizeMolecule(m)
        
    conf = m.GetConformer()
    species = [atom.GetAtomicNum() for atom in m.GetAtoms()]
    coords = [[pos.x, pos.y, pos.z] for pos in [conf.GetAtomPosition(i) for i in range(m.GetNumAtoms())]]
    
    Z = torch.tensor(species, dtype=torch.int64)
    R = torch.tensor(coords, dtype=torch.float32).unsqueeze(0)
    
    with torch.no_grad():
        res = model(Z, R)
    all_shifts = res[2].tolist()
    
    return {i: round(all_shifts[i], 2) for i, atom in enumerate(m.GetAtoms()) if atom.GetAtomicNum() == 6}

def draw_annotated_mol(mol):
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
        self.smiles_input.setPlaceholderText("Geben Sie hier den SMILES-Code ein (z.B. c1ccccc1)")
        
        self.calc_button = QPushButton("Berechnen")
        self.calc_button.setFont(QFont("Segoe UI", 11, QFont.Bold))
        self.calc_button.setStyleSheet("background-color: #0078D4; color: white; padding: 5px 15px; border-radius: 4px;")
        self.calc_button.clicked.connect(self.run_analysis)
        
        input_layout.addWidget(self.smiles_label)
        input_layout.addWidget(self.smiles_input)
        input_layout.addWidget(self.calc_button)
        
        main_layout.addLayout(input_layout)
        
        # Splitter für Bild und Tabelle
        splitter = QSplitter(Qt.Horizontal)
        
        # SVG Bildanzeige
        self.svg_widget = QSvgWidget()
        self.svg_widget.setMinimumWidth(400)
        splitter.addWidget(self.svg_widget)
        
        # Tabelle für Ergebnisse
        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["Atom Index", "CASCADE", "EST-NMR", "|Diff|"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setFont(QFont("Segoe UI", 10))
        splitter.addWidget(self.table)
        
        main_layout.addWidget(splitter)
        
    def run_analysis(self):
        smiles = self.smiles_input.text().strip()
        if not smiles:
            QMessageBox.warning(self, "Fehler", "Bitte einen SMILES-Code eingeben!")
            return
            
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            QMessageBox.warning(self, "Fehler", "Ungültiger SMILES-Code!")
            return
            
        self.calc_button.setEnabled(False)
        self.calc_button.setText("Berechne...")
        QApplication.processEvents()
        
        try:
            # SVG zeichnen
            svg_bytes = draw_annotated_mol(mol)
            self.svg_widget.load(svg_bytes)
            
            # Modelle laden falls noch nicht geschehen
            if NMR_model_C is None or NMR_model_E is None:
                QMessageBox.information(self, "Info", "Modelle werden geladen (einmalig). Dies kann einen Moment dauern...")
                init_models()
                
            # Vorhersagen
            pred_cascade = predict_cascade(mol, NMR_model_C, _models_dir)
            pred_est_nmr = predict_est_nmr(mol, NMR_model_E)
            
            # Ergebnisse sammeln
            results = []
            for atom in mol.GetAtoms():
                if atom.GetAtomicNum() == 6:
                    idx = atom.GetIdx()
                    pc = pred_cascade.get(idx, np.nan)
                    pe = pred_est_nmr.get(idx, np.nan)
                    
                    diff = abs(pc - pe) if not (np.isnan(pc) or np.isnan(pe)) else np.nan
                    
                    results.append({
                        'idx': idx,
                        'cascade': pc,
                        'est_nmr': pe,
                        'diff': round(diff, 2) if not np.isnan(diff) else '-'
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
                
                d_val = QTableWidgetItem(str(res['diff']))
                d_val.setTextAlignment(Qt.AlignCenter)
                
                # Highlight große Abweichungen zwischen den beiden Modellen
                if res['diff'] != '-' and float(res['diff']) > 5.0:
                    d_val.setBackground(QColor(255, 200, 200))
                
                self.table.setItem(row, 0, item_idx)
                self.table.setItem(row, 1, c_val)
                self.table.setItem(row, 2, e_val)
                self.table.setItem(row, 3, d_val)
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Fehler", f"Ein Fehler ist aufgetreten:\n{str(e)}")
        finally:
            self.calc_button.setEnabled(True)
            self.calc_button.setText("Berechnen")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle("Fusion") # Sieht moderner aus auf Windows
    window = NMRApp()
    window.show()
    sys.exit(app.exec_())
