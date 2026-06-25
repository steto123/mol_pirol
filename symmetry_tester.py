import os
import sys
import math
import numpy as np
import json
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QLineEdit, QPushButton, 
                             QTableWidget, QTableWidgetItem, QMessageBox, QHeaderView, QSplitter,
                             QGraphicsView, QGraphicsScene, QStackedWidget, QCheckBox)
from PyQt5.QtCore import Qt, QUrl, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QColor, QPainter, QIcon
from PyQt5.QtSvg import QGraphicsSvgItem, QSvgRenderer

try:
    from PyQt5.QtWebEngineWidgets import QWebEngineView
    WEB_ENGINE_AVAILABLE = True
except ImportError:
    QWebEngineView = QWidget
    WEB_ENGINE_AVAILABLE = False

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D

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

def generate_3d_structure(mol):
    """
    Generates an optimized 3D structure using MMFF94.
    """
    m = Chem.AddHs(mol, addCoords=True)
    # Use multiple conformers to find a good local minimum
    cids = AllChem.EmbedMultipleConfs(m, numConfs=10, randomSeed=42, pruneRmsThresh=0.5)
    
    if not cids:
        # Fallback if 3D embedding fails
        if AllChem.EmbedMolecule(m, randomSeed=42) == -1:
            AllChem.Compute2DCoords(m)
        AllChem.MMFFOptimizeMolecule(m)
        return m
        
    # Optimize all and find lowest energy conformer
    energies = []
    for cid in cids:
        ff = AllChem.MMFFGetMoleculeForceField(m, AllChem.MMFFGetMoleculeProperties(m), confId=cid)
        if ff:
            ff.Initialize()
            ff.Minimize()
            energies.append((cid, ff.CalcEnergy()))
            
    if energies:
        best_cid = min(energies, key=lambda x: x[1])[0]
        # Keep only the best conformer
        new_m = Chem.Mol(m)
        new_m.RemoveAllConformers()
        new_m.AddConformer(m.GetConformer(best_cid), assignId=True)
        return new_m
    else:
        AllChem.MMFFOptimizeMolecule(m)
        return m

def calculate_symmetry_ranks(mol, m_3d=None):
    """
    Hybrid Fragment-Based Symmetry Ranking.

    Two atoms receive the same rank if:
    (1) they share the same topological rank (CanonicalRankAtoms, breakTies=False)
    AND
    (2) they have identical distance profiles within their rigid fragment
        (molecule cut at all rotatable bonds, idealized 2D coords per fragment).

    This correctly keeps homotopic rotating groups united (e.g. tert-butyl methyls,
    ortho/meta on phenyl) while separating diastereotopic groups in rigid
    scaffolds (e.g. cis/trans, dithiolane CH2).

    Note: m_3d is accepted for API compatibility but no longer used.
    """
    from collections import defaultdict

    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    base_ranks = list(Chem.CanonicalRankAtoms(mol, breakTies=False, includeChirality=True))
    num_at = mol.GetNumAtoms()

    # 1. Find rotatable bonds (single, non-ring, between non-terminal heavy atoms)
    rot_smarts = Chem.MolFromSmarts('[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]')
    rot_matches = mol.GetSubstructMatches(rot_smarts)
    rot_bonds = [mol.GetBondBetweenAtoms(a, b).GetIdx() for a, b in rot_matches]

    # 2. Cut molecule into rigid fragments
    if rot_bonds:
        frags_mol = Chem.FragmentOnBonds(mol, rot_bonds)
    else:
        frags_mol = Chem.Mol(mol)

    frag_indices = Chem.GetMolFrags(frags_mol, asMols=False)
    frags_mols   = Chem.GetMolFrags(frags_mol, asMols=True, sanitizeFrags=False)

    # 3. Compute idealized 2D coords for each fragment
    frag_confs = []
    for frag in frags_mols:
        try:
            AllChem.Compute2DCoords(frag)
            frag_confs.append(frag.GetConformer())
        except Exception:
            frag_confs.append(None)

    # 4. Group atoms by topological base rank, then refine via fragment geometry
    rank_groups = defaultdict(list)
    for i, r in enumerate(base_ranks):
        rank_groups[r].append(i)

    final_ranks = [-1] * num_at
    next_rank   = 0

    for r in sorted(rank_groups.keys()):
        atoms = rank_groups[r]
        if len(atoms) == 1:
            final_ranks[atoms[0]] = next_rank
            next_rank += 1
            continue

        # Build distance-profile signature for each atom within its rigid fragment
        signatures = {}
        for a_idx in atoms:
            for f_idx, f_atoms in enumerate(frag_indices):
                if a_idx in f_atoms:
                    frag_mol  = frags_mols[f_idx]
                    conf      = frag_confs[f_idx]
                    local_idx = f_atoms.index(a_idx)
                    if conf is not None:
                        pos   = conf.GetAtomPosition(local_idx)
                        dists = tuple(sorted(
                            round(pos.Distance(conf.GetAtomPosition(k)), 3)
                            for k in range(frag_mol.GetNumAtoms())
                        ))
                        signatures[a_idx] = dists
                    else:
                        signatures[a_idx] = (0,)
                    break

        # Group by signature, assign ranks deterministically
        sig_groups = defaultdict(list)
        for a_idx, sig in signatures.items():
            sig_groups[sig].append(a_idx)

        for sig in sorted(sig_groups.keys()):
            for a_idx in sig_groups[sig]:
                final_ranks[a_idx] = next_rank
            next_rank += 1

    return final_ranks

def draw_2d_svg(mol, highlight_indices=None):
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
    
    highlight_atoms = []
    if highlight_indices:
        for idx_tm, atom in enumerate(tm.GetAtoms()):
            if atom.HasProp("orig_idx") and atom.GetIntProp("orig_idx") in highlight_indices:
                highlight_atoms.append(idx_tm)
    
    if highlight_atoms:
        drawer.DrawMolecule(tm, highlightAtoms=highlight_atoms)
    else:
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
        self.resetTransform()
        self.fitInView(self.scene.itemsBoundingRect(), Qt.KeepAspectRatio)

    def wheelEvent(self, event):
        zoom_factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.scale(zoom_factor, zoom_factor)

class SymmetryWorker(QThread):
    done = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, smiles):
        super().__init__()
        self.smiles = smiles

    def run(self):
        try:
            mol = Chem.MolFromSmiles(self.smiles)
            if mol is None:
                self.error.emit("Invalid SMILES!")
                return
            
            # 1. Generate optimized 3D structure
            m_3d = generate_3d_structure(mol)
            
            # 2. Calculate Ranks using the 3D structure
            ranks = calculate_symmetry_ranks(mol, m_3d)
            
            self.done.emit({
                'mol': mol,
                'm_3d': m_3d,
                'ranks': ranks
            })
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))

class SymmetryTesterApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NMR Symmetry Tester (Experimental)")
        self.resize(1200, 800)
        self.setup_ui()

    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # Input
        input_box = QHBoxLayout()
        self.smiles_input = QLineEdit()
        self.smiles_input.setPlaceholderText("Enter SMILES (e.g. C1C=CC=C(/C=C(\\C)/C)C=1)")
        self.smiles_input.returnPressed.connect(self.start_calc)
        
        self.calc_btn = QPushButton("Analyze Symmetry")
        self.calc_btn.clicked.connect(self.start_calc)
        
        self.view_3d_cb = QCheckBox("3D View")
        self.view_3d_cb.toggled.connect(self.toggle_view)
        
        input_box.addWidget(QLabel("SMILES:"))
        input_box.addWidget(self.smiles_input)
        input_box.addWidget(self.calc_btn)
        input_box.addWidget(self.view_3d_cb)
        layout.addLayout(input_box)

        # Main Splitter
        splitter = QSplitter(Qt.Horizontal)
        
        # Left: Views
        self.view_stack = QStackedWidget()
        self.svg_view = InteractiveSvgView()
        self.view_stack.addWidget(self.svg_view)
        
        if WEB_ENGINE_AVAILABLE:
            self.web_view = QWebEngineView()
            self.web_view.setHtml(HTML_3DMOL)
            self.view_stack.addWidget(self.web_view)
        
        splitter.addWidget(self.view_stack)

        # Right: Table
        self.table = QTableWidget()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["Atom Index", "Element", "Sym. Rank"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.itemSelectionChanged.connect(self.on_selection)
        splitter.addWidget(self.table)
        
        layout.addWidget(splitter)
        self.statusBar().showMessage("Ready")

    def toggle_view(self, checked):
        if checked and WEB_ENGINE_AVAILABLE:
            self.view_stack.setCurrentIndex(1)
        else:
            self.view_stack.setCurrentIndex(0)

    def start_calc(self):
        smiles = self.smiles_input.text().strip()
        if not smiles: return
        
        self.calc_btn.setEnabled(False)
        self.statusBar().showMessage("Calculating (3D Optimization)...")
        
        self.worker = SymmetryWorker(smiles)
        self.worker.done.connect(self.on_done)
        self.worker.error.connect(self.on_error)
        self.worker.start()

    def on_error(self, msg):
        QMessageBox.critical(self, "Error", msg)
        self.calc_btn.setEnabled(True)
        self.statusBar().showMessage("Error occurred.")

    def on_done(self, result):
        self.result = result
        self.calc_btn.setEnabled(True)
        self.statusBar().showMessage("Analysis complete.")
        
        # Update Table
        mol = result['mol']
        ranks = result['ranks']
        self.table.blockSignals(True)
        self.table.setRowCount(0)
        
        # Colors for ranks
        rank_colors = {}
        def get_color(r):
            if r not in rank_colors:
                import colorsys
                hue = (r * 137.508) % 1.0
                rgb = colorsys.hls_to_rgb(hue, 0.9, 0.6)
                rank_colors[r] = QColor.fromRgbF(*rgb)
            return rank_colors[r]

        for i, atom in enumerate(mol.GetAtoms()):
            self.table.insertRow(i)
            
            idx_item = QTableWidgetItem(str(i))
            elem_item = QTableWidgetItem(atom.GetSymbol())
            rank_item = QTableWidgetItem(str(ranks[i]))
            
            idx_item.setFlags(idx_item.flags() ^ Qt.ItemIsEditable)
            elem_item.setFlags(elem_item.flags() ^ Qt.ItemIsEditable)
            rank_item.setFlags(rank_item.flags() ^ Qt.ItemIsEditable)

            if atom.GetSymbol() == 'C':
                font = idx_item.font()
                font.setBold(True)
                idx_item.setFont(font)
            
            color = get_color(ranks[i])
            rank_item.setBackground(color)
            
            self.table.setItem(i, 0, idx_item)
            self.table.setItem(i, 1, elem_item)
            self.table.setItem(i, 2, rank_item)
            
        self.table.blockSignals(False)

        # Update 2D
        svg = draw_2d_svg(mol)
        self.svg_view.load(svg)
        
        # Update 3D
        if WEB_ENGINE_AVAILABLE:
            mol_block = Chem.MolToMolBlock(result['m_3d'])
            self.web_view.page().runJavaScript(f"loadMolecule({json.dumps(mol_block)})")

    def on_selection(self):
        if not hasattr(self, 'result'): return
        rows = self.table.selectionModel().selectedRows()
        if not rows: return
        
        row_idx = rows[0].row()
        item = self.table.item(row_idx, 0)
        if not item: return
        
        selected_idx = int(item.text())
        rank = self.result['ranks'][selected_idx]
        
        # Find all atoms with same rank
        equivalent_indices = [i for i, r in enumerate(self.result['ranks']) if r == rank]
        
        # Update 2D highlight
        svg = draw_2d_svg(self.result['mol'], equivalent_indices)
        self.svg_view.load(svg)
        
        # Update 3D highlight
        if WEB_ENGINE_AVAILABLE:
            self.web_view.page().runJavaScript(f"highlightAtoms({json.dumps(equivalent_indices)})")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SymmetryTesterApp()
    window.show()
    sys.exit(app.exec_())
