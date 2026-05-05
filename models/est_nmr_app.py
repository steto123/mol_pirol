from flask import Flask, render_template, request, jsonify
import torch
import math
import sys
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D

from est_nmr_read_xyz import read_xyz
from est_nmr_module import create_svg, smiles_to_xyz

app = Flask(__name__)

# Modell beim Start laden
model = torch.jit.load('DLNMR1.pt')
MODEL_VERSION = model.version
citation = "neural network: Thomas Hehre, Philip E. Klunzinger, Bernard J. Deppmeier, William Sean Ohlinger, Warren J Hehre, doi:10.1021/acs.joc.5c00927"

@app.route('/', methods=['GET', 'POST'])
def index():
  html_1 = "<h2>NMR Predict</h2>Version with force field geometry only<br><br>"
  html_2 = f"<br><br>NMR model version: {model.version} Copyright 2025 Wavefunction, Inc.<br>{citation}"
  
  # Standardwert für SMILES
  smiles = "CC(=O)O"  # Essigsäure
  svg_text = ""
  html_3 = ""
  
  if request.method == 'POST':
      # SMILES-Code vom Formular übernehmen
      smiles = request.form.get('smiles', smiles)
  
  try:
      # XYZ-Koordinaten aus SMILES generieren
      moltemp = smiles_to_xyz(smiles)
      mols = read_xyz('input.xyz')
      
      mol = mols[0]  # Nimm nur das erste Element von mols
      html_3 = f"Read {mol.label}<br>"
      
      Z = torch.tensor(mol.species, dtype=torch.int64)
      R = torch.tensor(mol.coords, dtype=torch.float32).unsqueeze(0)  # add batch dimension
      res = model(Z, R)  # our model processes one unique molecule at a time
      shifts = res[2].tolist()
      
      html_3 += "<table border='1'><tr><th>Label</th><th>Shift (ppm)</th></tr>"
      for i in range(len(shifts)):
          shift = shifts[i]
          if not math.isnan(shift):
              label = mol.labels[i]
              html_3 += f"<tr><td>{label}</td><td>{shift:.5f}</td></tr>"
      html_3 += "</table>"
      
      # SVG generieren
      svg_text = create_svg(moltemp)
  
  except Exception as e:
      html_3 = f"Fehler: {str(e)}"
  
  # HTML-Formular hinzufügen
  form_html = '''
  <form method="POST">
      <label for="smiles">SMILES-Code eingeben:</label>
      <input type="text" id="smiles" name="smiles" value="{}" required>
      <input type="submit" value="NMR vorhersagen">
  </form>
  '''.format(smiles)
  
  html_out = (
      html_1 + 
      form_html + 
      svg_text + 
      html_3 +
      html_2
  )

  return html_out

if __name__ == '__main__':
  print(f"NMR model version: {MODEL_VERSION} Copyright 2025 Wavefunction, Inc.")
  app.run(debug=True, host='0.0.0.0', port=4000)