"""
Module für die Flask- WebApp

Rückgabe sollte immer HTML sein

"""
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import rdDepictor
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D


def create_svg(mol):
    rdDepictor.Compute2DCoords(mol)
    drawer = rdMolDraw2D.MolDraw2DSVG(400, 300)
    drawer.drawOptions().addAtomIndices = True
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    svg_content = drawer.GetDrawingText()    
    return(svg_content)


def ffoptimize(mol_roh):

    # Initiale Werte festlegen
    value = 10  # Anfangswert (kann beliebig sein)
    max_iterations = 20
    counter = 0
    max_iter=200
    mmff_variant="MMFF94"
    mol_opt=mol_roh

    # Schleife, die solange durchläuft, bis value 0 ist oder 20 Durchläufe abgeschlossen sind
    while value != 0 and counter < max_iterations:
        # Beispiel für eine Berechnung, die value verändert
        value =AllChem.MMFFOptimizeMolecule(mol_opt, mmff_variant, max_iter)  # Standardanzahl Iterartionen mit Kraftfeld machen
  
        # Zähler erhöhen
        counter += 1

    return mol_opt 



def smiles_to_xyz(smiles_code, output_file='input.xyz'):
    """
    Konvertiert einen SMILES-Code in eine 3D-Struktur und speichert als XYZ-Datei
    
    Parameter:
    - smiles_code: SMILES-Repräsentation des Moleküls
    - output_file: Pfad zur Ausgabe-XYZ-Datei
    """
    # RDKit Molekül erstellen
    mol = Chem.MolFromSmiles(smiles_code)
    
    # Wasserstoffatome hinzufügen
    mol = Chem.AddHs(mol)
    
    # 3D-Konformer generieren
    AllChem.EmbedMolecule(mol, randomSeed=42)  # Seed für Reproduzierbarkeit
    
    # Geometrie optimieren
    #AllChem.MMFFOptimizeMolecule(mol)
    
    mol_opt= ffoptimize(mol)
    mol=mol_opt
    # Atomkoordinaten und Symbole extrahieren
    conf = mol.GetConformer()
    atom_symbols = [mol.GetAtomWithIdx(i).GetSymbol() for i in range(mol.GetNumAtoms())]
    
    # XYZ-Datei schreiben
    with open(output_file, 'w') as f:
        # Erste Zeile: Anzahl der Atome
        f.write(f"{mol.GetNumAtoms()}\n")
        
        # Zweite Zeile: Kommentar (optional)
        f.write(f"Molecule from SMILES: {smiles_code}\n")
        
        # Koordinaten für jedes Atom
        for i, symbol in enumerate(atom_symbols):
            pos = conf.GetAtomPosition(i)
            f.write(f"{symbol}{i} {pos.x:.6f} {pos.y:.6f} {pos.z:.6f}\n")
    
    print(f"XYZ-Datei wurde als {output_file} gespeichert.")
    
    return mol


