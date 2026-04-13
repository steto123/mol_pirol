##########################################################################
#                                                                        #
#       Funktionen zur Geometriemanipulation                             #
#                                                                        #
##########################################################################

from rdkit import *
from rdkit.Chem import AllChem
import numpy as np


## Force Field optimierung in Schleife
### mol_roh ist das zu optimierenden Molekül

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


## explizite Wasserstoffe prüfen
### Wenn vorhanden dann True

def Hda(mol):
    if mol.GetNumAtoms() - mol.GetNumHeavyAtoms() > 0:
        da = True
    else:
        da = False
    
    return da

# den DCodeNamen nach den regeln festlegen


def DCodeName(mol, info=False):
    for atom in mol.GetAtoms():
        if (atom.GetAtomicNum() >1):
            NameString=""
            ringgroesse=""
            # für Debug Zwecke
            if info:
                debugname=""
                atom_index=atom.GetIdx()
                if atom.GetAtomicNum()==6:
                   debugname=str(atom_index)
            ##      
            symbolE=atom.GetSymbol()
            EValenz=str(len(atom.GetNeighbors()))  # Die Anzahl der Nachbarn anstelle expliziter Valenz
            ring_string="n" # n für nicht in einem Ring
            if atom.IsInRing():
                '''atom_index=atom.GetIdx()
                # Hole die Ringinformationen
                ring_info = mol.GetRingInfo()
                # Überprüfen, ob das Atom im Ring ist
                for ring in ring_info.BondRings():
                    # es gibt Fälle da klappt das nicht, dann gibt es nur ein r
                    if atom_index in ring:
                        ringgroesse=str(len(ring))
                ring_string="r"+ringgroesse
                '''
                ring_string="r"  #r für in einem Ring
            chiral_symbol="No"
            chiral_tag=f"{atom.GetChiralTag()}"
            if chiral_tag =='CHI_TETRAHEDRAL_CCW' : 
                chiral_symbol ='CC'    # sollte S sein, ist mir aber sicherer so
            if chiral_tag =='CHI_TETRAHEDRAL_CW' : 
                chiral_symbol ='CW'     # das sollte R sein
            if chiral_tag =='CHI_ALLENE' : 
                chiral_symbol ='Al'     # Allen          
            if chiral_tag =='CHI_TETRAHEDRAL' : 
                chiral_symbol ='Tt'     
            if chiral_tag =='CHI_OCTAHEDRAL' : 
                chiral_symbol ='Ot'  
            if chiral_tag =='CHI_OTHER' : 
                chiral_symbol ='Oh'    
            if chiral_tag =='CHI_SQUAREPLANAR' : 
                chiral_symbol ='Sp' 
            if chiral_tag =='CHI_TRIGONALBIPYRAMIDAL' : 
                chiral_symbol ='Bp' 
            if info:
                NameString=debugname+symbolE+EValenz+ring_string+chiral_symbol #debug name
            else:
                NameString=symbolE+EValenz+ring_string+chiral_symbol
        if (atom.GetAtomicNum()==1):
            NameString="H"
        
        # Codenamen setzen
        atom.SetProp("DCodeName", NameString)
        
    return mol

## Entfernungberechnung und Sortierung
## mol ist das Molekül
## index ist das betrachtete Atom
'''
Beispielliste
[(11, 0.9815844666166658), (1, 1.3856957338929305), (2, 2.3719027859719057), (10, 2.4047156378447645), (12, 2.66436417319777), (9, 2.9234605397454696), (6, 3.6909953350240277), (3, 3.708084626567193), (8, 4.1771121505773205), (5, 4.221245009619098), (7, 4.600664281332038), (15, 4.892456286980385), (4, 4.934183166234785), (16, 5.032188443780098), (14, 5.586519527612851), (13, 5.586597866349934)]

'''

def berechne_entfernungen(mol, index):
    # Holen der Koordinaten des gegebenen Atoms
    atom_koordinaten = mol.GetConformer().GetAtomPosition(index)
    
    entfernungen = []
    i=index # zur Sicherheit um i zu definieren
    for atom in mol.GetAtoms():
        i = atom.GetIdx()
        if i != index:  # Überspringe das gegebene Atom
            andere_atom_koordinaten = mol.GetConformer().GetAtomPosition(i)
            entfernung = np.linalg.norm(atom_koordinaten - andere_atom_koordinaten)  # Berechnung der Entfernung
            entfernungen.append((i, entfernung))  # Speichern von Index und Entfernung
            
    # Sortiere die Liste nach Entfernung
    entfernungen.sort(key=lambda x: x[1])
    
    return entfernungen

