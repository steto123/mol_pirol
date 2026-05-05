### Sammlung von Funktionen, die immer wieder benötigt werden
from rdkit import *
from rdkit.Chem import AllChem
import numpy as np
import pandas as pd
from dcode import geometry
from dcode.geometry import *

#Generate Atom numbers
def show_atom_number(mol, label):
    for atom in mol.GetAtoms():
        atom.SetProp(label, str(atom.GetIdx()+1))
    return mol

#### Funktion zum Code generieren und als Atomeigenschaft speichern
# das Molekül muss schon die veränderten Atomnamen enthalten und optimiert sein
# Es wird die Atomeigenschaft DCode mit kompletten Code gesetzt

def DCodeMol(mol, sorte=6):
    for atom in mol.GetAtoms():
         # komplette Code generieren für C- Atome als Standard für Sorte = 6 
         # es können aber auch andere Sorten übergeben werden - für die Zukunft
        if (atom.GetAtomicNum()==sorte):
            dname=atom.GetProp("DCodeName") # Startwert
            completDcode=f"{dname}@"  # String zur Bildung des DCodes mit Startwert für das betractete Atom
            index=atom.GetIdx()
            entfernungen_liste = berechne_entfernungen(mol, index)
            for index, entfernung in entfernungen_liste:
                  atom2=mol.GetAtomWithIdx(index)
                  #print(f"Atom Index: {index}, Typ: {atom.GetSymbol()} Entfernung: {entfernung:.2f}") # zum Debuggen
                  if (entfernung<6):
                       dname=atom2.GetProp("DCodeName")
                       completDcode=completDcode + dname +"#"
            #print(f"Der Code für Atom {atom.GetIdx()} lautet: {completDcode}")
            atom.SetProp("DCode", completDcode)
    return mol


### Der ElCode ist identisch zum DCode nur mit Gasteiger Ladungen

def ElCodeMol(mol, sorte=6):
    # Berechne Gasteiger-Ladungen
    Chem.rdPartialCharges.ComputeGasteigerCharges(mol)
    for atom in mol.GetAtoms():
         # komplette Code generieren für C- Atome als Standard für Sorte = 6 
         # es können aber auch andere Sorten übergeben werden - für die Zukunft
        if (atom.GetAtomicNum()==sorte):
            dname=atom.GetProp("DCodeName") # Startwert
            completDcode=f"{dname}@"  # String zur Bildung des DCodes mit Startwert für das betrachtete Atom
            index=atom.GetIdx()
            entfernungen_liste = berechne_entfernungen(mol, index)
            for index, entfernung in entfernungen_liste:
                  atom2=mol.GetAtomWithIdx(index)
                  #print(f"Atom Index: {index}, Typ: {atom.GetSymbol()} Entfernung: {entfernung:.2f}") # zum Debuggen
                  if (entfernung<6):
                       dname = str(atom2.GetDoubleProp('_GasteigerCharge'))
                       completDcode=completDcode + dname +"#"
            #print(f"Der Code für Atom {atom.GetIdx()} lautet: {completDcode}")
            atom.SetProp("ElCode", completDcode)
    return mol


## Funktion zum Ausgeben einer Atomeigenschaft für eine bestimmte Atomsorte

def Ausgeben(mol, eigenschaft="DCode", sorte=6):
    for atom in mol.GetAtoms():
        if (atom.GetAtomicNum()==sorte):
            codestring=atom.GetProp(eigenschaft)
            aindex=atom.GetIdx()
            print(f"Atom {aindex} hat {eigenschaft}: {codestring}")            
    return


## Funktion zum Einlesen der NMR Daten und speichern der Verschiebungen
## als Atomeigenschaft Shift
## erst mal nur für 13C
## Übergeben wird das zu manipulierende Molekülobjekt

def smilesback(mol):
    N_smiles="NMREDATA_SMILES"   
    properties = mol.GetPropsAsDict()  #Alle in der SDF gespeicherten Eigenschaften einlesen
    cleaned_string = properties[N_smiles].replace('\\','') # das replace ersetzt den Backslash am Ende
    nsmiles=cleaned_string
    return nsmiles


#Dataframe für Assigment generieren
# alle ist der rohe String mit dem Assignment

def get_assignment(alle):
    assignment1 = alle.replace('\\', '').splitlines()

    # Aufteilen der Zeilen in Spalten und Erstellen eines DataFrames
    assign1 = []
    for line in assignment1:
        parts = line.split(', ')
        if len(parts) >= 3:
            # Die ersten zwei Teile als Value1 und Value2
            label = parts[0].strip()
            cleaned_label = label.strip()           
            value1 = float(parts[1])
            '''
            #mögliche Fehlerbehebung
            # Initialisierung der Liste für die Ergebnisse
            filtered_parts = []

            # Durchlaufen der Teile ab dem Index 2
            for part in parts[2:]:
                if ';' in part:
                    break  # Beende die Schleife, wenn ein Semikolon gefunden wird
            filtered_parts.append(part)  # Füge das Teil zur Ergebnisliste hinzu

            # Konvertiere die gefilterten Teile in Integer
            value2 = list(map(int, filtered_parts))

            '''
            value2 = list(map(int, parts[2:]))  # Restliche Teile als Liste von Integern
            assign1.append([cleaned_label, value1, value2])
    # Erstellen des DataFrames
    full_assign = pd.DataFrame(assign1, columns=['Label', 'Shift', 'Atome'])
    return full_assign

    
#Spektrum generieren
#rohspek ist der String mit den rohdaten
#nmrspek ist das dataframe was zurück gegeben wird
def generate_spektrum(rohspek):
    speklines_roh=rohspek.replace('\\', '').splitlines()
    speklines= speklines_roh[2:]
    spek = []
    for line in speklines:
        parts = line.split(', ')      
        label = parts[1].strip()
        cleaned_label = label.strip().replace('L=','')          
        value1 = float(parts[0])
        spek.append([cleaned_label, value1])
    # Erstellen des DataFrames
    nmrspek = pd.DataFrame(spek, columns=['Label', 'Shift'])
    return nmrspek
    

    
    

    
    
# Alte NMR EInlesefunktion, kann weg wenn neuer Weg geht
def nmrein(mol,mol2):
    # mol2 ist nur für die NMR datenextraktion wichtig
    # wichtige strings definieren
    N_version="NMREDATA_VERSION"
    N_solvent="NMREDATA_SOLVENT"
    N_level="NMREDATA_LEVEL"
    N_ID="NMREDATA_ID"
    N_temp="NMREDATA_TEMPERATURE"
    N_smiles="NMREDATA_SMILES"
    N_assign="NMREDATA_ASSIGNMENT"
    N_c_spektrum="NMREDATA_1D_13C"
    N_c_spektrum2="NMREDATA_1D_13C#2"
    N_c_spektrum3="NMREDATA_1D_13C#3"
    N_c_spektrum4="NMREDATA_1D_13C#4"
    properties = mol2.GetPropsAsDict()  #Alle in der SDF gespeicherten Eigenschaften einlesen
    # die properties enthalten jetzt die NMRedaten u.a.
    """
    # Ausgabe für Debug Zwecke
    if mol is not None:
        for prop_name, prop_value in properties.items():
            print(f"{prop_name}: {prop_value}")
    """
    # Signale extrahieren
    alle_signale = properties[N_assign]
    
    def extrahiere_signale(input_string):
        """
        Diese Funktion nimmt einen String entgegen, der durch Backslashes 
        getrennte Werte enthält, und gibt eine Liste dieser Werte zurück.
    
        :param input_string: Der Eingabestring, der verarbeitet werden soll
        :return: Liste von Strings, die durch Backslashes getrennt sind
        """
        # Erstellen der Liste von Strings, die durch den Backslash getrennt sind
        rohsignale = input_string.split('\\')
    
        # Entfernen von leeren Strings aus der Liste
        rohsignale = [signal for signal in rohsignale if signal]
    
        return rohsignale
    
    
    
    
    #print('Alle Signale')  # Debug
    #print(alle_signale)   #Debug 
    # Aufruf der Extraktionsfunktion
    signale_roh = extrahiere_signale(alle_signale)
    #print("Rohe Signale")   #Debug
    #print(signale_roh)      #Debug
    
    ###########################################################################################
    #                                                                                         #
    #      pandas Dataframe für alle C- Atome anlegen                                         #
    #          Spalte 0  enthält die jeweilige Atomnummer                                     #
    #          Spalte 1 die chemische Verschiebung, 0 ist initial                             #
    #                                                                                         #
    ###########################################################################################
    
    # Extrahiere die Atomnummern aller C-Atome
    c_atoms = [atom.GetIdx() + 1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'C']

    # Erstelle ein DataFrame mit den Atomnummern und einer zweiten Spalte 'Verschiebung'
    df = pd.DataFrame({
        'A-Nummer': c_atoms,
        'Verschiebung': [0.0] * len(c_atoms),  # Fülle die Spalte 'Verschiebung' mit 0
        'Equivalents': [''] * len(c_atoms),  # Fülle die Spalte 'Verschiebung' mit 0
    })


    ##################################################################################
    #                                                                                #
    #     Die Liste signale_roh[] enthält die Signale als                            #
    #     Name, Verschiebung, Liste der Atomnummern; Kommentar                       #
    #                                                                                #
    #      zuerst auf Existenz des Kommentars prüfen und diesen löschen              #
    #                                                                                #
    ##################################################################################

    # string zum zerlegen
    signalzahl = len(signale_roh)   # legt die Anzahl der zu zerlegenden Signale fest, ganz grosse Schleife
    
    for j in range(signalzahl):

        rohsignal = signale_roh[j]       
        # Überprüfen, ob der String ein Semikolon enthält
        if ';' in rohsignal:
            # Alles ab dem Semikolon inklusive löschen
            rohsignal = rohsignal.split(';')[0]
 
        ##################################################################################
        #                                                                                #
        #       Anzahl Kommatas im verbliebenen string Rohsignal pruefen                 #
        #       Bei 2 kommas in drei Teile splitten                                      #
        #       Teil 1 = Signalname                                                      #
        #       Teil 2 = chem. Verschiebung                                              #
        #       Teil 3 = Atomnummer  und weitere Teile                                                    #
        #                                                                                #


        #string splitten, nach dem Entfernen des Kommentars enthält er nur noch Spektrendaten
        teile = rohsignal.split(',')
        shift = float(str.strip(teile[1]))
        #print(len(teile))

        if (len(teile)) == 3:
            atomnummer = int(teile[2])  # Umwandlung in Integer
            # Suche im DataFrame nach der Zeile mit der gleichen Atomnummer
            if atomnummer in df['A-Nummer'].values:
                # Ändere Spalte 2 (Verschiebung) in den Wert von shift
                df.loc[df['A-Nummer'] == atomnummer, 'Verschiebung'] = shift
            
        #für mehr als ein Atom welches zu einer Verschiebung gehört - Equivalenz

        if (len(teile)) >3:
            # Feststellen der Anzahl equivalenter Kerne
            equiv = len(teile) - 2
            equiatome =''
            for i in range(equiv):
                teile_id = i+2
                atomnummer = int(teile[teile_id])  # Umwandlung in Integer
                equiatome=equiatome + str(atomnummer) + ','
            equiatome=equiatome.rstrip(',')
            for i in range(equiv):
                teile_id = i+2
                atomnummer = int(teile[teile_id])  # Umwandlung in Integer
                # Ändere Spalte 2 (Verschiebung) in den Wert von shift
                df.loc[df['A-Nummer'] == atomnummer, 'Verschiebung'] = shift
                df.loc[df['A-Nummer'] == atomnummer, 'Equivalents'] = equiatome
            

    # Ende der grossen Schleife j

    # Ausgabe des aktualisierten DataFrames
    # das sollte jetzt alle C- Atome zugeordnet enthalten
    #print("Zuordnungen der Verschiebung")
    #print(df)
    
    # Kopie des Moleküls erstellen, um das Original nicht zu verändern
    mol = Chem.Mol(mol)
    
    # Sicherstellen, dass wir editierbares Molekül haben
    mol = Chem.RWMol(mol)
    
    # Für jede Zeile im DataFrame
    for index, row in df.iterrows():
        # Atomindex ist A-Nummer minus 1
        atom_idx = int(row['A-Nummer']) - 1
        
        # Verschiebungswert
        shift = row['Verschiebung']
        
        # Prüfe, ob der Atomindex gültig ist
        if atom_idx < mol.GetNumAtoms():
            # Hole das Atom
            atom = mol.GetAtomWithIdx(atom_idx)
            
            # Setze die Shift-Eigenschaft
            atom.SetProp('Shift', str(shift))
        else:
            print(f"Warnung: Atomindex {atom_idx} überschreitet Molekülgröße ({mol.GetNumAtoms()} Atome)")
    
    
    #Datenbank ID auslesen
    cleaned_string = properties[N_ID].replace('\\','') # das replace ersetzt den Backslash am Ende
    roh_ndbid=cleaned_string
    teile2 = roh_ndbid.split('DB_ID=')
    nbid=teile2[1]
    
    
    # Konvertiere zurück zu normalem Mol-Objekt
    
    return   Chem.Mol(mol),nbid
