# Dokumentation der NMR 13C Prediction App

Diese Applikation bietet eine grafische Oberfläche zur Vorhersage von 13C-NMR-Verschiebungen (Nuclear Magnetic Resonance) anhand von SMILES-Codes. Sie vereint drei unabhängige Vorhersagemethoden und aggregiert diese, um eine robuste Konsens-Schätzung und eine Fehlereinschätzung über die Spannweite bereitzustellen. 

## 1. Programmablaufplan (Flowchart)

```mermaid
graph TD
    A["Start: UI initialisiert"] --> B["Benutzer gibt SMILES ein"]
    B --> C("Klick auf 'Berechnen'")
    
    C --> D{"Wurde SMILES erkannt?"}
    D -- Nein --> E["Fehlermeldung"]
    D -- Ja --> F["RDKit generiert 2D-Molekül & SVG Bild"]
    
    F --> G{"Sind Modelle im RAM?"}
    G -- Nein --> H["Modelle & CSV-DB Laden"]
    H --> I
    G -- Ja --> I["Starte parallele Predictions"]
    
    I --> J["CASCADE Graph Neural Net"]
    I --> K["EST-NMR NN 3D (Single & Boltz)"]
    I --> L["DCode Topologie Algorithmus"]
    
    subgraph Konformere ["Konformer & Boltzmann Zyklus"]
        L1["Generiere 10 Konformere"] --> L2["Minimiere Energien mit MMFF94"]
        L2 --> L3["Berechne Boltzmann-Gewichte e^-dE/RT"]
        L3 --> L4["Generiere Features: EST-NMR Tensoren / DCode Strings"]
        L4 --> L5["Führe Vorhersage aus (PyTorch / CSV)"]
        L5 --> L6["Mittelwert gewichten"]
    end
    
    L --> L1
    K -.->|Nutzt für EST-NMR Boltz| L1
    
    J --> M["Ergebnisse sammeln"]
    K --> M
    L6 --> M
    
    M --> N["Berechne Spannweite Max - Min"]
    N --> O{Spannweite > 5.0 ppm?}
    O -- Ja --> P["Zelle im UI Rötlich markieren"]
    O -- Nein --> Q["Normal in Tabelle eintragen"]
    
    P --> R["Anzeige im UI TableWidget"]
    Q --> R
    R --> S["Bereit für nächsten SMILES"]
```

## 2. Implementierte Methoden

### 2.1 CASCADE (Graph Neural Network)
Die `predict_cascade`-Funktion basiert auf einem Graph Neural Network Modell (`GraphModel` via `tf_keras`/`nfp` Layer). 
* **Funktionsweise**: Es formt das RDKit-Molekül über einen Preprocessor in ein Graph-Format um (Knoten = Atome, Kanten = Bindungen). Das Modell erzeugt konformerspezifische Graph-Embeddings.
* **Besonderheit**: CASCADE generiert intern eigene Konformere und wichtet die chemische Vorhersage per Boltzmann-Ansatz (durch MMFF generierte relative Energien).
* **Quelle**: G.N.N Entwicklungen im Rahmen des CASCADE 13C-C-NMR Prediction Projekts.

### 2.2 EST-NMR (DLNMR1.pt PyTorch Modell)
Die Funktionen `predict_est_nmr` und `predict_est_nmr_boltzmann` laden ein pre-trained PyTorch Modell. Dieses Neural Network operiert direkt auf den dreidimensionalen Koordinaten der Atomarten.
* **Funktionsweise (Single)**: Das 1D RDKit-Molekül wird (sofern nicht anders vorhanden) gebettet und via MMFF94 optimiert. Koordinatenvektor und Atomtypenvektor werden als Tensoren an das PyTorch-Modell übergeben, welches die Shift-Werte pro Atom index-genau schätzt.
* **Funktionsweise (Boltzmann)**: Analog zur DCode Methode werden 10 verschiedene Konformere generiert und im Kraftfeld minimiert. Das NN schätzt die chemischen Verschiebungen für jedes Konformer separat. Das Endergebnis ist der gemäß der relativen Energien Boltzmann-gewichtete Mittelwert ($\exp(-\Delta E/RT)$). In der Ergebnistabelle werden beide Varianten (`EST-NMR` und `EST-NMR (Boltz)`) für einen direkten Vergleich dargestellt.
* **Quelle**: Neuronales Netzwerk. Zitat: Thomas Hehre, Philip E. Klunzinger, Bernard J. Deppmeier, William Sean Ohlinger, Warren J Hehre, doi:10.1021/acs.joc.5c00927.

### 2.3 DCode (Distance & Topology Code)
Die `predict_dcode_boltzmann`-Funktion verwendet einen proprietären, neu integrierten Topologie-Code-Algorithmus aus dem bereitgestellten Github (`steto123/dcode`).
* **Funktionsweise**: Für jedes 3D-Konformer (10 Stück) wird für jedes Kohlenstoffatom ermittelt, welche Nachbarn (und Atom-Art) in einem Radius von 6 Ångström liegen. Diese topologische Information führt zu einem String-Code (z.B. "C4rn@..."). Danach wird in einer >47 MB großen CSV-Datenbank nach genau diesem Topologie-Code gesucht und der Durchschnitt als Shift angenommen.
* **Boltzmann-Wichtung**: Die Verschiebungen aus den 10 Konformeren werden anhand ihrer MMFF-Energien nach $\exp(-\Delta E/RT)$ gewichtet. Weicht eine Geometrie stark vom Energieminimum ab, fließt ihr prognostizierter Shift kaum in den finalen Wert mit ein.

### 2.4 Spannweite als Qualitätssiegel
Für jedes C-Atom vergleicht das Skript die berechneten Werte aller 3 Modelle. Die `"Spannweite"` wird als `Maximum(Shifts) - Minimum(Shifts)` berechnet. Liegt sie höher als 5 ppm, wird die Tabelle rot markiert. Dies weist den Experten auf herausfordernde stereochemische Bereiche, anormale Elektronegativitäten oder RDKit-Geometriefehler hin.

## 2.5 Symmetrie-Mittelung (Experimentelles Feature)
Die Applikation nutzt ein hybrides Verfahren zur Identifizierung chemisch äquivalenter Atome, um die Vorhersagequalität zu steigern.

*   **Hybrid-Ansatz**:
    1.  **Topologisches Ranking**: Zuerst wird via `RDKit.Chem.CanonicalRankAtoms` ein Basis-Ranking erstellt, das Chiralität und stereochemische Zentren berücksichtigt.
    2.  **Räumliche Verfeinerung (3D)**: Da rein topologische Verfahren oft Probleme haben, räumliche Unterschiede wie Cis/Trans-Positionen oder Axial/Äquatorial-Stellungen in Ringen zu unterscheiden (wenn diese topologisch identisch erscheinen), wird eine zusätzliche geometrische Analyse durchgeführt.
*   **Funktionsweise der 3D-Verfeinerung**:
    *   Für jedes Atom wird ein "geometrischer Fingerabdruck" basierend auf den Distanzen zu allen anderen Atomen in der stabilsten 3D-Konformation berechnet.
    *   Atome werden nur dann als symmetrisch äquivalent gruppiert, wenn sie sowohl topologisch gleich sind als auch räumlich innerhalb einer **Toleranz von 0,4 Å** liegen.
    *   **Nutzen**: Dies ermöglicht es, rotierende Gruppen (wie Phenylringe) trotz minimaler 3D-Verzerrungen zusammenzufassen, während starre geometrische Isomere (Cis/Trans) zuverlässig unterschieden werden.
*   **Symmetry Average Option**: Über die Checkbox `"Symmetry Average"` im UI kann gesteuert werden, ob die Vorhersagewerte für äquivalente Atome gemittelt werden sollen. 
    *   *Aktiviert*: Alle Atome desselben Rangs erhalten denselben gemittelten Shift-Wert. Dies entspricht der chemischen Erwartung für frei rotierende oder symmetrische Moleküle in Lösung.
    *   *Deaktiviert*: Jedes Atom behält seinen individuellen Vorhersagewert (nützlich zur Analyse von Geometrie-Effekten oder Instabilitäten der Modelle).
*   **Hinweis (Experimentell)**: Da die Symmetrie-Erkennung von der Qualität der initialen 3D-Einbettung (MMFF94) abhängt, kann es in seltenen Fällen bei sehr flexiblen Molekülen zu einer Über- oder Untersegmentierung kommen. Die Ergebnisse sollten daher im Einzelfall kritisch geprüft werden. Der Rang wird zur Kontrolle in der Spalte `"Sym. Rank"` angezeigt.
