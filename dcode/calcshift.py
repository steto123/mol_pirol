import pandas as pd
import statistics
# siehe auch https://docs.python.org/3/library/statistics.html

# einfache Verschiebungsberechnung durch Ermittelung des Median 
# übergeben werden das dataframe mit den DCodes und der DCode für den gerechnet werden soll
# zurück gegeben werden der ´Mittelwert und die Anzahl der Treffer
# zur weiteren auswertung kann auch die Trefferliste zurück gegeben werden
# und später auch ein getrimmter Suchstring

def trimmed_mean(data, trim_percent=0.1):
    """
    Berechnet den getrimmten Mittelwert
    
    :param data: Liste der Werte
    :param trim_percent: Prozentsatz der zu entfernenden Werte (beidseitig)
    :return: Getrimmter Mittelwert
    """
    # Sortieren der Daten
    sorted_data = sorted(data)
    
    # Anzahl der zu entfernenden Elemente
    trim_count = int(len(data) * trim_percent)
    
    # Trimmen der Daten
    trimmed_data = sorted_data[trim_count:-trim_count] if trim_count > 0 else sorted_data
    
    # Berechnung des Durchschnitts
    return sum(trimmed_data) / len(trimmed_data)




def calcshift(dataframe,suchstring,index,dbid=0):
    calcshift=-999
    #print(suchstring)
    #wert3 = dataframe[dataframe['DCode'].str.contains(suchstring)]['Shift']
    # Es kann dbid übergeben werden, wenn Tests mit Daten aus der NMRShiftDB gemacht werden
    '''
    wert3 = dataframe[
         (dataframe['DCode'].str.contains(suchstring)) & 
         (dataframe['Database_ID'] != dbid)
    ]['Shift']    
    treffer= len(wert3)
    '''
    # rekursion zur String Kürzung
    # Initiale Bedingungen
    treffer = 0  # Setzen Sie treffer initial auf 0, um die Schleife zu starten
    min_hashes = 5  # Minimale Anzahl an '#' im Suchstring
    kuerz=0
    standardabweichung = 999
    spannweite = -999
    # Solange die Bedingungen erfüllt sind
    while treffer == 0 and suchstring.count('#') >= min_hashes:
        # Kürzen des Suchstrings bis zum letzten '#' 
        last_hash_index = suchstring.rfind('#')  # Finde den Index des letzten '#'
    
        if last_hash_index != -1:  # Überprüfen, ob ein '#' gefunden wurde
            suchstring = suchstring[:last_hash_index]  # Kürzen bis zum letzten '#'
        
            # Erneute Berechnung von wert3 und treffer
            wert3 = dataframe[
                 (dataframe['DCode'].str.contains(suchstring)) & 
                 (dataframe['Database_ID'] != dbid)
            ]['Shift']
        
            treffer = len(wert3)  # Aktualisieren der Trefferzahl
            kuerz+=1
        else:
             break  # Schleife beenden, wenn kein '#' mehr gefunden wird
    
    ## Ende der rekursion
    if (treffer > 0):
        median_wert = statistics.median(wert3)        # median als chemische Verschiebung
        # Überprüfung der Liste vor der Berechnung
        if len(wert3) < 2:
            print("Fehler: Für die Standardabweichung werden mindestens zwei Werte benötigt.")
            standardabweichung = 999
        else:
            try:
                # Standardabweichung berechnen
                standardabweichung = statistics.stdev(wert3)
        
                # Spannweite (Range)
                spannweite = max(wert3) - min(wert3)
        
                # Optional: Varianz
                varianz = statistics.variance(wert3)
        
            except TypeError:
                print("Fehler: Die Liste enthält ungültige Datentypen. Nur numerische Werte sind erlaubt.")
                standardabweichung = 999
                spannweite = -999
                varianz = 999
    else:
        median_wert = -999
        
    ### Version 20251024
    """
    Vor mehr als 50 Treffer und einer Spannweite größer 10 wird anstelle des median der 10%
    getrimmte Mittelwert benutzt
    
        
    ### Version 20251118

    Es wird ab eine größer Anzahl von Treffern eine Außreißereliminierung vorgenommen
    Dazu wird der z.score benutzt
    Ich nehme eine Mindestlänge von Wert3 von 20 an
   
    """
    if len(wert3) > 50:
        print("C- Atom Nummer: ", index)

        ### z-score Methode
        
        # Berechne den Mittelwert und die Standardabweichung
        mean = wert3.mean()
        std_dev = wert3.std()

        # Definiere einen Schwellenwert für den Z-Score
        threshold = 3  # Ein häufig verwendeter Schwellenwert entspricht 3 Standardabweichungen vom Mittelwert

        # Berechne die Z-Scores
        z_scores = (wert3 - mean) / std_dev

        # Identifiziere Ausreißer
        ausreißer = wert3[abs(z_scores) > threshold]
        
        # Entferne die Ausreißer aus wert3
        wert3_cleaned = wert3[abs(z_scores) <= threshold]    
        
        
        #Neuberechnung der Statistik
        # Standardabweichung berechnen
        standardabweichung = statistics.stdev(wert3)
        
        # Spannweite (Range)
        spannweite = max(wert3) - min(wert3)
        
        # Optional: Varianz
        varianz = statistics.variance(wert3)
        
        #Neuberechnung der chem. Verschiebung als Median
        median_wert = statistics.median(wert3_cleaned)
        
        if (spannweite>10):
            median_wert = trimmed_mean(wert3_cleaned, 0.1)
        
        print("Ausreißer:")
        print(ausreißer)
    
    
    #Rückgabewerte berechnen
    calcshift = median_wert
    return calcshift, treffer, kuerz, spannweite, standardabweichung