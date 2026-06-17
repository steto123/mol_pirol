"""
symmetry_ranking.py
===================
Standalone Symmetry-Ranking für Moleküle aus SMILES.

Algorithmus (Vollständig / Hybrid):
  1. Topologisches Canonical Ranking (RDKit, inkl. Stereochemie aus SMILES)
  2. 3D-Stereochemie-Erkennung:
     - MMFF94-optimiertes Konformer wird erzeugt
     - AssignStereochemistryFrom3D kodiert E/Z-Doppelbindungen und R/S-Zentren
     - Erneutes CanonicalRankAtoms mit diesen 3D-Stereodeskriptoren
     → unterscheidet echte geometrische Isomerie (cis/trans), aber NICHT
        rotationell äquivalente Gruppen (z. B. tert-Butyl-Methyls)

Algorithmus (Einfach):
  - Nur RDKit CanonicalRankAtoms(breakTies=False) ohne 3D-Verfeinerung

Ausgabe:
  - Grafik: 2D-Strukturformel, farbcodiert nach Symmetriegruppe (matplotlib/RDKit)
  - DataFrame: Atom-Index | Element | Symmetrie-Rang
  - Vergleichsansicht: Beide Rankings nebeneinander (plot_comparison_result)
"""

from __future__ import annotations

import math
import colorsys
from collections import defaultdict

import numpy as np
try:
    from IPython.display import SVG, display as _ip_display
    _IN_JUPYTER = True
except ImportError:
    _IN_JUPYTER = False
    import matplotlib
    matplotlib.use("Agg")   # non-interactive backend – no GUI needed

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import to_hex
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Draw import rdMolDraw2D
import pandas as pd


# ---------------------------------------------------------------------------
# Kern-Algorithmus
# ---------------------------------------------------------------------------

def generate_3d_conformer(mol: Chem.Mol) -> Chem.Mol:
    """Erzeugt ein MMFF94-optimiertes 3D-Konformer (bestes aus 10 Versuchen)."""
    m = Chem.AddHs(mol, addCoords=True)
    cids = AllChem.EmbedMultipleConfs(m, numConfs=10, randomSeed=42, pruneRmsThresh=0.5)

    if not cids:
        if AllChem.EmbedMolecule(m, randomSeed=42) == -1:
            AllChem.Compute2DCoords(m)
        AllChem.MMFFOptimizeMolecule(m)
        return m

    energies = []
    for cid in cids:
        ff = AllChem.MMFFGetMoleculeForceField(
            m, AllChem.MMFFGetMoleculeProperties(m), confId=cid
        )
        if ff:
            ff.Initialize()
            ff.Minimize()
            energies.append((cid, ff.CalcEnergy()))

    if energies:
        best_cid = min(energies, key=lambda x: x[1])[0]
        new_m = Chem.Mol(m)
        new_m.RemoveAllConformers()
        new_m.AddConformer(m.GetConformer(best_cid), assignId=True)
        return new_m

    AllChem.MMFFOptimizeMolecule(m)
    return m


def calculate_symmetry_ranks(mol: Chem.Mol, m_3d: Chem.Mol | None = None,
                              num_confs: int = 50, tol: float = 0.4) -> list[int]:
    """
    Hybrides Symmetrie-Ranking (Topologisch-Geometrischer Fragment-Ansatz):

    Zwei Atome erhalten denselben Rang, wenn
    (1) sie denselben topologischen Rang haben (CanonicalRankAtoms, breakTies=False)
    UND
    (2) sie in ihrem jeweiligen starren Fragment (nach Schnitt aller rotierbaren 
        Bindungen) identische Distanzprofile aufweisen.

    Dadurch werden korrekt behandelt:
    • Homotope, rotierende Gruppen (ortho/meta am Phenylring):
      Sie befinden sich in einem in sich symmetrischen Fragment und bleiben vereint.
    • Diastereotope Gruppen an starren Doppelbindungen/Ringen (z.B. im Dithiolan):
      Sie sitzen im selben starren Fragment und weisen asymmetrische Abstände zum 
      Rest des Fragments (z.B. C=O) auf → werden korrekterweise aufgetrennt.

    Parameter
    ---------
    mol       : RDKit-Mol (Schweratome)
    m_3d, num_confs, tol : Veraltet, nur für Abwärtskompatibilität der Signatur.
    """
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    base_ranks = list(Chem.CanonicalRankAtoms(mol, breakTies=False, includeChirality=True))
    num_at = mol.GetNumAtoms()

    # 1. Rotierbare Bindungen finden (Einfachbindungen, Nicht-Ring, zwischen Schweratomen)
    rot_smarts = Chem.MolFromSmarts('[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]')
    rot_matches = mol.GetSubstructMatches(rot_smarts)
    rot_bonds = [mol.GetBondBetweenAtoms(a, b).GetIdx() for a, b in rot_matches]

    # 2. Molekül in starre Fragmente zerschneiden
    if rot_bonds:
        frags_mol = Chem.FragmentOnBonds(mol, rot_bonds)
    else:
        frags_mol = Chem.Mol(mol)

    frag_indices = Chem.GetMolFrags(frags_mol, asMols=False)
    frags_mols = Chem.GetMolFrags(frags_mol, asMols=True, sanitizeFrags=False)

    # 2D-Koordinaten für jedes Fragment vorbereiten (perfekt idealisierte Geometrie)
    frag_confs = []
    for frag in frags_mols:
        try:
            AllChem.Compute2DCoords(frag)
            frag_confs.append(frag.GetConformer())
        except Exception:
            frag_confs.append(None)

    # 3. Atome nach topologischem Basis-Rang gruppieren
    rank_groups = defaultdict(list)
    for i, r in enumerate(base_ranks):
        rank_groups[r].append(i)

    final_ranks = [-1] * num_at
    next_rank = 0

    # 4. Ränge anhand der starren Fragment-Geometrie verfeinern
    for r in sorted(rank_groups.keys()):
        atoms = rank_groups[r]
        if len(atoms) == 1:
            final_ranks[atoms[0]] = next_rank
            next_rank += 1
            continue

        signatures = {}
        for a_idx in atoms:
            for f_idx, f_atoms in enumerate(frag_indices):
                if a_idx in f_atoms:
                    frag_mol = frags_mols[f_idx]
                    conf = frag_confs[f_idx]
                    local_idx = f_atoms.index(a_idx)
                    
                    if conf is not None:
                        pos = conf.GetAtomPosition(local_idx)
                        dists = []
                        for i in range(frag_mol.GetNumAtoms()):
                            dists.append(round(pos.Distance(conf.GetAtomPosition(i)), 3))
                        signatures[a_idx] = tuple(sorted(dists))
                    else:
                        signatures[a_idx] = (0,) # Fallback falls 2D fehlschlägt
                    break

        sig_groups = defaultdict(list)
        for a_idx, sig in signatures.items():
            sig_groups[sig].append(a_idx)

        # Deterministische Rang-Zuweisung
        for sig in sorted(sig_groups.keys()):
            for a_idx in sig_groups[sig]:
                final_ranks[a_idx] = next_rank
            next_rank += 1

    return final_ranks


# ---------------------------------------------------------------------------
# Farbpalette
# ---------------------------------------------------------------------------

def _rank_color_hex(rank: int, lightness: float = 0.75, saturation: float = 0.7) -> str:
    """Goldener-Winkel-Hue-Verteilung → distinkte, helle Pastellfarben."""
    hue = (rank * 137.508) % 360.0 / 360.0
    r, g, b = colorsys.hls_to_rgb(hue, lightness, saturation)
    return to_hex((r, g, b))


# ---------------------------------------------------------------------------
# Grafische Ausgabe
# ---------------------------------------------------------------------------

def draw_symmetry_svg(mol: Chem.Mol, ranks: list[int]) -> str:
    """
    Erzeugt eine SVG-Darstellung des Moleküls (ohne H) mit
    farbcodierten Atomen nach Symmetrierang.
    Gibt den SVG-String zurück.
    """
    # Kopie mit gespeicherten Original-Indizes (vor RemoveHs)
    tm_full = Chem.Mol(mol)
    for i, atom in enumerate(tm_full.GetAtoms()):
        atom.SetIntProp("orig_idx", i)

    tm = Chem.RemoveHs(tm_full)
    for atom in tm.GetAtoms():
        orig = atom.GetIntProp("orig_idx")
        rank = ranks[orig]
        atom.SetProp("atomNote", str(orig))
        atom.SetIntProp("sym_rank", rank)

    AllChem.Compute2DCoords(tm)

    # Atom-Highlight-Farben sammeln
    highlight_atom_map: dict[int, tuple[float, float, float]] = {}
    for idx_tm, atom in enumerate(tm.GetAtoms()):
        rank = atom.GetIntProp("sym_rank")
        hex_col = _rank_color_hex(rank)
        r, g, b = [int(hex_col[i : i + 2], 16) / 255.0 for i in (1, 3, 5)]
        highlight_atom_map[idx_tm] = (r, g, b)

    from rdkit.Chem.Draw import rdMolDraw2D as rdd
    drawer = rdd.MolDraw2DSVG(600, 450)
    opts = drawer.drawOptions()
    opts.addAtomIndices = False
    opts.annotationFontScale = 0.55
    opts.padding = 0.15

    drawer.DrawMolecule(
        tm,
        highlightAtoms=list(highlight_atom_map.keys()),
        highlightAtomColors=highlight_atom_map,
        highlightBonds=[],
    )
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


def plot_symmetry_result(smiles: str, show_hydrogens: bool = False) -> pd.DataFrame:
    """
    Hauptfunktion:
      1. Parst SMILES
      2. Berechnet 3D-Konformer
      3. Berechnet Symmetrie-Ränge
      4. Zeigt SVG-Strukturformel (in Jupyter) + matplotlib-Legende
      5. Gibt DataFrame mit Atom-Index | Element | Symmetrie-Rang zurück

    Parameter
    ---------
    smiles : str
        Gültiger SMILES-String.
    show_hydrogens : bool
        Falls True, werden auch H-Atome in der Tabelle gezeigt.

    Returns
    -------
    pd.DataFrame mit Spalten: Atom_Index, Element, Symmetrie_Rang
    """
    # --- 1. Parse SMILES ---
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Ungültiger SMILES-String: '{smiles}'")

    print(f"Molekül: {Chem.MolToSmiles(mol)}")
    print(f"Atome (ohne H): {mol.GetNumAtoms()}  |  Formel: {Chem.rdMolDescriptors.CalcMolFormula(mol)}")
    print("Berechne 3D-Konformer (MMFF94)…", end=" ", flush=True)

    # --- 2. 3D-Struktur ---
    m_3d = generate_3d_conformer(mol)
    print("fertig.")

    # --- 3. Symmetrie-Ränge ---
    ranks = calculate_symmetry_ranks(mol, m_3d)

    # Anzahl einzigartiger Symmetriegruppen (nur Schwer-Atome)
    heavy_atom_ranks = [ranks[i] for i, a in enumerate(mol.GetAtoms()) if a.GetAtomicNum() > 1]
    unique_groups = len(set(heavy_atom_ranks))
    print(f"Einzigartige Symmetriegruppen (Schwer-Atome): {unique_groups}")

    # --- 4. Grafische Ausgabe ---
    svg_str = draw_symmetry_svg(mol, ranks)

    # SVG in Jupyter anzeigen
    if _IN_JUPYTER:
        _ip_display(SVG(svg_str))
    else:
        # Im Script-Modus: SVG-Datei speichern
        svg_path = "symmetry_output.svg"
        with open(svg_path, "w", encoding="utf-8") as fh:
            fh.write(svg_str)
        print(f"SVG gespeichert: {svg_path}")

    # Legende mit matplotlib
    # Normiere Ränge auf dichte Ganzzahlen (0, 1, 2, …) für konsistente Beschriftung
    unique_r = sorted(set(ranks))
    rank_norm = {r: i for i, r in enumerate(unique_r)}

    rank_to_atoms: dict[int, list[str]] = defaultdict(list)
    for i, atom in enumerate(mol.GetAtoms()):
        if show_hydrogens or atom.GetAtomicNum() > 1:
            rank_to_atoms[rank_norm[ranks[i]]].append(f"{atom.GetSymbol()}{i}")

    sorted_norm_ranks = sorted(rank_to_atoms.keys())
    fig, ax = plt.subplots(figsize=(max(4, len(sorted_norm_ranks) * 0.6 + 1), 1.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    patches = [
        mpatches.Patch(
            facecolor=_rank_color_hex(unique_r[r]),   # Farbe via Original-Rang
            edgecolor="#555",
            label=f"Rang {r}: {', '.join(rank_to_atoms[r])}",
        )
        for r in sorted_norm_ranks
    ]
    ax.legend(
        handles=patches,
        loc="center",
        ncol=max(1, len(patches) // 4),
        fontsize=8,
        frameon=True,
        title="Symmetrie-Gruppen (Farblegende)",
        title_fontsize=9,
    )
    plt.tight_layout()
    if _IN_JUPYTER:
        plt.show()
    else:
        legend_path = "legend_output.png"
        plt.savefig(legend_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Legende gespeichert: {legend_path}")

    # --- 5. DataFrame (normierte Ränge) ---
    rows = []
    for i, atom in enumerate(mol.GetAtoms()):
        if show_hydrogens or atom.GetAtomicNum() > 1:
            rows.append({
                "Atom_Index": i,
                "Element": atom.GetSymbol(),
                "Symmetrie_Rang": rank_norm[ranks[i]],
            })

    df = pd.DataFrame(rows).sort_values("Symmetrie_Rang").reset_index(drop=True)
    return df


def calculate_simple_ranks(mol: Chem.Mol) -> list[int]:
    """
    Einfaches Symmetrie-Ranking nur via RDKit CanonicalRankAtoms(breakTies=False).
    Kein 3D-Konformer, keine räumliche Verfeinerung.
    Gibt eine Liste von Rängen zurück (Länge = Anzahl Atome, nur Schwer-Atome des mol).
    """
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    return list(Chem.CanonicalRankAtoms(mol, breakTies=False, includeChirality=True))


def plot_comparison_result(
    smiles: str,
    show_hydrogens: bool = False,
) -> pd.DataFrame:
    """
    Vergleichsansicht beider Rankings für dasselbe Molekül:
      - Links:  Einfaches Ranking (CanonicalRankAtoms, breakTies=False)
      - Rechts: Hybrides 3D-Ranking (MMFF94 + 0.4 Å Toleranz)

    Gibt einen DataFrame zurück mit den Spalten:
      Atom_Index | Element | Einfacher_Rang | Hybrides_Rang | Rang_Identisch

    Parameter
    ---------
    smiles : str
        Gültiger SMILES-String.
    show_hydrogens : bool
        Falls True, werden auch H-Atome in der Tabelle gezeigt.

    Returns
    -------
    pd.DataFrame
    """
    # --- 1. Parse SMILES ---
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Ungültiger SMILES-String: '{smiles}'")

    formula = Chem.rdMolDescriptors.CalcMolFormula(mol)
    print(f"Molekül: {Chem.MolToSmiles(mol)}")
    print(f"Atome (ohne H): {mol.GetNumAtoms()}  |  Formel: {formula}")

    # --- 2. Einfaches Ranking ---
    simple_ranks = calculate_simple_ranks(mol)

    # --- 3. Hybrides 3D-Ranking ---
    print("Berechne 3D-Konformer (MMFF94)…", end=" ", flush=True)
    m_3d = generate_3d_conformer(mol)
    print("fertig.")
    hybrid_ranks = calculate_symmetry_ranks(mol, m_3d)

    # Statistiken ausgeben
    heavy_simple = [simple_ranks[i] for i, a in enumerate(mol.GetAtoms()) if a.GetAtomicNum() > 1]
    heavy_hybrid = [hybrid_ranks[i] for i, a in enumerate(mol.GetAtoms()) if a.GetAtomicNum() > 1]
    print(f"Einzigartige Gruppen – Einfach: {len(set(heavy_simple))}  |  Hybrid: {len(set(heavy_hybrid))}")

    # --- 4. Beide SVGs erzeugen ---
    svg_simple = draw_symmetry_svg(mol, simple_ranks)
    svg_hybrid = draw_symmetry_svg(mol, hybrid_ranks)

    if not _IN_JUPYTER:
        # Script-Modus: zwei SVG-Dateien speichern
        for name, svg in [("simple_ranking.svg", svg_simple), ("hybrid_ranking.svg", svg_hybrid)]:
            with open(name, "w", encoding="utf-8") as fh:
                fh.write(svg)
            print(f"SVG gespeichert: {name}")

    # --- 5. Legenden für beide Rankings ---
    import io
    import base64
    
    legend_b64_list = []
    
    for title, ranks in [
        ("Einfaches Ranking – Symmetrie-Gruppen", simple_ranks),
        ("Hybrides 3D-Ranking – Symmetrie-Gruppen", hybrid_ranks),
    ]:
        # Normiere auf dichte Ganzzahlen (0, 1, 2, …) – identisch mit dem DataFrame
        unique_r = sorted(set(ranks))
        rank_norm = {r: i for i, r in enumerate(unique_r)}

        rank_to_atoms: dict[int, list[str]] = defaultdict(list)
        for i, atom in enumerate(mol.GetAtoms()):
            if show_hydrogens or atom.GetAtomicNum() > 1:
                rank_to_atoms[rank_norm[ranks[i]]].append(f"{atom.GetSymbol()}{i}")

        sorted_norm_ranks = sorted(rank_to_atoms.keys())
        fig, ax = plt.subplots(figsize=(max(4, len(sorted_norm_ranks) * 0.6 + 1), 1.4))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        patches = [
            mpatches.Patch(
                facecolor=_rank_color_hex(unique_r[r]),   # Farbe via Original-Rang
                edgecolor="#555",
                label=f"Rang {r}: {', '.join(rank_to_atoms[r])}",
            )
            for r in sorted_norm_ranks
        ]
        ax.legend(
            handles=patches,
            loc="center",
            ncol=max(1, len(patches) // 4),
            fontsize=8,
            frameon=True,
            title=title,
            title_fontsize=9,
        )
        plt.tight_layout()
        
        if _IN_JUPYTER:
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=150, bbox_inches="tight")
            plt.close(fig)
            buf.seek(0)
            b64_str = base64.b64encode(buf.read()).decode('utf-8')
            legend_b64_list.append(b64_str)
        else:
            fn = "simple_legend.png" if "Einfach" in title else "hybrid_legend.png"
            plt.savefig(fn, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"Legende gespeichert: {fn}")

    if _IN_JUPYTER:
        from IPython.display import HTML
        # Beide SVGs und Legenden nebeneinander in einer HTML-Tabelle
        html_str = (
            "<table style='width:100%;border-collapse:collapse;'>"
            "<tr>"
            "<th style='text-align:center;padding:6px;font-family:sans-serif;font-size:14px;'"
            ">🔵 Einfaches Ranking<br><small>CanonicalRankAtoms(breakTies=False)</small></th>"
            "<th style='text-align:center;padding:6px;font-family:sans-serif;font-size:14px;'"
            ">🟢 Hybrides 3D-Ranking<br><small>AssignStereochemistryFrom3D + CanonicalRankAtoms</small></th>"
            "</tr>"
            "<tr>"
            f"<td style='text-align:center;vertical-align:top;'>{svg_simple}</td>"
            f"<td style='text-align:center;vertical-align:top;'>{svg_hybrid}</td>"
            "</tr>"
            "<tr>"
            f"<td style='text-align:center;vertical-align:top;'><img src='data:image/png;base64,{legend_b64_list[0]}'></td>"
            f"<td style='text-align:center;vertical-align:top;'><img src='data:image/png;base64,{legend_b64_list[1]}'></td>"
            "</tr>"
            "</table>"
        )
        _ip_display(HTML(html_str))

    # --- 6. Vergleichs-DataFrame ---
    # Normiere einfache Ränge auf dichte Ganzzahlen (0, 1, 2, …) für bessere Lesbarkeit
    unique_simple = sorted(set(simple_ranks))
    simple_rank_map = {r: i for i, r in enumerate(unique_simple)}

    unique_hybrid = sorted(set(hybrid_ranks))
    hybrid_rank_map = {r: i for i, r in enumerate(unique_hybrid)}

    rows = []
    for i, atom in enumerate(mol.GetAtoms()):
        if show_hydrogens or atom.GetAtomicNum() > 1:
            sr = simple_rank_map[simple_ranks[i]]
            hr = hybrid_rank_map[hybrid_ranks[i]]
            rows.append({
                "Atom_Index": i,
                "Element": atom.GetSymbol(),
                "Einfacher_Rang": sr,
                "Hybrides_Rang": hr,
                "Rang_Identisch": sr == hr,
            })

    df = pd.DataFrame(rows).sort_values(["Einfacher_Rang", "Atom_Index"]).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# CLI-Nutzung (direkt als Script)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Nutzung: python symmetry_ranking.py \"<SMILES>\"")
        print("Beispiel: python symmetry_ranking.py \"c1ccccc1\"")
        sys.exit(0)

    smiles_input = sys.argv[1]
    result_df = plot_symmetry_result(smiles_input)
    print("\n--- Symmetrie-Ranking ---")
    print(result_df.to_string(index=False))
