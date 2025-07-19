# src/preprocessing.py

"""
Module preprocessing.py

Description:
- Lecture, nettoyage et featurisation de molécules SMILES.
- Utilise RDKit pour la manipulation de molécules.
- Fournit des fonctions pour créer des objets Mol, calculer des empreintes moléculaires,
  descripteurs de Lipinski, et appliquer des filtres de qualité.
"""

from typing import Optional, List, Dict, Tuple, Any
from rdkit import RDLogger, Chem
from rdkit.Chem import AllChem, Descriptors, FilterCatalog, Draw
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit.Chem.Draw import rdMolDraw2D
# Désactive les warnings sur les SMILES invalides
RDLogger.DisableLog('rdApp.*')


def create_molecule_from_smiles(smiles: str) -> Optional[Chem.Mol]:
    """
    Convertit une chaîne SMILES en objet RDKit Mol.
    Retourne None si la conversion échoue ou si la chaîne est vide.
    """
    if not smiles or not smiles.strip():
        return None
    try:
        return Chem.MolFromSmiles(smiles)
    except Exception:
        return None


def process_smiles(smiles: str) -> Optional[str]:
    """
    Standardise un SMILES :
      - None si vide ou invalide
      - Kekulisation
      - Retourne SMILES isomérique
    """
    mol = create_molecule_from_smiles(smiles)
    if mol is None:
        return None
    try:
        Chem.Kekulize(mol, clearAromaticFlags=True)
    except Exception:
        pass
    return Chem.MolToSmiles(mol, isomericSmiles=True)


def compute_ecfp4(
    smiles: str,
    radius: int = 2,
    nBits: int = 2048
) -> Optional[List[int]]:
    """
    Calcule l'empreinte ECFP4 (Morgan fingerprint, radius=2).
    Retourne une liste de 0/1 ou None si SMILES invalide.
    """
    mol = create_molecule_from_smiles(smiles)
    if mol is None:
        return None
    fp_vect = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits)
    return list(map(int, fp_vect.ToBitString()))


def calculate_ro5_descriptors(smiles: str) -> Optional[Dict[str, float]]:
    """
    Calcule les descripteurs de Lipinski : MW, LogP, HBA, HBD.
    Retourne dict ou None si SMILES invalide.
    """
    mol = create_molecule_from_smiles(smiles)
    if mol is None:
        return None
    return {
        "MW":   Descriptors.MolWt(mol),
        "LogP": Descriptors.MolLogP(mol),
        "HBA":  Descriptors.NumHAcceptors(mol),
        "HBD":  Descriptors.NumHDonors(mol)
    }


def apply_lipinski_filter(
    df: pd.DataFrame,
    smiles_col: str = "smiles"
) -> pd.DataFrame:
    """
    Filtre les molécules selon la règle de Lipinski :
      MW ≤ 500, LogP ≤ 5, HBA ≤ 10, HBD ≤ 5
    """
    # calcule et remplace None par nan
    desc = df[smiles_col].apply(calculate_ro5_descriptors)
    desc = desc.apply(lambda x: {"MW":math.nan, "LogP":math.nan, "HBA":math.nan, "HBD":math.nan} if x is None else x)
    desc_df = pd.DataFrame(list(desc), index=df.index)
    mask = (
        (desc_df["MW"]   <= 500) &
        (desc_df["LogP"] <= 5)   &
        (desc_df["HBA"]  <= 10)  &
        (desc_df["HBD"]  <= 5)
    )
    return df[mask].copy()


def apply_pains_filter(
    df: pd.DataFrame,
    smiles_col: str = "smiles"
) -> pd.DataFrame:
    """
    Filtre les molécules contenant des motifs PAINS A/B/C.
    """
    params = FilterCatalog.FilterCatalogParams()
    for cat in ("PAINS_A","PAINS_B","PAINS_C"):
        params.AddCatalog(getattr(FilterCatalog.FilterCatalogParams.FilterCatalogs, cat))
    catalog = FilterCatalog.FilterCatalog(params)

    def is_pains(smi: str) -> bool:
        mol = create_molecule_from_smiles(smi)
        return True if (mol is None) else catalog.HasMatch(mol)

    mask = ~df[smiles_col].apply(is_pains)
    return df[mask].copy()


def compute_descriptor(
    smiles: str,
    descriptor_name: str
) -> Optional[float]:
    """
    Calcule un descripteur RDKit par nom (Descriptors.<name>).
    """
    mol = create_molecule_from_smiles(smiles)
    if mol is None:
        return None
    func = getattr(Descriptors, descriptor_name, None)
    if not func:
        raise ValueError(f"Descriptor inconnu: {descriptor_name}")
    return func(mol)


def load_sdf_file(path: str) -> pd.DataFrame:
    """
    Charge un fichier SDF et retourne un DataFrame avec une colonne 'smiles'.
    """
    supplier = Chem.SDMolSupplier(path)
    records = []
    for mol in supplier:
        if mol is None:
            continue
        smi = Chem.MolToSmiles(mol, isomericSmiles=True)
        records.append({"smiles": smi})
    return pd.DataFrame(records)


def setup_visualization_style():
    """
    Configure un style cohérent pour matplotlib/seaborn.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette(sns.color_palette('PuBu'))
    plt.rcParams['axes.titlesize'] = 18
    plt.rcParams['axes.labelsize'] = 16


def setup_rdkit_drawing() -> rdMolDraw2D.MolDrawOptions:
    """
    Configure les options de dessin RDKit pour des visualisations moléculaires.
    """
    d2d = rdMolDraw2D.MolDraw2DSVG(-1, -1)
    opts = d2d.drawOptions()
    opts.useBWAtomPalette()
    opts.setHighlightColour((0.635, 0.0, 0.145, 0.4))
    opts.baseFontSize = 1.0
    opts.additionalAtomLabelPadding = 0.15
    return opts


def draw_target_vs_hits(
    target_smiles: str,
    hits_smiles: List[str],
    scores: List[float],
    mols_per_row: int = 6,
    sub_img_size: Tuple[int, int] = (200, 200),
    legend_prefix: str = "Hit"
) -> str:
    """
    Retourne une grille SVG de la molécule cible et de ses hits.
    """
    tgt = Chem.MolFromSmiles(target_smiles)
    hit_mols = [Chem.MolFromSmiles(s) for s in hits_smiles]
    mols = [tgt] + hit_mols

    legends = ["Target"] + [f"{legend_prefix}{i+1}: {s:.2f}" for i, s in enumerate(scores)]
    svg_obj = Draw.MolsToGridImage(
        mols,
        molsPerRow=mols_per_row,
        subImgSize=sub_img_size,
        legends=legends,
        useSVG=True
    )
    # svg_obj est un IPython.core.display.SVG
    return svg_obj.data


def plot_similarity_histogram(
    scores: List[float],
    bins: int = 10,
    title: str = "Distribution des similarités",
    xlabel: str = "Tanimoto",
    ylabel: str = "Nombre de molécules",
    figsize: Tuple[int, int] = (6, 4)
):
    """
    Trace un histogramme des scores de similarité.
    """
    plt.figure(figsize=figsize)
    plt.hist(scores, bins=bins, edgecolor='black')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(axis='y', alpha=0.75)
    plt.tight_layout()
    plt.show()

