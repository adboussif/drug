# src/similarity.py

"""
Module similarity.py

Description:
- Fonctions de virtual screening ligand-based.
- Chargement de librairies moléculaires, application de filtres,
  calcul de similarités (Tanimoto, Dice), sélection et analyse des top-n composés.
- Support d’empreintes ECFP4, ECFP6 et MACCS.
"""

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, FilterCatalog, MACCSkeys

# --- Chargement de librairies moléculaires ---

def load_mol_library(path: str, file_type: str = 'sdf') -> pd.DataFrame:
    """
    Charge une librairie moléculaire depuis un fichier SDF ou CSV
    et renvoie un DataFrame avec colonnes ['smiles','mol'].
    """
    records = []
    if file_type.lower() == 'sdf':
        supplier = Chem.SDMolSupplier(path)
        for mol in supplier:
            if mol is None: continue
            smi = Chem.MolToSmiles(mol, isomericSmiles=True)
            records.append({'smiles': smi, 'mol': mol})
    else:
        df = pd.read_csv(path)
        for smi in df['smiles']:
            mol = Chem.MolFromSmiles(smi)
            if mol is None: continue
            records.append({'smiles': smi, 'mol': mol})
    return pd.DataFrame(records)


# --- Filtres qualité (PAINS, Glaxo) ---

def apply_pains_filter(df: pd.DataFrame, smiles_col: str = 'smiles') -> pd.DataFrame:
    """
    Filtre les molécules contenant des motifs PAINS (A, B, C).
    """
    params = FilterCatalog.FilterCatalogParams()
    params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_A)
    params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_B)
    params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_C)
    catalog = FilterCatalog.FilterCatalog(params)

    def is_pains(smi):
        mol = Chem.MolFromSmiles(smi)
        return catalog.HasMatch(mol) if mol else True

    mask = ~df[smiles_col].apply(is_pains)
    return df[mask].copy()


def apply_glaxo_filters(df: pd.DataFrame, alerts_df: pd.DataFrame, mol_col: str = 'mol') -> pd.DataFrame:
    """
    Applique les filtres structuraux Glaxo (A, B, C) contenus dans alerts_df.
    alerts_df doit contenir colonnes ['alert_name','pattern'].
    """
    params = FilterCatalog.FilterCatalogParams()
    catalog = FilterCatalog.FilterCatalog(params)
    for _, row in alerts_df.iterrows():
        catalog.AddEntry(
            FilterCatalog.FilterCatalogEntry(
                row['alert_name'],
                row['pattern']
            )
        )
    mask = ~df[mol_col].apply(lambda m: bool(catalog.GetFirstMatch(m)))
    return df[mask].copy()


# --- Empreintes moléculaires alternatives ---

def compute_fingerprint(smiles: str, radius: int = 2, nBits: int = 2048) -> list:
    """
    ECFP4 (Morgan radius=2) sous forme de liste 0/1.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits)
    return list(map(int, fp.ToBitString()))


def compute_ecfp6(smiles: str, nBits: int = 2048) -> list:
    """
    ECFP6 (Morgan radius=3) sous forme de liste 0/1.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits)
    return list(map(int, fp.ToBitString()))


def compute_maccs(smiles: str) -> list:
    """
    MACCS keys (166 bits) sous forme de liste 0/1.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    fp = MACCSkeys.GenMACCSKeys(mol)
    return list(map(int, fp.ToBitString()))


def add_fingerprints(
    df: pd.DataFrame,
    smiles_col: str = 'smiles',
    fp_type: str = 'ecfp4',
    nBits: int = 2048
) -> pd.DataFrame:
    """
    Ajoute une colonne 'fp' selon le type demandé :
      - 'ecfp4', 'ecfp6', 'maccs'
    """
    df = df.copy()
    t = fp_type.lower()
    if t == 'ecfp4':
        df['fp'] = df[smiles_col].apply(lambda s: compute_fingerprint(s, 2, nBits))
    elif t == 'ecfp6':
        df['fp'] = df[smiles_col].apply(lambda s: compute_ecfp6(s, nBits))
    elif t == 'maccs':
        df['fp'] = df[smiles_col].apply(compute_maccs)
    else:
        raise ValueError(f"FP type '{fp_type}' non supporté")
    return df


def generate_morgan_fingerprints(
    df: pd.DataFrame,
    smiles_col: str = 'smiles',
    radius: int = 2,
    nBits: int = 2048
) -> pd.DataFrame:
    """
    Ajoute une colonne 'fp' avec des empreintes ECFP(r, nBits).
    """
    fps = df[smiles_col].apply(lambda s: compute_fingerprint(s, radius, nBits))
    df2  = df.copy()
    df2['fp'] = fps.tolist()
    return df2


# --- Similarité moléculaire ---

def calculate_tanimoto(fp1: list, fp2: list) -> float:
    """
    Similarité de Tanimoto entre deux vecteurs binaires.
    """
    if fp1 is None or fp2 is None:
        return 0.0
    a = np.array(fp1, dtype=int)
    b = np.array(fp2, dtype=int)
    return float(np.dot(a, b) / float(a.sum() + b.sum() - np.dot(a, b)))


def calculate_dice(fp1: list, fp2: list) -> float:
    """
    Similarité de Dice entre deux vecteurs binaires.
    """
    if fp1 is None or fp2 is None:
        return 0.0
    a = np.array(fp1, dtype=int)
    b = np.array(fp2, dtype=int)
    return float(2 * np.dot(a, b) / float(a.sum() + b.sum()))


def compute_similarity_to_query(
    df: pd.DataFrame,
    target_fp: list,
    fp_col: str = 'fp'
) -> pd.DataFrame:
    """
    Calcule 'tanimoto' et 'dice' vs target_fp, retourne un nouveau DataFrame.
    """
    df2 = df.copy()
    df2['tanimoto'] = df2[fp_col].apply(lambda x: calculate_tanimoto(x, target_fp))
    #df2['dice']     = df2[fp_col].apply(lambda x: calculate_dice(x, target_fp))
    return df2


def select_top_n(
    df: pd.DataFrame,
    metric: str = 'tanimoto',
    n: int = 10
) -> pd.DataFrame:
    """
    Sélectionne les n meilleures molécules selon la colonne metric.
    """
    return df.sort_values(metric, ascending=False).head(n).reset_index(drop=True)


def search_similar_compounds(
    df: pd.DataFrame,
    query_smiles: str,
    fp_type: str = 'ecfp4',
    nBits: int = 2048,
    metric: str = 'tanimoto',
    top_n: int = 10
) -> pd.DataFrame:
    """
    Pipeline complet :
      1) add_fingerprints(df, fp_type)
      2) calcul de similarités vs query_smiles
      3) top_n selon metric
    """
    # 1) Empreintes
    df2 = add_fingerprints(df, fp_type=fp_type, nBits=nBits)
    # 2) Empreinte requête
    if fp_type.lower() == 'ecfp4':
        qfp = compute_fingerprint(query_smiles, 2, nBits)
    elif fp_type.lower() == 'ecfp6':
        qfp = compute_ecfp6(query_smiles, nBits)
    else:
        qfp = compute_maccs(query_smiles)
    # 3) Calcul sim
    df_sim = compute_similarity_to_query(df2, qfp, fp_col='fp')
    # 4) Top-n
    return select_top_n(df_sim, metric=metric, n=top_n)


def analyze_similarity_scores(df: pd.DataFrame, metric: str = 'tanimoto') -> None:
    """
    Affiche min, max, mean, median pour la colonne metric.
    """
    scores = df[metric].dropna()
    print(f"{metric} scores:")
    print(f" Min:    {scores.min():.4f}")
    print(f" Max:    {scores.max():.4f}")
    print(f" Mean:   {scores.mean():.4f}")
    print(f" Median: {scores.median():.4f}")
