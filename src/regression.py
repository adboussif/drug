# src/regression.py

"""
Module regression.py

Description:
- Entraînement de modèles de régression pour la solubilité
- Split scaffold-aware
"""

import numpy as np
from sklearn.linear_model import LinearRegression, RANSACRegressor, HuberRegressor
from sklearn.svm import SVR

def train_regressor(X_train, y_train):
    """
    Entraîne plusieurs modèles de régression et les renvoie dans un dict.
    """
    models = {}
    models['linear'] = LinearRegression().fit(X_train, y_train)
    models['svr']    = SVR().fit(X_train, y_train)
    models['ransac'] = RANSACRegressor().fit(X_train, y_train)
    models['huber']  = HuberRegressor().fit(X_train, y_train)
    return models

def split_scaffold(df: pd.DataFrame, smiles_col: str='smiles', test_size: float=0.2, random_state: int=42):
    """
    Split train/test en se basant sur le scaffold (Murcko) pour éviter le leakage.
    Nécessite RDKit et MurckoScaffold.
    """
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    scaffolds = df[smiles_col].apply(lambda s: Chem.MolToSmiles(
        MurckoScaffold.GetScaffoldForMol(Chem.MolFromSmiles(s)), True
    ))
    unique, counts = np.unique(scaffolds, return_counts=True)
    # Simple split sur scaffolds majoritaires
    train_idx = df.index[scaffolds.isin(unique[counts>1])]
    test_idx  = df.index.difference(train_idx)
    X_train = df.loc[train_idx]
    X_test  = df.loc[test_idx]
    return X_train, X_test
