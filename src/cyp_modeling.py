# src/cyp_modeling.py

"""
Module cyp_modeling.py

Description:
- Chargement & préprocessing des données multi-label CYP450
- Train/Test split, gestion du déséquilibre
- Entraînement d’un OneVsRestClassifier ou ClassifierChain
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import ClassifierChain
from sklearn.preprocessing import MultiLabelBinarizer

def load_cyp_data(path: str, targets: list) -> pd.DataFrame:
    """
    Charge un CSV multi-label avec colonnes ['smiles', <une colonne par isozyme>].
    targets = ['CYP3A4','CYP2C9','CYP2D6'] par exemple.
    """
    df = pd.read_csv(path)
    cols = ['smiles'] + targets
    return df.dropna(subset=['smiles'])[cols].reset_index(drop=True)

def process_cyp_data(df: pd.DataFrame, feature_cols: list, target_cols: list):
    """
    À partir d’un DataFrame avec smiles + target_cols, 
    renvoie X (empireintes/features) et Y (DataFrame targets binaires).
    """
    X = df[feature_cols]
    Y = df[target_cols]
    return X, Y

def multilabel_train_test_split(X, Y, test_size: float=0.2, random_state: int=42):
    """
    Sépare X, Y en train/test.
    """
    return train_test_split(X, Y, test_size=test_size, random_state=random_state)

def train_and_evaluate_cyp(X_train, Y_train, method: str='ovr'):
    """
    Entraîne un modèle multi-label pour CYP.
    method = 'ovr' ou 'chain'.
    """
    base = RandomForestClassifier(random_state=0)
    if method == 'chain':
        clf = ClassifierChain(base, order='random', random_state=0)
    else:
        clf = OneVsRestClassifier(base)
    clf.fit(X_train, Y_train)
    return clf

def adversarial_validation(X_source, X_target, y_source_label=0, y_target_label=1):
    """
    Monte un modèle pour distinguer source vs target (évaluer shift).
    Retourne clf, X_val, y_val.
    """
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier

    X = pd.concat([X_source, X_target], ignore_index=True)
    y = [y_source_label]*len(X_source) + [y_target_label]*len(X_target)
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, stratify=y, random_state=0)
    clf = RandomForestClassifier(random_state=0).fit(X_tr, y_tr)
    return clf, X_val, y_val
