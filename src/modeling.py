# src/modeling.py

"""
Module modeling.py

Description:
- Fonctions d'entraînement de différents modèles pour hERG et autres cibles.
- Contient SGD, LogisticRegression, RandomForest, SVM.
- Permet recherche d'hyperparamètres (GridSearchCV) et retourne des pipelines prêts à l'emploi.
"""

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, matthews_corrcoef,
    RocCurveDisplay, PrecisionRecallDisplay
)
import matplotlib.pyplot as plt


def load_herg_blockers_data(path: str) -> pd.DataFrame:
    """
    Charge un CSV hERG avec colonnes ['smiles','label'] (0/1).
    """
    df = pd.read_csv(path)
    return df.dropna(subset=['smiles','label']).reset_index(drop=True)


def split_data(X, y, test_size: float=0.2, random_state: int=42):
    """
    Sépare X, y en ensembles train et test.
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def train_sgd_classifier(X_train, y_train, param_grid: dict=None):
    """
    Entraîne un SGDClassifier (loss='log_loss') avec StandardScaler.
    Si param_grid est fourni, utilise GridSearchCV en séquentiel (n_jobs=1).
    """
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('sgd', SGDClassifier(loss='log_loss', max_iter=1000, tol=1e-3, random_state=0))
    ])
    if param_grid:
        gs = GridSearchCV(pipe, param_grid, cv=5, scoring='roc_auc', n_jobs=1, verbose=1)
        gs.fit(X_train, y_train)
        return gs.best_estimator_
    pipe.fit(X_train, y_train)
    return pipe


def train_logistic_regression(X_train, y_train, param_grid: dict=None):
    """
    Entraîne une régression logistique (LogisticRegression) avec StandardScaler.
    Si param_grid est fourni, utilise GridSearchCV.
    """
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=1000, solver='liblinear', random_state=0))
    ])
    if param_grid:
        gs = GridSearchCV(pipe, param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=1)
        gs.fit(X_train, y_train)
        return gs.best_estimator_
    pipe.fit(X_train, y_train)
    return pipe


def train_rf_classifier(X_train, y_train, param_grid: dict=None):
    """
    Entraîne un RandomForestClassifier.
    Si param_grid est fourni, utilise GridSearchCV.
    """
    rf = RandomForestClassifier(random_state=0)
    if param_grid:
        gs = GridSearchCV(rf, param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=1)
        gs.fit(X_train, y_train)
        return gs.best_estimator_
    rf.fit(X_train, y_train)
    return rf


def train_svm_classifier(X_train, y_train, param_grid: dict=None):
    """
    Entraîne un SVM (SVC) avec StandardScaler.
    Si param_grid est fourni, utilise GridSearchCV.
    Retourne un pipeline avec probability=True.
    """
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('svc', SVC(kernel='rbf', probability=True, random_state=0))
    ])
    if param_grid:
        gs = GridSearchCV(pipe, param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=1)
        gs.fit(X_train, y_train)
        return gs.best_estimator_
    pipe.fit(X_train, y_train)
    return pipe


def train_all_models(X_train, y_train, param_grids: dict=None):
    """
    Entraîne tous les modèles disponibles et renvoie un dict {name: trained_model}.
    param_grids: mapping model name -> param_grid dict (ou None).
    """
    models = {}
    # SGD
    models['SGD'] = train_sgd_classifier(
        X_train, y_train,
        param_grid=(param_grids.get('SGD') if param_grids else None)
    )
    # Logistic Regression
    models['LogReg'] = train_logistic_regression(
        X_train, y_train,
        param_grid=(param_grids.get('LogReg') if param_grids else None)
    )
    # Random Forest
    models['RF'] = train_rf_classifier(
        X_train, y_train,
        param_grid=(param_grids.get('RF') if param_grids else None)
    )
    # SVM
    models['SVM'] = train_svm_classifier(
        X_train, y_train,
        param_grid=(param_grids.get('SVM') if param_grids else None)
    )
    return models

def evaluate_model(model, X_test, y_test, feature_names=None, plot_curves: bool=True):
    """
    Calcule et affiche les métriques et trace ROC/PR si souhaité.
    Retourne un dict de scores.
    """
    # Prédictions et scores
    y_pred  = model.predict(X_test)
    y_score = (model.predict_proba(X_test)[:,1]
               if hasattr(model, 'predict_proba')
               else model.decision_function(X_test))

    # Métriques
    scores = {
        'accuracy':    accuracy_score(y_test, y_pred),
        'precision':   precision_score(y_test, y_pred),
        'recall':      recall_score(y_test, y_pred),
        'f1':          f1_score(y_test, y_pred),
        'roc_auc':     roc_auc_score(y_test, y_score),
        'pr_auc':      average_precision_score(y_test, y_score),
        'mcc':         matthews_corrcoef(y_test, y_pred)
    }

    # Affichage
    print(f"Metrics for {model.__class__.__name__}:")
    for k,v in scores.items(): print(f"  {k:>10}: {v:.3f}")

    # Courbes
    if plot_curves:
        fig, ax = plt.subplots(1,2, figsize=(12,5))
        RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax[0])
        ax[0].set_title('ROC Curve')
        PrecisionRecallDisplay.from_estimator(model, X_test, y_test, ax=ax[1])
        ax[1].set_title('Precision‑Recall Curve')
        plt.tight_layout()
        plt.show()

    # Importance CAse RF
    if feature_names is not None and hasattr(model, 'feature_importances_'):
        imp = pd.Series(model.feature_importances_, index=feature_names)
        print('\nFeature importances:')
        display(imp.sort_values(ascending=False))

    return scores


def save_model(model, path: str = 'artifacts/model.pkl'):
    """
    Sauvegarde le modèle entraîné avec joblib.
    """
    joblib.dump(model, path)
    print(f"Model saved to {path}")