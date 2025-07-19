# src/evaluation.py

"""
Module evaluation.py

Description:
- M�triques et visualisations pour classificaton et r�gression
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    roc_curve, precision_recall_curve, auc,
    matthews_corrcoef, f1_score, hamming_loss,
    confusion_matrix
)
from sklearn.calibration import calibration_curve

def plot_roc_pr(y_true, y_score):
    """
    Trace ROC et PR curves avec AUC.
    """
    fpr, tpr, _ = roc_curve(y_true, y_score)
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    pr_auc  = auc(rec, prec)

    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC AUC={roc_auc:.2f}')
    plt.plot(rec, prec, label=f'PR AUC={pr_auc:.2f}')
    plt.xlabel('False Positive Rate / Recall')
    plt.ylabel('True Positive Rate / Precision')
    plt.legend()
    plt.show()

def compute_mcc(y_true, y_pred):
    """Matthews Correlation Coefficient."""
    return matthews_corrcoef(y_true, y_pred)

def compute_multilabel_metrics(Y_true, Y_pred):
    """
    F1-micro, F1-macro, Hamming loss.
    """
    return {
        'f1_micro': f1_score(Y_true, Y_pred, average='micro'),
        'f1_macro': f1_score(Y_true, Y_pred, average='macro'),
        'hamming_loss': hamming_loss(Y_true, Y_pred)
    }

def plot_confusion_matrix(y_true, y_pred, labels=None):
    """
    Trace la matrice de confusion.
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    plt.xticks(np.arange(len(labels)), labels, rotation=45)
    plt.yticks(np.arange(len(labels)), labels)
    plt.xlabel('Prédit')
    plt.ylabel('Vrai')
    plt.title('Matrice de confusion')
    plt.show()

def plot_calibration_curve(y_true, y_prob, n_bins=10):
    """
    Trace la calibration curve (reliability diagram).
    """
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
    plt.figure()
    plt.plot(prob_pred, prob_true, marker='o')
    plt.plot([0,1],[0,1],'--')
    plt.xlabel('Prob prédite')
    plt.ylabel('Prob vraie')
    plt.show()

def plot_learning_curve(train_scores, val_scores, train_sizes):
    """
    Trace la learning curve (bias-variance).
    """
    plt.figure()
    plt.plot(train_sizes, np.mean(train_scores, axis=1), label='Train')
    plt.plot(train_sizes, np.mean(val_scores, axis=1), label='Val')
    plt.xlabel('Taille du train')
    plt.ylabel('Score')
    plt.legend()
    plt.show()
