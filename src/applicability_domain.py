# -------------------------
# src/applicability_domain.py
# -------------------------
"""
Module applicability_domain.py

Description:
- Domaines d'applicabilité: bounding box & convex hull
"""
import numpy as np
from scipy.spatial import ConvexHull, Delaunay

def bounding_box_ad(X_ref: np.ndarray, X: np.ndarray) -> np.ndarray:
    """
    Retourne un bool array indiquant pour chaque point de X s'il est dans l'hyper-rectangle défini par X_ref.
    """
    mins = X_ref.min(axis=0)
    maxs = X_ref.max(axis=0)
    return np.all((X >= mins) & (X <= maxs), axis=1)


def convex_hull_ad(X_ref: np.ndarray, X: np.ndarray) -> np.ndarray:
    """
    Retourne un bool array indiquant si chaque point de X est dans l'enveloppe convexe de X_ref.
    """
    hull = ConvexHull(X_ref)
    delaunay = Delaunay(X_ref[hull.vertices])
    return delaunay.find_simplex(X) >= 0
