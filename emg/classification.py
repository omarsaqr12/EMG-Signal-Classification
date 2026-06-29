"""Classifiers and leave-one-out cross-validation.

Standardisation is fitted *inside* each CV fold (on the training trials only)
to avoid leaking test-set statistics, then applied to the held-out trial.
"""

from __future__ import annotations

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import LeaveOneOut
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Factory functions keep estimators fresh per fold and per call.
CLASSIFIERS = {
    "knn": lambda k=5: KNeighborsClassifier(n_neighbors=k),
    "svm": lambda: SVC(kernel="rbf", C=10.0, gamma="scale"),
    "random_forest": lambda: RandomForestClassifier(n_estimators=200, random_state=0),
    "lda": lambda: LinearDiscriminantAnalysis(),
}


def build_classifier(name="knn", **kwargs):
    """Return a ``StandardScaler -> estimator`` pipeline for ``name``."""
    if name not in CLASSIFIERS:
        raise ValueError(f"Unknown classifier {name!r}; choose from {list(CLASSIFIERS)}")
    estimator = CLASSIFIERS[name](**kwargs)
    return make_pipeline(StandardScaler(), estimator)


def leave_one_out_accuracy(X, y, name="knn", return_predictions=False, **kwargs):
    """Leave-one-trial-out CV accuracy for the chosen classifier.

    The dataset is small (tens of trials), so LOOCV is the natural,
    low-variance estimator used throughout this project.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    loo = LeaveOneOut()
    y_true, y_pred = [], []
    for train_idx, test_idx in loo.split(X):
        clf = build_classifier(name, **kwargs)
        clf.fit(X[train_idx], y[train_idx])
        y_pred.append(clf.predict(X[test_idx])[0])
        y_true.append(y[test_idx][0])
    acc = accuracy_score(y_true, y_pred)
    if return_predictions:
        return acc, np.asarray(y_true), np.asarray(y_pred)
    return acc


def best_k(X, y, k_range=range(1, 16)):
    """Sweep K for KNN and return ``(best_k, best_accuracy, {k: acc})``."""
    scores = {k: leave_one_out_accuracy(X, y, "knn", k=k) for k in k_range}
    bk = max(scores, key=scores.get)
    return bk, scores[bk], scores


def confusion(X, y, name="knn", **kwargs):
    """Leave-one-out confusion matrix and the sorted class labels."""
    _, y_true, y_pred = leave_one_out_accuracy(
        X, y, name, return_predictions=True, **kwargs
    )
    labels = np.unique(y)
    return confusion_matrix(y_true, y_pred, labels=labels), labels
