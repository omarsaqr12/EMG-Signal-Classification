"""Nested, stratified, within-subject KNN evaluation of the existing EMG pipeline.

Outer folds estimate performance; K is selected *only* in the inner folds.
This reduces K-selection bias in the original exploratory LOOCV results, but
is NOT an unseen-person, unseen-session, or prospective evaluation.

Run: python scripts/run_nested_cv.py --output results/nested_cv_results.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from emg import apply_filter, extract_feature_matrix, load_subject, segment_trials  # noqa: E402


def nested_accuracy(X, y, k_values=range(1, 16), outer_splits=5, inner_splits=3):
    """Return held-out outer accuracy and selected inner-fold K values.

    The caller must fix features/filter/window and K search range independently
    of the results. Standardization is fitted afresh inside every inner fold.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    if X.ndim != 2 or y.ndim != 1 or len(X) != len(y) or len(y) == 0:
        raise ValueError("Expected nonempty X (n, features) and y (n,) with matching n")
    if not np.isfinite(X).all():
        raise ValueError("Feature matrix contains NaN or infinity")
    k_values = tuple(int(k) for k in k_values)
    if not k_values or min(k_values) < 1:
        raise ValueError("K values must be positive and nonempty")
    if outer_splits < 2 or inner_splits < 2:
        raise ValueError("Both cross-validation levels need at least two folds")
    _, counts = np.unique(y, return_counts=True)
    if len(counts) < 2 or min(counts) < outer_splits:
        raise ValueError("Each of at least two classes needs >= outer_splits trials")

    outer = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=0)
    true_labels, predictions, selected_k = [], [], []
    for train, test in outer.split(X, y):
        _, inner_counts = np.unique(y[train], return_counts=True)
        if min(inner_counts) < inner_splits:
            raise ValueError("An outer training fold lacks enough trials for inner CV")
        # A KNN fit on the smallest inner training split needs at least max(K) rows.
        minimum_train = len(train) - (len(train) + inner_splits - 1) // inner_splits
        if max(k_values) > minimum_train:
            raise ValueError(f"K={max(k_values)} exceeds minimum inner training size {minimum_train}")
        model = make_pipeline(StandardScaler(), KNeighborsClassifier())
        inner = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=1)
        search = GridSearchCV(model, {"kneighborsclassifier__n_neighbors": k_values},
                              cv=inner, scoring="accuracy", refit=True, n_jobs=1)
        search.fit(X[train], y[train])
        selected_k.append(int(search.best_params_["kneighborsclassifier__n_neighbors"]))
        predictions.extend(search.predict(X[test]))
        true_labels.extend(y[test])
    return float(accuracy_score(true_labels, predictions)), selected_k


def evaluate(subjects=(1, 2, 3)):
    rows = []
    for subject_id in subjects:
        subject = load_subject(subject_id)
        filtered = apply_filter(subject.emg, kind="bandpass", fs=subject.fs)
        trials = segment_trials(filtered, subject.stimulus, window_samples=300)
        for domain in ("time", "frequency", "combined"):
            X, y = extract_feature_matrix(trials, fs=subject.fs, domain=domain)
            accuracy, selected = nested_accuracy(X, y)
            rows.append({"subject": subject_id, "domain": domain,
                         "nested_accuracy": round(accuracy, 4),
                         "n_trials": len(y), "outer_folds": 5, "inner_folds": 3,
                         "selected_k_by_outer_fold": ";".join(map(str, selected))})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "nested_cv_results.csv")
    args = parser.parse_args()
    rows = evaluate()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(f"Subject {row['subject']} {row['domain']:<9} "
              f"nested accuracy={row['nested_accuracy']:.3f}, "
              f"K by outer fold={row['selected_k_by_outer_fold']}")
    print(f"Saved {args.output}. Within-subject only; not unseen-subject accuracy.")


if __name__ == "__main__":
    main()
