"""Ablation studies for the signal-conditioning front end and classifier.

Three studies, each averaged across subjects and saved as a CSV + figure:

  1. Filter design      none / high-pass / band-pass / band-pass + notch
  2. Window length      1.0 s ... 3.0 s
  3. Classifier         KNN / SVM / random forest / LDA

    python scripts/run_experiments.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from emg import apply_filter, extract_feature_matrix, load_subject, segment_trials  # noqa: E402
from emg.classification import best_k, leave_one_out_accuracy  # noqa: E402

SUBJECTS = (1, 2, 3)
RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"

FILTERS = ("none", "highpass", "bandpass", "bandpass_notch")
WINDOWS = (100, 150, 200, 250, 300)
CLASSIFIERS = ("knn", "svm", "random_forest", "lda")
DEFAULT_WINDOW = 300
DEFAULT_FILTER = "bandpass"
DOMAIN = "combined"


def _features(subject_id, filter_kind, window):
    sub = load_subject(subject_id)
    filtered = apply_filter(sub.emg, kind=filter_kind, fs=sub.fs)
    trials = segment_trials(filtered, sub.stimulus, window_samples=window)
    return extract_feature_matrix(trials, fs=sub.fs, domain=DOMAIN)


def experiment_filters():
    rows = []
    for sid in SUBJECTS:
        for kind in FILTERS:
            X, y = _features(sid, kind, DEFAULT_WINDOW)
            _, acc, _ = best_k(X, y)
            rows.append({"subject": sid, "filter": kind, "accuracy": round(acc, 4)})
    return rows


def experiment_windows():
    rows = []
    for sid in SUBJECTS:
        for win in WINDOWS:
            X, y = _features(sid, DEFAULT_FILTER, win)
            _, acc, _ = best_k(X, y)
            rows.append({"subject": sid, "window_samples": win,
                         "window_seconds": win / 100, "accuracy": round(acc, 4)})
    return rows


def experiment_classifiers():
    rows = []
    for sid in SUBJECTS:
        X, y = _features(sid, DEFAULT_FILTER, DEFAULT_WINDOW)
        for clf in CLASSIFIERS:
            if clf == "knn":
                _, acc, _ = best_k(X, y)
            else:
                acc = leave_one_out_accuracy(X, y, clf)
            rows.append({"subject": sid, "classifier": clf, "accuracy": round(acc, 4)})
    return rows


def _mean_by(rows, key, order):
    return [float(np.mean([r["accuracy"] for r in rows if r[key] == v])) for v in order]


def _write_csv(rows, name, fields):
    path = RESULTS_DIR / name
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _bar(order, values, labels, title, ylabel, fname, xlabel=""):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.bar(range(len(order)), values, color="#3b76b0")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    for i, v in enumerate(values):
        ax.text(i, v + 0.01, f"{v:.2f}", ha="center")
    fig.tight_layout()
    fig.savefig(FIG_DIR / fname, dpi=130)
    plt.close(fig)


def _window_lines(rows):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for sid in SUBJECTS:
        accs = [next(r["accuracy"] for r in rows
                     if r["subject"] == sid and r["window_samples"] == w)
                for w in WINDOWS]
        ax.plot([w / 100 for w in WINDOWS], accs, marker="o", label=f"Subject {sid}")
    mean = _mean_by(rows, "window_samples", WINDOWS)
    ax.plot([w / 100 for w in WINDOWS], mean, marker="s", linewidth=2.5,
            color="black", label="Mean")
    ax.set_xlabel("Window length (s)")
    ax.set_ylabel("LOOCV accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("Accuracy vs. window length (band-pass, combined features)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "experiment_windows.png", dpi=130)
    plt.close(fig)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("1) Filter-design comparison")
    f_rows = experiment_filters()
    _write_csv(f_rows, "experiment_filters.csv", ["subject", "filter", "accuracy"])
    f_mean = _mean_by(f_rows, "filter", FILTERS)
    _bar(FILTERS, f_mean,
         ["None", "High-pass", "Band-pass", "Band-pass\n+ notch"],
         "Filter design vs. mean accuracy (combined features, KNN)",
         "Mean LOOCV accuracy", "experiment_filters.png")
    for kind, m in zip(FILTERS, f_mean):
        print(f"   {kind:<16} mean accuracy={m:.3f}")

    print("\n2) Window-length comparison")
    w_rows = experiment_windows()
    _write_csv(w_rows, "experiment_windows.csv",
               ["subject", "window_samples", "window_seconds", "accuracy"])
    _window_lines(w_rows)
    for w, m in zip(WINDOWS, _mean_by(w_rows, "window_samples", WINDOWS)):
        print(f"   {w / 100:.1f}s window  mean accuracy={m:.3f}")

    print("\n3) Classifier comparison")
    c_rows = experiment_classifiers()
    _write_csv(c_rows, "experiment_classifiers.csv",
               ["subject", "classifier", "accuracy"])
    c_mean = _mean_by(c_rows, "classifier", CLASSIFIERS)
    _bar(CLASSIFIERS, c_mean, ["KNN", "SVM", "Random\nForest", "LDA"],
         "Classifier vs. mean accuracy (band-pass, combined features)",
         "Mean LOOCV accuracy", "experiment_classifiers.png")
    for clf, m in zip(CLASSIFIERS, c_mean):
        print(f"   {clf:<14} mean accuracy={m:.3f}")

    print(f"\nSaved CSVs to {RESULTS_DIR.relative_to(ROOT)} and figures to "
          f"{FIG_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
