"""Headline classification run.

For every subject and every feature domain (time / frequency / combined) this
sweeps K for the KNN classifier under leave-one-trial-out CV, writes a tidy
results table, and saves per-subject confusion matrices and an accuracy
summary figure.

    python scripts/run_classification.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: render straight to PNG
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from emg import (  # noqa: E402
    apply_filter,
    extract_feature_matrix,
    load_subject,
    segment_trials,
)
from emg.classification import best_k, confusion  # noqa: E402

SUBJECTS = (1, 2, 3)
DOMAINS = ("time", "frequency", "combined")
WINDOW_SAMPLES = 300       # 3.0 s at 100 Hz
FILTER = "bandpass"        # signal-conditioning front end
RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"
CLASS_NAMES = {0: "Rest", 1: "Gesture 1", 2: "Gesture 2"}


def run_subject(subject_id):
    sub = load_subject(subject_id)
    filtered = apply_filter(sub.emg, kind=FILTER, fs=sub.fs)
    trials = segment_trials(filtered, sub.stimulus, window_samples=WINDOW_SAMPLES)

    rows = []
    per_domain = {}
    for domain in DOMAINS:
        X, y = extract_feature_matrix(trials, fs=sub.fs, domain=domain)
        bk, acc, _ = best_k(X, y)
        rows.append({
            "subject": subject_id,
            "domain": domain,
            "best_k": bk,
            "accuracy": round(acc, 4),
            "n_trials": len(y),
        })
        per_domain[domain] = (X, y, bk, acc)
    return rows, per_domain


def save_confusion(subject_id, X, y, k):
    cm, labels = confusion(X, y, "knn", k=k)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    names = [CLASS_NAMES.get(int(l), str(l)) for l in labels]
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_yticklabels(names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Subject {subject_id} - combined features (K={k})")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out = FIG_DIR / f"confusion_subject_{subject_id}.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def save_summary(all_rows):
    subjects = sorted({r["subject"] for r in all_rows})
    width = 0.25
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for i, domain in enumerate(DOMAINS):
        accs = [next(r["accuracy"] for r in all_rows
                     if r["subject"] == s and r["domain"] == domain)
                for s in subjects]
        ax.bar(np.arange(len(subjects)) + i * width, accs, width,
               label=domain.capitalize())
    ax.set_xticks(np.arange(len(subjects)) + width)
    ax.set_xticklabels([f"Subject {s}" for s in subjects])
    ax.set_ylabel("LOOCV accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("Best KNN accuracy by feature domain")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = FIG_DIR / "accuracy_summary.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    all_rows = []
    print(f"Front end: {FILTER} filter | window: {WINDOW_SAMPLES} samples "
          f"({WINDOW_SAMPLES / 100:.1f}s) | classifier: KNN (LOOCV)\n")
    for sid in SUBJECTS:
        rows, per_domain = run_subject(sid)
        all_rows.extend(rows)
        for r in rows:
            print(f"  Subject {r['subject']} | {r['domain']:<9} | "
                  f"best K={r['best_k']:<2} | accuracy={r['accuracy']:.3f}")
        X, y, bk, _ = per_domain["combined"]
        save_confusion(sid, X, y, bk)
        print()

    csv_path = RESULTS_DIR / "classification_results.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["subject", "domain", "best_k", "accuracy", "n_trials"])
        writer.writeheader()
        writer.writerows(all_rows)
    save_summary(all_rows)

    best = max(all_rows, key=lambda r: r["accuracy"])
    print(f"Saved {csv_path.relative_to(ROOT)} and figures to "
          f"{FIG_DIR.relative_to(ROOT)}")
    print(f"Best overall: subject {best['subject']} | {best['domain']} | "
          f"K={best['best_k']} | accuracy={best['accuracy']:.3f}")


if __name__ == "__main__":
    main()
