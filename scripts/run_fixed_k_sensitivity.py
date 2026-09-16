"""Evaluate a specified K without selecting it on the evaluation folds.

This is a within-subject sensitivity analysis, NOT an independent-subject test.
The original K-sweep results remain available as exploratory artifacts.

Usage: python scripts/run_fixed_k_sensitivity.py --k 5
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from emg import apply_filter, extract_feature_matrix, load_subject, segment_trials  # noqa: E402
from emg.classification import leave_one_out_accuracy  # noqa: E402


def evaluate(k: int, subjects=(1, 2, 3)) -> list[dict]:
    """Use one user-specified K for all subjects and feature domains."""
    if k < 1:
        raise ValueError("K must be a positive integer")
    rows = []
    for subject_id in subjects:
        subject = load_subject(subject_id)
        filtered = apply_filter(subject.emg, kind="bandpass", fs=subject.fs)
        trials = segment_trials(filtered, subject.stimulus, window_samples=300)
        for domain in ("time", "frequency", "combined"):
            features, labels = extract_feature_matrix(trials, fs=subject.fs, domain=domain)
            if len(labels) <= k:
                raise ValueError(
                    f"Subject {subject_id}: K={k} needs at least {k + 1} trials; "
                    f"found {len(labels)}"
                )
            accuracy = leave_one_out_accuracy(features, labels, name="knn", k=k)
            rows.append({
                "subject": subject_id,
                "domain": domain,
                "k": k,
                "accuracy": round(float(accuracy), 4),
                "n_trials": len(labels),
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=5,
                        help="Fixed K for every subject and feature domain (default: 5)")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results" / "fixed_k_sensitivity.csv")
    args = parser.parse_args()
    rows = evaluate(args.k)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("subject", "domain", "k", "accuracy", "n_trials"))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(f"Subject {row['subject']}: {row['domain']:<9} "
              f"K={row['k']} accuracy={row['accuracy']:.3f} n={row['n_trials']}")
    print(f"Saved {args.output}; within-subject results, not unseen-subject performance.")


if __name__ == "__main__":
    main()
