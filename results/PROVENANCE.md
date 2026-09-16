# Evaluation-result provenance

The pre-existing `classification_results.csv` and `experiment_*.csv` files are **historical exploratory** results: K and/or configuration were selected using the same recordings used for reporting. The corresponding figures were retained unmodified.

`nested_cv_results.csv` and `fixed_k_sensitivity.csv` reproduce the numeric rows printed by the [GitHub Actions run of commit `47385d20`](https://github.com/omarsaqr12/EMG-Signal-Classification/actions/runs/35084982286), September 16, 2026, using the three version-controlled `data/subject_*.mat` files. The CSV rows were transcribed from that run's output and should be checked against an automatically regenerated file (see the PR workflow). That run used Ubuntu 24.04, Python 3.11.16, NumPy 2.4.6, SciPy 1.17.1 and scikit-learn 1.9.1. The code and datasets are available in this repository; these are fresh script executions, not independent replication by another team.

- **Nested CV:** per participant, 5 stratified outer folds and 3 stratified inner folds, both shuffled with fixed seeds (0 outer, 1 inner). Inner grid searches K=1–15 and fits scaling only on the inner training split. The selected model is refit on each outer training split and predicts its outer test fold. The exported `selected_k_by_outer_fold` entries record the five inner-selected K values.
- **Fixed K:** K=5 is specified in advance for the command (not tuned on its own LOOCV folds). It remains a sensitivity analysis: that configuration is not justified as independently selected before the broader experiments.
- **Both:** 100 Hz sampling rate; zero-phase 10–45 Hz band-pass applied to the entire continuous recording before segmentation; one first-300-sample window per contiguous label run; 40 trials per participant; feature domains are time, frequency, or combined. No participant-level held-out test, temporal-block holdout, prospective evaluation, or independent front-end selection is demonstrated.

To reproduce locally:

```bash
python -m pip install -r requirements.txt
python scripts/run_nested_cv.py
python scripts/run_fixed_k_sensitivity.py --k 5
```

The new CSVs should not be compared as though their repeated folds constituted independent participants, or promoted as a deployment accuracy guarantee. Dataset license and original report/figure provenance have not been independently verified.
