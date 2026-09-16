# EMG signal classification

A small, inspectable digital-signal-processing and machine-learning project for **three-class surface-EMG gesture recognition** (rest and two gestures). Three bundled subject recordings are processed with digital filters, segmented into fixed windows, converted to time- and frequency-domain features, and classified with conventional models. The project is an **offline, within-subject course experiment**, not a validated real-time gesture interface or a test of generalization to new people.

## What to inspect first

| Component | Code or artifact | Purpose |
| --- | --- | --- |
| Signal loading | [`emg/io.py`](emg/io.py) | Read the three MATLAB recordings; assume a 100 Hz sampling rate. |
| Filtering and segmentation | [`emg/preprocessing.py`](emg/preprocessing.py), [`emg/segmentation.py`](emg/segmentation.py) | Zero-phase filtering and one fixed-length window per contiguous label run. |
| Features and classifiers | [`emg/features.py`](emg/features.py), [`emg/classification.py`](emg/classification.py) | Channel-wise MAV/RMS/WL/ZC/SSC and spectral features; fold-local standardization and KNN/SVM/RF/LDA. |
| Exploratory experiments | [`scripts/run_classification.py`](scripts/run_classification.py), [`scripts/run_experiments.py`](scripts/run_experiments.py) | Per-subject K search and filter/window/classifier comparisons. |
| Nested evaluation | [`scripts/run_nested_cv.py`](scripts/run_nested_cv.py) | Select K on inner folds; evaluate selected model only on held-out outer folds. |
| Fixed-setting sensitivity | [`scripts/run_fixed_k_sensitivity.py`](scripts/run_fixed_k_sensitivity.py) | Evaluate a user-specified K consistently without choosing it on those folds. |
| Historical artifacts | [`results/`](results/), [`docs/`](docs/), [`legacy/`](legacy/) | Previously generated outputs, project reports, and original course scripts. |

## Results and how to interpret them

The committed [`results/classification_results.csv`](results/classification_results.csv) contains the following **exploratory** leave-one-trial-out KNN accuracies. Each feature domain and subject was evaluated over K=1–15, and the **largest accuracy on those same folds** was selected. These are observed, within-subject results—not an independent held-out evaluation or an estimate of performance on unseen subjects.

| Subject | Time features | Frequency features | Combined features | Trials per domain |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.975 | 0.950 | 0.950 | 40 |
| 2 | 1.000 | 1.000 | 1.000 | 40 |
| 3 | 1.000 | 1.000 | 1.000 | 40 |

The [filter](results/experiment_filters.csv), [window](results/experiment_windows.csv), and [classifier](results/experiment_classifiers.csv) CSVs are also **exploratory**. The original scripts compare configurations on the same recordings, so those comparisons are not independent validation. The figures in [`results/figures/`](results/figures/) visualize saved experiments, not unseen-subject generalization.

A separate **nested-CV protocol** uses five stratified outer folds and three stratified inner folds per subject and feature domain. Each outer training partition alone selects K from 1–15 with an inner `StandardScaler → KNN` pipeline; its held-out outer partition is scored once. This addresses the specific reuse of test folds for K selection in the original headline, while the default filter, window, feature domains, and K range were still selected in the broader research workflow. It remains **within-subject** and does not resolve adjacent-trial dependence, independent front-end selection, or deployment robustness. Do not mix nested-CV output with the exploratory results above or describe either as generalization across participants.

A third **fixed-K sensitivity command** evaluates K=5 by default without fitting K on those folds. Choosing K after seeing the original dataset does not make that evaluation independent. A trustworthy unseen-person claim requires appropriate participant-level held-out data and protocol.

## Run from a fresh checkout

```bash
git clone https://github.com/omarsaqr12/EMG-Signal-Classification.git
cd EMG-Signal-Classification
python -m venv .venv
source .venv/bin/activate             # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt

# Nested K-selection protocol; writes a separate results file.
python scripts/run_nested_cv.py

# Fixed-K sensitivity; specify the K rather than selecting it on these folds.
python scripts/run_fixed_k_sensitivity.py --k 5

# Original exploratory outputs: rewrites results/*.csv and results/figures/*.
python scripts/run_classification.py
python scripts/run_experiments.py

# Lightweight synthetic-data regression checks.
python -m unittest discover -s tests -v
```

The scripts expect the three tracked `.mat` files under `data/`. They operate on local recordings without external service credentials. The original pipeline filters the *complete recording* before splitting into trials; `filtfilt` is noncausal and uses samples on both sides of a time point. Consequently, these scripts are **offline analyses**, not causal streaming-inference demonstrations. The historical numbers above come from committed CSVs; they are not reproduced merely by reading them.

## Method and scope

The sampling rate used throughout the code is 100 Hz (50 Hz Nyquist). The default 10–45 Hz band-pass removes lower-frequency drift; a 50 Hz notch is deliberately a no-op at this rate. The segmentation code takes the first 300 samples (3 seconds) of each contiguous stimulus-label run and drops shorter runs. Five time-domain and four frequency-domain features are calculated for each of 10 channels. A `StandardScaler` is fitted inside each original leave-one-out fold, or inside each nested search pipeline's inner training fold.

**Limitations:** only three participants; within-participant rather than leave-one-subject-out evaluation; front-end/feature choices on reused recordings; possible temporal dependence between adjacent rest/gesture runs; no measured streaming latency, cross-device robustness, or prospective deployment. The dataset's origin, redistribution rights and individual contributions should be confirmed against the original course documentation before reuse outside this repository. The [project report](docs/DSP_Project_Report.pdf) and [original code](legacy/) are retained for provenance; the `emg/` package is a later modular implementation.

## Academic context and attribution

Developed for CSCE 3611 — Digital Signal Processing at the American University in Cairo, supervised by Dr. Seif Eldawlatly. Project authors listed in the original repository: **Omar Saqr, Adham Ali, and Saif Abdelfattah**. The repository does not establish a per-author breakdown of implementation work; please do not infer one from the repository owner. See [`LICENSE`](LICENSE) for the code license; that file should not be assumed to establish third-party dataset redistribution rights.
