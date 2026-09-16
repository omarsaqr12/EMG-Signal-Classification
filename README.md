# EMG signal classification

A small, inspectable digital-signal-processing and machine-learning project for **three-class surface-EMG gesture recognition** (rest and two gestures). Three bundled subject recordings are processed with digital filters, segmented into fixed windows, converted to time- and frequency-domain features, and classified with conventional models. The project is an **offline, within-subject course experiment**, not a validated real-time gesture interface or a test of generalization to new people.

## What to inspect first

| Component | Code or artifact | Purpose |
| --- | --- | --- |
| Signal loading | [`emg/io.py`](emg/io.py) | Read the three MATLAB recordings; assume a 100 Hz sampling rate. |
| Filtering and segmentation | [`emg/preprocessing.py`](emg/preprocessing.py), [`emg/segmentation.py`](emg/segmentation.py) | Zero-phase filtering and one fixed-length window per contiguous label run. |
| Features and classifiers | [`emg/features.py`](emg/features.py), [`emg/classification.py`](emg/classification.py) | Channel-wise MAV/RMS/WL/ZC/SSC and spectral features; fold-local standardization and KNN/SVM/RF/LDA. |
| Exploratory experiments | [`scripts/run_classification.py`](scripts/run_classification.py), [`scripts/run_experiments.py`](scripts/run_experiments.py) | Per-subject K search and filter/window/classifier comparisons. |
| Fixed-setting sensitivity | [`scripts/run_fixed_k_sensitivity.py`](scripts/run_fixed_k_sensitivity.py) | Evaluate a user-specified K consistently without choosing it on those folds. |
| Historical artifacts | [`results/`](results/), [`docs/`](docs/), [`legacy/`](legacy/) | Previously generated outputs, project reports, and original course scripts. |

## Results and how to interpret them

The committed [`results/classification_results.csv`](results/classification_results.csv) contains the following **exploratory** leave-one-trial-out KNN accuracies. Each feature domain and subject was evaluated over K=1–15, and the **largest accuracy on those same folds** was selected. These are observed, within-subject results—not an independent held-out evaluation or an estimate of performance on unseen subjects.

| Subject | Time features | Frequency features | Combined features | Trials per domain |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.975 | 0.950 | 0.950 | 40 |
| 2 | 1.000 | 1.000 | 1.000 | 40 |
| 3 | 1.000 | 1.000 | 1.000 | 40 |

The [filter](results/experiment_filters.csv), [window](results/experiment_windows.csv), and [classifier](results/experiment_classifiers.csv) CSVs are also **exploratory**. In particular, do not interpret the highest-performing front end or classifier as independently validated: the original scripts perform configuration comparison on the same recordings. The figures in [`results/figures/`](results/figures/) visualize those saved experiments; they are not evidence of unseen-subject generalization.

There is now a separate **fixed-K sensitivity command** below. Its results should be reported separately, with the chosen K and its rationale; choosing K after seeing the original dataset still does not make this an independent external test. A trustworthy unseen-subject claim would require appropriate participant-level held-out data and protocol.

## Run from a fresh checkout

```bash
git clone https://github.com/omarsaqr12/EMG-Signal-Classification.git
cd EMG-Signal-Classification
python -m venv .venv
source .venv/bin/activate             # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt

# Original exploratory outputs: rewrites results/*.csv and results/figures/*.
python scripts/run_classification.py
python scripts/run_experiments.py

# Separate sensitivity analysis; fixes K across subjects and feature domains.
python scripts/run_fixed_k_sensitivity.py --k 5

# Lightweight core checks (do not need bundled recordings).
python -m unittest discover -s tests -v
```

The scripts expect the three tracked `.mat` files under `data/`. They operate on local recordings without external service credentials. The original pipeline filters the *complete recording* before splitting into trials; `filtfilt` is noncausal and uses samples on both sides of a time point. Consequently, these scripts are **offline analyses**, not causal streaming-inference demonstrations. No fresh rerun or independent verification of the committed metrics is implied by this README.

## Method and scope

The sampling rate used throughout the code is 100 Hz (50 Hz Nyquist). The default 10–45 Hz band-pass removes lower-frequency drift; a 50 Hz notch is deliberately a no-op at this rate. The segmentation code takes the first 300 samples (3 seconds) of each contiguous stimulus-label run and drops shorter runs. Five time-domain and four frequency-domain features are calculated for each of 10 channels. A `StandardScaler` is fitted inside each leave-one-out fold, followed by the selected classifier.

**Limitations:** only three participants; within-participant rather than leave-one-subject-out evaluation; hyperparameter and front-end selection on reused recordings; possible temporal dependence between adjacent rest/gesture runs; no measured streaming latency, cross-device robustness, or prospective deployment. The dataset's origin, redistribution rights and individual contributions should be confirmed against the original course documentation before reuse outside this repository. The [project report](docs/DSP_Project_Report.pdf) and [original code](legacy/) are retained for provenance; the `emg/` package is a later modular implementation.

## Academic context and attribution

Developed for CSCE 3611 — Digital Signal Processing at the American University in Cairo, supervised by Dr. Seif Eldawlatly. Project authors listed in the original repository: **Omar Saqr, Adham Ali, and Saif Abdelfattah**. The repository does not establish a per-author breakdown of implementation work; please do not infer one from the repository owner. See [`LICENSE`](LICENSE) for the code license; that file should not be assumed to establish third-party dataset redistribution rights.
