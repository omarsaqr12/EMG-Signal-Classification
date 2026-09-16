# Engineering review handoff · 2026-09-16

**Baseline:** `main` at `6c312bd281fdec10db746d3e16d89131ab6a198e`. **Review branch:** `audit/validated-evaluation-and-recruiter-guide-20260916`. [Draft PR #1](https://github.com/omarsaqr12/EMG-Signal-Classification/pull/1). No default-branch edits, removed experiments, changed licenses, or merged PRs.

## Calibration, architecture and file coverage

This is a three-person, 100 Hz sEMG offline DSP/classification course project, not a streaming device or new-person recognition benchmark. The critical validation boundary is *trial versus participant*, with hyperparameter and signal-conditioning choices inside evaluation folds. Architecture: bundled MATLAB recordings → per-recording zero-phase filter → contiguous stimulus-run segmentation (first 300 samples) → per-channel time/FFT features → fold-local feature scaling and classifier → exploratory CSVs/figures.

The baseline recursive Git tree contains **32 tracked files**. All **19 readable text/code/config/CSV files** were fetched and reviewed: root README, `.gitignore`, LICENSE; six `emg/` Python modules; three legacy text/script files; requirements; two original entry scripts; four historical result CSVs. The three `.mat` binaries were exercised by the fresh CI runs; their underlying participant provenance and full binary contents were not manually audited. The three PDF documents and seven PNG figures were inventoried but their visual pages/pixels were **not** inspected via the available repository connector. Therefore the project report, dataset distribution rights, and figure-to-CSV correspondence remain **unverified**.

## Findings and claim-to-evidence mapping

| Claim / issue | Classification and support |
| --- | --- |
| Original README says 95–100% LOOCV accuracy | **Historically documented / selection-biased:** original `emg/classification.py` searches K=1–15 on LOOCV folds, then selects/reports maximum on the *same* folds; `results/classification_results.csv` stores those peaks. Not a clean independent estimate. |
| Feature scaling is applied on held-out folds | **Implemented correctly:** original classifier fits `StandardScaler` to each training fold and transforms held-out examples. |
| Three-subject results generalize to unseen people | **Not supported:** original and new analyses all train/test *within* one participant; no participant-level holdout. |
| Streaming recognition or real-time latency | **Not supported:** the entire recording is passed through noncausal `filtfilt` before segmentation. No streaming latency was measured. |
| New K-selection and fixed-K estimates | **Tested:** [2026-09-16 CI run](https://github.com/omarsaqr12/EMG-Signal-Classification/actions/runs/35085162672) runs both new methods against all three tracked subjects and compares freshly generated nine-row CSVs with the committed rows. |
| Academic author roles and recording redistribution rights | **Partial:** original MIT source license lists Omar Saqr, Adham Ali and Saif Abdelfattah, without a per-person work breakdown or independently established third-party recording rights. |

## Implemented changes and exact verification

- New [`scripts/run_nested_cv.py`](../scripts/run_nested_cv.py): each subject/domain uses 5 stratified held-out outer folds; K=1–15 and scaling are fitted exclusively using three inner folds of outer-training data. It removes the specific original K-selection/evaluation reuse, **not** previous front-end choice or within-subject temporal dependence.
- New [`scripts/run_fixed_k_sensitivity.py`](../scripts/run_fixed_k_sensitivity.py): explicit K=5 LOOCV sensitivity; not independently preselected.
- New [`results/nested_cv_results.csv`](../results/nested_cv_results.csv), [`results/fixed_k_sensitivity.csv`](../results/fixed_k_sensitivity.csv) and [`results/PROVENANCE.md`](../results/PROVENANCE.md): fresh observed outcomes, parameter choices, CI environment and precise limitations; original CSVs and figures remain unchanged.
- Nine synthetic unit tests across segmentation, features, filtering, KNN, nested CV and invalid input; [readme](../README.md) now separates exploratory, nested and fixed-K protocols, links code and acknowledges the three original authors. Pull-request CI reproduces both new CSVs from actual `.mat` data and checks all nine rows for exact agreement.

**PASS:** Ubuntu 24.04/Python 3.11.16, pip install of existing requirements, Python compilation, nine unit tests, all three bundled recordings processed in both new protocols, exact agreement for 18 new subject/domain CSV rows. **NOT RUN / BLOCKED:** original full experiment/figure regeneration and pixel verification; report/PDF examination; participant-level independent test; true causal per-trial preprocessing, streaming deployment or hardware metrics; third-party data permission review. Fresh nested accuracies range 0.925–1.000 *within subjects*; these are not a cross-subject guarantee.

## Scope choices, recruiter description, and next work

Accepted nested selection, deterministic sensitivity and CI-backed provenance. Rejected new classifiers, real-time deployment claims or selectively choosing the best subject/domain after seeing test results. Suggested GitHub repository description (metadata **not changed**): **“Offline surface-EMG gesture classification with DSP features, KNN baselines, nested cross-validation, and reproducible within-subject results.”**

For a materially stronger research claim, obtain permission and more participants/sessions, select the entire preprocessing/feature/model pipeline *inside* the training partition, evaluate on held-out participants or sessions, use causal per-trial preprocessing if deploying, audit PDF reports/figures against CSVs, and document the coauthors' specific contributions. Leave this PR draft and unmerged for reviewer approval.
