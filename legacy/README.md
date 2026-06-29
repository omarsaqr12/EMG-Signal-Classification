# Legacy scripts

These are the **original course-submission scripts**, kept for reference and
provenance:

- `g.py` — single file running time- and frequency-domain KNN over each channel.
- `main.py` — per-channel high-pass + KNN sweep.

They feed mostly raw / minimally-summarised signals into the classifier. The
maintained, modular version of the project — with proper time/frequency feature
extraction, band-pass filtering, multiple classifiers and reproducible ablation
studies — lives in the [`emg/`](../emg) package and [`scripts/`](../scripts).
