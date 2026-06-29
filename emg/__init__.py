"""EMG signal classification toolkit.

A small, dependency-light library for the classic surface-EMG (sEMG)
gesture-recognition pipeline:

    load  ->  filter  ->  segment into windows  ->  extract features  ->  classify

The public API is intentionally flat so the scripts in ``scripts/`` and the
README examples stay readable.
"""

from .io import load_subject, Subject
from .preprocessing import bandpass_filter, highpass_filter, notch_filter, apply_filter
from .segmentation import segment_trials, Trial
from .features import (
    time_domain_features,
    frequency_domain_features,
    extract_feature_matrix,
    TIME_FEATURE_NAMES,
    FREQ_FEATURE_NAMES,
)
from .classification import (
    build_classifier,
    leave_one_out_accuracy,
    CLASSIFIERS,
)

__all__ = [
    "load_subject",
    "Subject",
    "bandpass_filter",
    "highpass_filter",
    "notch_filter",
    "apply_filter",
    "segment_trials",
    "Trial",
    "time_domain_features",
    "frequency_domain_features",
    "extract_feature_matrix",
    "TIME_FEATURE_NAMES",
    "FREQ_FEATURE_NAMES",
    "build_classifier",
    "leave_one_out_accuracy",
    "CLASSIFIERS",
]

FS = 100  # Sampling rate of the bundled dataset, in Hz.
