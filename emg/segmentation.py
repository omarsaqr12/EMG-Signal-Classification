"""Turn a continuous recording into fixed-length, labelled windows.

The stimulus channel is piece-wise constant: each contiguous run of one label
is one repetition of a gesture (or rest). We cut a fixed-length window from the
start of every repetition so that all trials share the same dimensionality,
which the feature step and the distance-based classifiers require.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Trial:
    data: np.ndarray  # (window_samples, n_channels)
    label: int


def _stimulus_runs(stimulus):
    """Yield ``(start, end, label)`` for each contiguous run in ``stimulus``."""
    changes = np.where(np.diff(stimulus) != 0)[0] + 1
    starts = np.concatenate(([0], changes))
    ends = np.concatenate((changes, [len(stimulus)]))
    for start, end in zip(starts, ends):
        yield start, end, int(stimulus[start])


def segment_trials(emg, stimulus, window_samples=300, include_rest=True,
                   rest_label=0):
    """Segment ``emg`` into equal-length trials.

    Parameters
    ----------
    window_samples:
        Number of samples taken from the start of each repetition. Runs shorter
        than this are dropped so every returned trial is exactly this long.
    include_rest:
        If ``False``, rest repetitions (``rest_label``) are skipped, leaving a
        gesture-only problem.

    Returns
    -------
    list[Trial]
    """
    trials = []
    for start, end, label in _stimulus_runs(stimulus):
        if not include_rest and label == rest_label:
            continue
        if end - start < window_samples:
            continue
        window = emg[start:start + window_samples, :]
        trials.append(Trial(data=window, label=label))
    return trials
