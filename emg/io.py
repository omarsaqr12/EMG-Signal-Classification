"""Loading of the bundled MATLAB ``.mat`` recordings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.io

# Repository layout: <root>/emg/io.py  ->  <root>/data
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@dataclass
class Subject:
    """A single subject's continuous multi-channel recording.

    Attributes
    ----------
    subject_id:
        Integer identifier stored in the ``.mat`` file.
    emg:
        ``(n_samples, n_channels)`` array of raw sEMG amplitudes.
    stimulus:
        ``(n_samples,)`` array of movement labels. ``0`` marks rest;
        positive integers mark the active gestures.
    fs:
        Sampling rate in Hz.
    """

    subject_id: int
    emg: np.ndarray
    stimulus: np.ndarray
    fs: int = 100

    @property
    def n_channels(self) -> int:
        return self.emg.shape[1]


def load_subject(subject: int | str | Path, fs: int = 100) -> Subject:
    """Load one subject.

    ``subject`` may be an integer (``1`` -> ``data/subject_1.mat``) or a path
    to a ``.mat`` file.
    """
    if isinstance(subject, int):
        path = DATA_DIR / f"subject_{subject}.mat"
    else:
        path = Path(subject)
    if not path.exists():
        raise FileNotFoundError(f"Recording not found: {path}")

    mat = scipy.io.loadmat(path)
    emg = np.asarray(mat["emg"], dtype=float)
    stimulus = np.asarray(mat["stimulus"], dtype=int).flatten()
    subject_id = int(np.asarray(mat["subject"]).flatten()[0])
    return Subject(subject_id=subject_id, emg=emg, stimulus=stimulus, fs=fs)
