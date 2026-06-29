"""Signal-conditioning front end: band-pass, high-pass and notch filtering.

All filters are zero-phase (``filtfilt``) so they introduce no group delay,
which matters when features are later computed on short windows.

A note on sampling rate: the bundled dataset is sampled at ``fs = 100 Hz``,
so its Nyquist frequency is 50 Hz. A conventional 50 Hz power-line notch sits
*at* Nyquist and is therefore not meaningful here -- the band-pass already
rejects everything above ~45 Hz. ``notch_filter`` is provided for completeness
and for higher-rate recordings; it guards against an out-of-range frequency.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch


def _design_butter(cutoff, fs, order, btype):
    nyquist = 0.5 * fs
    normalized = np.asarray(cutoff, dtype=float) / nyquist
    return butter(order, normalized, btype=btype, analog=False)


def highpass_filter(emg, cutoff=10.0, fs=100, order=4):
    """Remove low-frequency drift / motion artefacts below ``cutoff`` Hz."""
    b, a = _design_butter(cutoff, fs, order, "high")
    return filtfilt(b, a, emg, axis=0)


def bandpass_filter(emg, low=10.0, high=45.0, fs=100, order=4):
    """Keep the EMG band ``[low, high]`` Hz, rejecting drift and high-freq noise.

    ``high`` is clipped just below Nyquist so the design is always valid.
    """
    nyquist = 0.5 * fs
    high = min(high, nyquist * 0.99)
    b, a = _design_butter([low, high], fs, order, "band")
    return filtfilt(b, a, emg, axis=0)


def notch_filter(emg, freq=50.0, fs=100, quality=30.0):
    """Suppress a narrow band around ``freq`` Hz (e.g. mains interference).

    Returns the signal unchanged when ``freq`` is not below Nyquist, which is
    the case for a 50 Hz notch at ``fs = 100 Hz``.
    """
    nyquist = 0.5 * fs
    if freq >= nyquist:
        return np.asarray(emg, dtype=float)
    b, a = iirnotch(freq / nyquist, quality)
    return filtfilt(b, a, emg, axis=0)


def apply_filter(emg, kind="bandpass", fs=100, **kwargs):
    """Dispatch helper used by the experiment scripts.

    ``kind`` is one of ``"none"``, ``"highpass"``, ``"bandpass"`` or
    ``"bandpass_notch"`` (band-pass followed by a notch where applicable).
    """
    if kind == "none":
        return np.asarray(emg, dtype=float)
    if kind == "highpass":
        return highpass_filter(emg, fs=fs, **kwargs)
    if kind == "bandpass":
        return bandpass_filter(emg, fs=fs, **kwargs)
    if kind == "bandpass_notch":
        return notch_filter(bandpass_filter(emg, fs=fs, **kwargs), fs=fs)
    raise ValueError(f"Unknown filter kind: {kind!r}")
