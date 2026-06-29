"""Time-domain and frequency-domain feature extraction.

Each trial is a ``(window_samples, n_channels)`` window. Features are computed
per channel and concatenated, so a ``C``-channel recording yields
``C * len(TIME_FEATURE_NAMES)`` time-domain features (and likewise for the
frequency domain).
"""

from __future__ import annotations

import numpy as np

TIME_FEATURE_NAMES = ["MAV", "RMS", "WL", "ZC", "SSC"]
FREQ_FEATURE_NAMES = ["MNF", "MDF", "TotalPower", "PeakFreq"]


# --------------------------------------------------------------------------- #
# Time domain
# --------------------------------------------------------------------------- #
def _zero_crossings(x, threshold=0.0):
    """Count sign changes whose amplitude step exceeds ``threshold``."""
    x0, x1 = x[:-1], x[1:]
    sign_change = (x0 * x1) < 0
    big_enough = np.abs(x0 - x1) >= threshold
    return int(np.sum(sign_change & big_enough))


def _slope_sign_changes(x, threshold=0.0):
    """Count changes in the sign of the first difference (turning points)."""
    d1 = x[1:-1] - x[:-2]
    d2 = x[1:-1] - x[2:]
    turning = (d1 * d2) > 0
    big_enough = (np.abs(d1) >= threshold) | (np.abs(d2) >= threshold)
    return int(np.sum(turning & big_enough))


def time_domain_features(window, threshold=0.0):
    """Hudgins-style time-domain features for every channel of ``window``.

    Returns a 1-D vector ordered channel-major:
    ``[ch0_MAV, ch0_RMS, ch0_WL, ch0_ZC, ch0_SSC, ch1_MAV, ...]``.
    """
    window = np.asarray(window, dtype=float)
    feats = []
    for ch in range(window.shape[1]):
        x = window[:, ch]
        mav = np.mean(np.abs(x))
        rms = np.sqrt(np.mean(x ** 2))
        wl = np.sum(np.abs(np.diff(x)))           # waveform length
        zc = _zero_crossings(x, threshold)         # zero crossings
        ssc = _slope_sign_changes(x, threshold)    # slope sign changes
        feats.extend([mav, rms, wl, zc, ssc])
    return np.asarray(feats, dtype=float)


# --------------------------------------------------------------------------- #
# Frequency domain
# --------------------------------------------------------------------------- #
def _channel_freq_features(x, fs):
    n = len(x)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    psd = (np.abs(np.fft.rfft(x)) ** 2) / n        # power spectral density
    total = float(np.sum(psd))
    if total == 0.0:
        return [0.0, 0.0, 0.0, 0.0]
    mnf = float(np.sum(freqs * psd) / total)       # mean frequency
    cumulative = np.cumsum(psd)
    mdf = float(freqs[np.searchsorted(cumulative, total / 2.0)])  # median freq
    peak = float(freqs[np.argmax(psd)])            # dominant frequency
    return [mnf, mdf, total, peak]


def frequency_domain_features(window, fs=100):
    """Spectral features (mean/median freq, total spectral power, peak freq)."""
    window = np.asarray(window, dtype=float)
    feats = []
    for ch in range(window.shape[1]):
        feats.extend(_channel_freq_features(window[:, ch], fs))
    return np.asarray(feats, dtype=float)


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #
def extract_feature_matrix(trials, fs=100, domain="combined"):
    """Build ``(X, y)`` from a list of :class:`~emg.segmentation.Trial`.

    ``domain`` is ``"time"``, ``"frequency"`` or ``"combined"``.
    """
    rows = []
    labels = []
    for trial in trials:
        if domain == "time":
            row = time_domain_features(trial.data)
        elif domain == "frequency":
            row = frequency_domain_features(trial.data, fs)
        elif domain == "combined":
            row = np.concatenate([
                time_domain_features(trial.data),
                frequency_domain_features(trial.data, fs),
            ])
        else:
            raise ValueError(f"Unknown domain: {domain!r}")
        rows.append(row)
        labels.append(trial.label)
    return np.asarray(rows, dtype=float), np.asarray(labels)


def feature_names(n_channels, domain="combined"):
    """Human-readable names matching :func:`extract_feature_matrix` columns."""
    time = [f"ch{c}_{name}" for c in range(n_channels) for name in TIME_FEATURE_NAMES]
    freq = [f"ch{c}_{name}" for c in range(n_channels) for name in FREQ_FEATURE_NAMES]
    if domain == "time":
        return time
    if domain == "frequency":
        return freq
    return time + freq
