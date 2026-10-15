"""Greedy extraction of the ridges of a spectrogram.

For each of the ``K`` components:

1. start from the largest remaining coefficient ``(k*, m*)``;
2. track it forward and backward in time: at each frame, take the maximum of
   ``|V|`` within ``+/- max_jump`` bins of the previous ridge point;
3. erase a band of ``+/- band`` bins around the ridge before extracting the
   next component.

Near a crossing of two components, the band erased for the first ridge also
erases the second component, which is then tracked through interference
only. The closed-form analysis of :mod:`chirp_interf.interference` predicts
the time interval where this happens.
"""

from __future__ import annotations

import logging
import math

import numpy as np

logger = logging.getLogger(__name__)


def max_jump_for_rate(rate: float, N: int, fs: float, R: int = 1, margin: int = 1) -> int:
    """Number of frequency bins swept between two frames by a chirp of a given rate.

    Between frames separated by ``R / fs`` seconds, a chirp of rate ``rate``
    moves by ``rate * R / fs`` Hz, i.e. ``rate * R * N / fs**2`` bins of width ``fs / N``.

    Parameters
    ----------
    rate : float
        Largest chirp rate expected (Hz/s); only its absolute value matters.
    N : int
        Number of frequency bins.
    fs : float
        Sampling rate (Hz).
    R : int, default 1
        Hop between frames.
    margin : int, default 1
        Extra bins added for robustness to noise.

    Returns
    -------
    int
        Maximum allowed jump, in bins.
    """
    return math.ceil(abs(rate) * R * N / fs**2) + margin


def band_for_window(sigma: float, rate: float, N: int, fs: float, n_std: float = 3.0) -> int:
    """Half-width of a Gaussian-window ridge, in frequency bins.

    The STFT modulus of a linear chirp is Gaussian in frequency, with standard
    deviation ``sqrt(1 + 4 pi^2 a^2 sigma^4) / (2 pi sigma)`` Hz.

    Parameters
    ----------
    sigma : float
        Window standard deviation (s).
    rate : float
        Chirp rate ``a`` (Hz/s).
    N : int
        Number of frequency bins.
    fs : float
        Sampling rate (Hz).
    n_std : float, default 3.0
        Number of standard deviations erased on each side of the ridge.

    Returns
    -------
    int
        Band half-width, in bins.
    """
    std_hz = math.sqrt(1 + (2 * math.pi * rate * sigma**2) ** 2) / (2 * math.pi * sigma)
    return math.ceil(n_std * std_hz * N / fs)


def _track(S: np.ndarray, k0: int, m0: int, max_jump: int) -> np.ndarray:
    """Follow a ridge of ``S`` from ``(k0, m0)``, forward then backward in time.

    Parameters
    ----------
    S : numpy.ndarray
        Non-negative magnitude, shape ``(N, n_frames)``.
    k0, m0 : int
        Starting frequency index and frame.
    max_jump : int
        Maximum change of frequency index between consecutive frames.

    Returns
    -------
    numpy.ndarray
        Frequency index of the ridge in every frame, length ``n_frames``.
    """
    N, Nt = S.shape
    ridge = np.empty(Nt, dtype=int)
    ridge[m0] = k0
    for step in (1, -1):                     # forward, then backward
        m = m0 + step
        while 0 <= m < Nt:
            k_prev = ridge[m - step]
            lo, hi = max(k_prev - max_jump, 0), min(k_prev + max_jump, N - 1)
            ridge[m] = lo + int(np.argmax(S[lo:hi + 1, m]))
            m += step
    return ridge


def extract_ridges(V, K: int, max_jump: int, band: int | None = None) -> np.ndarray:
    """Extract ``K`` ridges of ``|V|`` greedily.

    Parameters
    ----------
    V : array_like
        STFT of shape ``(N, n_frames)``, preferably frequency-centred (see
        :func:`chirp_interf.stft.spectrogram`) so that ridges do not wrap around ``fs / 2``.
    K : int
        Number of ridges to extract.
    max_jump : int
        Maximum change of frequency index between consecutive frames (see :func:`max_jump_for_rate`).
    band : int, optional
        Half-width (bins) erased around each ridge before the next one (see
        :func:`band_for_window`). Defaults to ``max_jump``.

    Returns
    -------
    numpy.ndarray
        Integer array of shape ``(K, n_frames)``: row ``l`` holds the frequency
        index of ridge ``l`` in every frame, in extraction order (strongest first).
    """
    S = np.abs(np.asarray(V)).astype(float)
    N, Nt = S.shape
    band = max_jump if band is None else band
    ridges = np.empty((K, Nt), dtype=int)
    k_idx = np.arange(N)[:, None]
    for l in range(K):
        k0, m0 = np.unravel_index(int(np.argmax(S)), S.shape)
        ridges[l] = _track(S, int(k0), int(m0), max_jump)
        S[np.abs(k_idx - ridges[l][None, :]) <= band] = 0.0     # erase the band around the ridge
        logger.debug("ridge %d started at (k=%d, m=%d)", l + 1, k0, m0)
    return ridges
