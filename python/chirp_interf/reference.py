"""Pure-NumPy reference implementations, used as a test oracle for the C++ core.

They follow the formulas of :mod:`chirp_interf.recovery` literally and rely on
``numpy.fft`` instead of the hand-written FFT of the core, so an error in one
implementation is very unlikely to be reproduced by the other. They are
vectorised but not optimised (``chirp-benchmark`` compares both).
"""

from __future__ import annotations

import numpy as np

from .windows import half_length


def stft_frames(f, g, N: int, R: int, p: int, m_min: int, m_max: int) -> np.ndarray:
    """STFT on the frames ``m_min <= m <= m_max`` of the lattice ``{m R + p}``.

    Parameters
    ----------
    f : array_like
        Signal of length ``L``, zero outside ``[0, L)``.
    g : numpy.ndarray
        Window of length ``2M + 1``.
    N : int
        Number of frequency bins, ``N >= 2M + 1``.
    R, p : int
        Hop and shift of the lattice.
    m_min, m_max : int
        Frame range (inclusive); frames may be centred outside the signal.

    Returns
    -------
    numpy.ndarray
        Complex array of shape ``(m_max - m_min + 1, N)``; row ``i`` is frame ``m_min + i``.
    """
    f = np.asarray(f, dtype=complex)
    M = half_length(g)
    L = len(f)
    t = np.arange(m_min, m_max + 1)[:, None] * R + p          # frame centres, column vector
    idx = t + np.arange(-M, M + 1)[None, :]                    # sample index of every window tap
    inside = (idx >= 0) & (idx < L)
    seg = np.where(inside, f[np.clip(idx, 0, L - 1)], 0) * g[None, :]
    # Put offset n in bin n mod N (no aliasing since N >= 2M+1); then V[k] = sum_n seg[n] e^{-2i pi kn/N}.
    buf = np.zeros((len(t), N), dtype=complex)
    np.add.at(buf, (slice(None), np.arange(-M, M + 1) % N), seg)
    return np.fft.fft(buf, axis=1)


def _frames_inverse(V: np.ndarray, M: int, N: int) -> np.ndarray:
    """Inverse DFT of every frame, restricted to offsets ``|d| <= M``.

    Parameters
    ----------
    V : numpy.ndarray
        Frames, shape ``(n_frames, N)``.
    M : int
        Window half-length.
    N : int
        Number of bins.

    Returns
    -------
    numpy.ndarray
        ``u[:, d + M] = (1/N) sum_k V[:, k] exp(2i pi k d / N)``, shape ``(n_frames, 2M + 1)``.
    """
    u = np.fft.ifft(V, axis=1)
    return u[:, np.arange(-M, M + 1) % N]


def _threshold(V: np.ndarray, threshold: float) -> np.ndarray:
    """Hard thresholding: keep the coefficients with ``|V| >= threshold``, zero the others."""
    return np.where(np.abs(V) >= threshold, V, 0)


def rt1(f, g, N, R=1, p=0, threshold=0.0):
    """Reference RT1 (see :func:`chirp_interf.recovery.rt1` for the parameters)."""
    M = half_length(g)
    L = len(f)
    V = _threshold(stft_frames(f, g, N, R, p, 0, (L - 1) // R), threshold)
    u = _frames_inverse(V, M, N)
    n = np.arange(L)
    d = n % R - p                       # offset of sample n = mR + q from the frame centre mR + p
    return u[n // R, d + M] / g[d + M]


def _overlap_add(f, g, N, R, p, threshold, least_squares):
    """Shared implementation of RT2 (``least_squares=True``) and RT3 (``False``).

    Every frame whose support meets ``[0, L)`` contributes, including frames
    centred outside the signal, so that edge samples are covered like interior ones.
    """
    M = half_length(g)
    L = len(f)
    m_min = -((M + p) // R)            # ceil((-M - p) / R)
    m_max = (L - 1 + M - p) // R       # floor((L - 1 + M - p) / R)
    V = _threshold(stft_frames(f, g, N, R, p, m_min, m_max), threshold)
    u = _frames_inverse(V, M, N)
    num = np.zeros(L, dtype=complex)
    den = np.zeros(L)
    for i, m in enumerate(range(m_min, m_max + 1)):
        n = m * R + p + np.arange(-M, M + 1)      # samples covered by frame m
        keep = (n >= 0) & (n < L)
        if least_squares:
            np.add.at(num, n[keep], (g * u[i])[keep])
            np.add.at(den, n[keep], (g**2)[keep])
        else:
            np.add.at(num, n[keep], u[i][keep])
            np.add.at(den, n[keep], g[keep])
    return num / den


def rt2(f, g, N, R=1, p=0, threshold=0.0):
    """Reference RT2 (see :func:`chirp_interf.recovery.rt2` for the parameters)."""
    return _overlap_add(f, g, N, R, p, threshold, least_squares=True)


def rt3(f, g, N, R=1, p=0, threshold=0.0):
    """Reference RT3 (see :func:`chirp_interf.recovery.rt3` for the parameters)."""
    return _overlap_add(f, g, N, R, p, threshold, least_squares=False)
