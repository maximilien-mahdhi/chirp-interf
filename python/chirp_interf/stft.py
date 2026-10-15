r"""Discrete STFT on the sub-sampled time lattice ``{m R + p}``.

.. math::

    V_g f(t, \omega) = \sum_{n=-M}^{M} f[t+n]\, g[n]\, e^{-2i\pi\omega n},
    \qquad t = mR + p,\quad \omega = k/N .

The signal is zero outside ``[0, L)``. Computations are delegated to the C++
core (one FFT per frame); this module only validates inputs and builds the
time and frequency axes.
"""

from __future__ import annotations

import numpy as np

from . import _core
from .windows import half_length


def _prep(f, g):
    """Cast ``f`` and ``g`` to the contiguous ``complex128`` / ``float64`` arrays expected by the core.

    Parameters
    ----------
    f : array_like
        Signal.
    g : array_like
        Window of odd length.

    Returns
    -------
    tuple of numpy.ndarray
        ``(f, g)`` ready to be passed to :mod:`chirp_interf._core`.

    Raises
    ------
    ValueError
        If ``g`` does not have odd length.
    """
    f = np.ascontiguousarray(f, dtype=np.complex128)
    g = np.ascontiguousarray(g, dtype=np.float64)
    half_length(g)
    return f, g


def n_frames(L: int, R: int = 1, p: int = 0) -> int:
    """Number of lattice points ``m R + p`` (``m >= 0``) inside ``[0, L)``.

    Parameters
    ----------
    L : int
        Signal length.
    R : int, default 1
        Hop (downsampling factor).
    p : int, default 0
        Shift, ``0 <= p < R``.

    Returns
    -------
    int
        ``floor((L - 1 - p) / R) + 1``, or ``0`` if ``L <= p``.
    """
    return 0 if L <= p else (L - 1 - p) // R + 1


def stft(f, g, N: int, R: int = 1, p: int = 0) -> np.ndarray:
    """STFT on the frames of the lattice that lie inside the signal.

    Parameters
    ----------
    f : array_like
        Signal of length ``L`` (cast to complex).
    g : array_like
        Window of length ``2M + 1``.
    N : int
        Number of frequency bins; must satisfy ``N >= 2M + 1``.
    R : int, default 1
        Hop between frames.
    p : int, default 0
        Shift of the lattice, ``0 <= p < R``.

    Returns
    -------
    numpy.ndarray
        Complex array ``V`` of shape ``(N, n_frames(L, R, p))`` with
        ``V[k, m] = V_g f(m R + p, k / N)``. Frequencies are in FFT order
        (``k = 0, ..., N - 1``); use :func:`spectrogram` for a centred axis.

    Raises
    ------
    ValueError
        If ``N < 2M + 1``, ``R < 1`` or ``p`` is outside ``[0, R)``.
    """
    f, g = _prep(f, g)
    nt = n_frames(len(f), R, p)
    return _core.stft_frames(f, g, N, R, p, 0, nt - 1).T


def frequencies(N: int, fs: float = 1.0) -> np.ndarray:
    """Centred frequency axis matching ``numpy.fft.fftshift`` of an ``N``-point grid.

    Parameters
    ----------
    N : int
        Number of bins.
    fs : float, default 1.0
        Sampling rate (Hz). With ``fs = 1`` the axis is in cycles per sample.

    Returns
    -------
    numpy.ndarray
        Frequencies in ``[-fs/2, fs/2)``, increasing.
    """
    return np.fft.fftshift(np.fft.fftfreq(N, d=1.0 / fs))


def frame_times(L: int, R: int = 1, p: int = 0, fs: float = 1.0) -> np.ndarray:
    """Centres of the frames inside the signal, in seconds.

    Parameters
    ----------
    L : int
        Signal length.
    R : int, default 1
        Hop.
    p : int, default 0
        Shift.
    fs : float, default 1.0
        Sampling rate (Hz).

    Returns
    -------
    numpy.ndarray
        ``(m R + p) / fs`` for ``m = 0, ..., n_frames(L, R, p) - 1``.
    """
    return (np.arange(n_frames(L, R, p)) * R + p) / fs


def spectrogram(f, g, N: int, R: int = 1, p: int = 0, fs: float = 1.0):
    """STFT with labelled, frequency-centred axes.

    Parameters
    ----------
    f, g, N, R, p
        See :func:`stft`.
    fs : float, default 1.0
        Sampling rate (Hz) used to label the axes.

    Returns
    -------
    times : numpy.ndarray
        Frame centres (s), length ``n_frames``.
    freqs : numpy.ndarray
        Centred frequencies (Hz), length ``N``.
    V : numpy.ndarray
        Complex STFT of shape ``(N, n_frames)``, rows ordered as ``freqs``.
    """
    V = np.fft.fftshift(stft(f, g, N, R, p), axes=0)
    return frame_times(len(f), R, p, fs), frequencies(N, fs), V
