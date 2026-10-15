"""Analysis windows.

Every window has odd length ``2M + 1`` and is centred: index ``n + M`` of the
array holds the value at offset ``n``, for ``n = -M, ..., M``. This is the
convention expected by the C++ core.
"""

from __future__ import annotations

import math

import numpy as np


def gaussian(sigma: float, M: int | None = None, *, width: float = 4.0) -> np.ndarray:
    r"""Sampled unit-mass Gaussian window.

    .. math:: g[n] = \frac{1}{\sqrt{2\pi}\,\sigma} e^{-n^2 / (2\sigma^2)}, \qquad n = -M, \dots, M.

    The factor :math:`1/(\sqrt{2\pi}\sigma)` makes the discrete STFT a Riemann
    sum of the continuous STFT with the unit-mass window, which is the
    normalisation assumed by :mod:`chirp_interf.interference`.

    Parameters
    ----------
    sigma : float
        Standard deviation, in samples. Must be positive.
    M : int, optional
        Half-length. Defaults to ``ceil(width * sigma)``.
    width : float, default 4.0
        Truncation in standard deviations when ``M`` is not given; the relative
        truncation error is about ``exp(-width**2 / 2)`` (3e-4 for 4).

    Returns
    -------
    numpy.ndarray
        Real array of length ``2M + 1``.

    Raises
    ------
    ValueError
        If ``sigma <= 0``.
    """
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if M is None:
        M = math.ceil(width * sigma)
    n = np.arange(-M, M + 1)
    return np.exp(-(n**2) / (2.0 * sigma**2)) / (math.sqrt(2.0 * math.pi) * sigma)


def gaussian_seconds(sigma_t: float, fs: float, *, width: float = 4.0) -> np.ndarray:
    """Gaussian window specified in seconds.

    Parameters
    ----------
    sigma_t : float
        Standard deviation, in seconds.
    fs : float
        Sampling rate, in Hz. The window std in samples is ``sigma_t * fs``.
    width : float, default 4.0
        Truncation in standard deviations (see :func:`gaussian`).

    Returns
    -------
    numpy.ndarray
        Real array of odd length.
    """
    return gaussian(sigma_t * fs, width=width)


def rectangular(M: int) -> np.ndarray:
    """Boxcar window of length ``2M + 1`` with unit sum.

    Parameters
    ----------
    M : int
        Half-length (``M >= 0``).

    Returns
    -------
    numpy.ndarray
        Array filled with ``1 / (2M + 1)``.
    """
    return np.full(2 * M + 1, 1.0 / (2 * M + 1))


def half_length(g: np.ndarray) -> int:
    """Return the half-length ``M`` of a window of length ``2M + 1``.

    Parameters
    ----------
    g : numpy.ndarray
        Window.

    Returns
    -------
    int
        ``(len(g) - 1) // 2``.

    Raises
    ------
    ValueError
        If ``g`` is not one-dimensional or has even length.
    """
    if g.ndim != 1 or len(g) % 2 == 0:
        raise ValueError("window must be 1-D with odd length 2M+1")
    return (len(g) - 1) // 2
