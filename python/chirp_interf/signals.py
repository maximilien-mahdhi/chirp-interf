"""Test signals and additive noise.

All signals are returned as ``complex128`` arrays, since the STFT and the
recovery techniques operate on complex signals. Times are in seconds and
frequencies in Hz.
"""

from __future__ import annotations

import numpy as np


def time_axis(fs: float, T: float) -> np.ndarray:
    """Sampling instants ``n / fs`` for ``n = 0, ..., round(fs T) - 1``.

    Parameters
    ----------
    fs : float
        Sampling rate (Hz).
    T : float
        Duration (s).

    Returns
    -------
    numpy.ndarray
        Time axis of length ``round(fs * T)``.
    """
    return np.arange(int(round(fs * T))) / fs


def linear_chirp(t: np.ndarray, a: float, b: float) -> np.ndarray:
    r"""Complex linear chirp :math:`e^{2i\pi(a t^2/2 + b t)}`.

    Its instantaneous frequency is :math:`a t + b`.

    Parameters
    ----------
    t : numpy.ndarray
        Time axis (s).
    a : float
        Chirp rate (Hz/s).
    b : float
        Initial frequency (Hz).

    Returns
    -------
    numpy.ndarray
        Complex signal of unit modulus.
    """
    return np.exp(2j * np.pi * (0.5 * a * t**2 + b * t))


def crossing_chirps(t: np.ndarray, a: float, b1: float, b2: float) -> np.ndarray:
    """Sum of two linear chirps with opposite rates ``+a`` and ``-a``.

    The instantaneous frequencies ``a t + b1`` and ``-a t + b2`` cross at
    ``t = (b2 - b1) / (2a)``.

    Parameters
    ----------
    t : numpy.ndarray
        Time axis (s).
    a : float
        Rate of the first component (Hz/s); the second one has rate ``-a``.
    b1, b2 : float
        Initial frequencies of the two components (Hz).

    Returns
    -------
    numpy.ndarray
        Complex two-component signal.
    """
    return linear_chirp(t, a, b1) + linear_chirp(t, -a, b2)


def sinusoidal_fm(t: np.ndarray, f0: float, depth: float, f_mod: float) -> np.ndarray:
    """Complex tone with sinusoidal frequency modulation.

    The instantaneous frequency is ``f0 + depth * cos(2 pi f_mod t)``.

    Parameters
    ----------
    t : numpy.ndarray
        Time axis (s).
    f0 : float
        Carrier frequency (Hz).
    depth : float
        Frequency excursion (Hz).
    f_mod : float
        Modulation frequency (Hz), must be non-zero.

    Returns
    -------
    numpy.ndarray
        Complex signal of unit modulus.
    """
    return np.exp(2j * np.pi * (f0 * t + depth / (2 * np.pi * f_mod) * np.sin(2 * np.pi * f_mod * t)))


def sine(t: np.ndarray, freq: float) -> np.ndarray:
    """Real sine wave ``sin(2 pi freq t)``, returned as complex.

    Parameters
    ----------
    t : numpy.ndarray
        Time axis (s).
    freq : float
        Frequency (Hz).

    Returns
    -------
    numpy.ndarray
    """
    return np.sin(2 * np.pi * freq * t).astype(complex)


def square(t: np.ndarray, freq: float) -> np.ndarray:
    """Square wave ``sign(sin(2 pi freq t))``, returned as complex.

    Parameters
    ----------
    t : numpy.ndarray
        Time axis (s).
    freq : float
        Frequency (Hz).

    Returns
    -------
    numpy.ndarray
    """
    return np.sign(np.sin(2 * np.pi * freq * t)).astype(complex)


def triangle(t: np.ndarray, freq: float) -> np.ndarray:
    """Triangle wave with values in ``[-1, 1]``, returned as complex.

    Parameters
    ----------
    t : numpy.ndarray
        Time axis (s).
    freq : float
        Frequency (Hz).

    Returns
    -------
    numpy.ndarray
    """
    x = t * freq
    return (2 * np.abs(2 * (x - np.floor(x + 0.5))) - 1).astype(complex)


def add_noise(f: np.ndarray, sigma: float, *, complex_noise: bool = False,
              rng: np.random.Generator | int | None = None) -> np.ndarray:
    """Add zero-mean white Gaussian noise of standard deviation ``sigma``.

    Parameters
    ----------
    f : numpy.ndarray
        Clean signal.
    sigma : float
        Noise standard deviation. ``0`` returns a copy of ``f``.
    complex_noise : bool, default False
        If ``False``, the noise is real (the setting of the experiments). If
        ``True``, it is circular complex with ``E|n|^2 = sigma**2``.
    rng : numpy.random.Generator, int or None
        Generator or seed, for reproducibility.

    Returns
    -------
    numpy.ndarray
        Noisy complex signal.
    """
    rng = np.random.default_rng(rng)
    if complex_noise:
        noise = (rng.standard_normal(len(f)) + 1j * rng.standard_normal(len(f))) * sigma / np.sqrt(2)
    else:
        noise = sigma * rng.standard_normal(len(f))
    return np.asarray(f, dtype=complex) + noise
