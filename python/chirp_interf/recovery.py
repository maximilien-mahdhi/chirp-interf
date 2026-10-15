r"""Recovery of a signal from its hard-thresholded, sub-sampled STFT.

Every technique observes :math:`V_g f(mR+p, k/N)`, sets to zero the
coefficients with :math:`|V_g f| < \tau` (``threshold``), inverts each frame,

.. math:: \tilde u_m[d] = \frac1N \sum_{k=0}^{N-1} \tilde V_g f(mR+p, k/N)\, e^{2i\pi kd/N}, \qquad |d| \le M,

and combines the frames:

* **RT1** — one frame per sample: :math:`f[n] = \tilde u_m[q-p] / g[q-p]` with :math:`n = mR + q`;
* **RT2** — least-squares dual window: :math:`f[n] = \sum_m g\,\tilde u_m / \sum_m g^2`;
* **RT3** — plain overlap-add: :math:`f[n] = \sum_m \tilde u_m / \sum_m g`,

where the sums run over the frames covering :math:`n` and are evaluated at
:math:`d = n - mR - p`. Without thresholding the three are exact. The
implementation is in ``src/recovery.cpp``; this module validates inputs and adds
the noise-level helpers.
"""

from __future__ import annotations

import logging

import numpy as np

from . import _core
from .stft import _prep, stft

logger = logging.getLogger(__name__)


def noise_level(sigma: float, g) -> float:
    r"""Standard deviation of an STFT coefficient of white noise.

    For white noise of standard deviation :math:`\sigma`,
    :math:`\gamma = \sigma \lVert g \rVert_2`.

    Parameters
    ----------
    sigma : float
        Noise standard deviation.
    g : array_like
        Analysis window.

    Returns
    -------
    float
        :math:`\gamma`.
    """
    return float(sigma * np.linalg.norm(g))


def estimate_noise_level(f, g, N: int, R: int = 1, p: int = 0) -> float:
    r"""Robust estimate of :math:`\gamma` from the STFT of a noisy signal.

    Uses the median absolute deviation of the real parts,
    :math:`\hat\gamma = \sqrt2\, \mathrm{median}|\mathrm{Re}\, V_g f| / 0.6745`
    (the real part of a noise coefficient has standard deviation
    :math:`\gamma/\sqrt2`, and 0.6745 is the median of :math:`|\mathcal N(0, 1)|`).
    The median is insensitive to the few large coefficients carrying the
    signal, as long as the signal is sparse in the time-frequency plane.

    Parameters
    ----------
    f : array_like
        Noisy signal.
    g, N, R, p
        STFT parameters (see :func:`chirp_interf.stft.stft`).

    Returns
    -------
    float
        :math:`\hat\gamma`.
    """
    V = stft(f, g, N, R, p)
    gamma = float(np.sqrt(2.0) * np.median(np.abs(V.real)) / 0.6745)
    logger.debug("estimated noise level gamma = %.4g", gamma)
    return gamma


def rt1(f, g, N: int, R: int = 1, p: int = 0, threshold: float = 0.0) -> np.ndarray:
    """Recovery technique 1: one frame per sample, division by a single window value.

    Cheapest technique, but very sensitive to the shift ``p`` once the STFT is
    thresholded, because ``g[q - p]`` can be small.

    Parameters
    ----------
    f : array_like
        Signal (typically noisy), length ``L``.
    g : array_like
        Window of length ``2M + 1``; ``g[q - p]`` must be non-zero for all ``0 <= q < R``.
    N : int
        Number of frequency bins, ``N >= 2M + 1``.
    R : int, default 1
        Hop; requires ``max(p, R - 1 - p) <= M``.
    p : int, default 0
        Shift, ``0 <= p < R``.
    threshold : float, default 0.0
        Coefficients with ``|V| < threshold`` are set to zero (typically ``3 * gamma``).

    Returns
    -------
    numpy.ndarray
        Recovered complex signal of length ``L``.

    Raises
    ------
    ValueError
        If the parameters violate the constraints above.
    """
    f, g = _prep(f, g)
    return _core.rt1(f, g, N, R, p, threshold)


def rt2(f, g, N: int, R: int = 1, p: int = 0, threshold: float = 0.0) -> np.ndarray:
    """Recovery technique 2: overlap-add with the least-squares dual window ``g / sum g^2``.

    Parameters
    ----------
    f, g, N, p, threshold
        See :func:`rt1`.
    R : int, default 1
        Hop; requires ``R <= 2M + 1`` so that every sample is covered by a frame.

    Returns
    -------
    numpy.ndarray
        Recovered complex signal of length ``L``.
    """
    f, g = _prep(f, g)
    return _core.rt2(f, g, N, R, p, threshold)


def rt3(f, g, N: int, R: int = 1, p: int = 0, threshold: float = 0.0) -> np.ndarray:
    """Recovery technique 3: plain overlap-add normalised by ``sum g``.

    Parameters
    ----------
    f, g, N, p, threshold
        See :func:`rt1`.
    R : int, default 1
        Hop; requires ``R <= 2M + 1``.

    Returns
    -------
    numpy.ndarray
        Recovered complex signal of length ``L``.
    """
    f, g = _prep(f, g)
    return _core.rt3(f, g, N, R, p, threshold)


#: Name -> function table used by :func:`recover` and the example scripts.
METHODS = {"RT1": rt1, "RT2": rt2, "RT3": rt3}


def recover(method: str, f, g, N: int, R: int = 1, p: int = 0, threshold: float = 0.0) -> np.ndarray:
    """Dispatch to :func:`rt1`, :func:`rt2` or :func:`rt3` by name.

    Parameters
    ----------
    method : {"RT1", "RT2", "RT3"}
        Technique name (case-insensitive).
    f, g, N, R, p, threshold
        See :func:`rt1`.

    Returns
    -------
    numpy.ndarray
        Recovered complex signal.

    Raises
    ------
    ValueError
        If ``method`` is unknown.
    """
    try:
        fn = METHODS[method.upper()]
    except KeyError:
        raise ValueError(f"unknown method {method!r}, expected one of {list(METHODS)}") from None
    return fn(f, g, N, R, p, threshold)


def snr_db(reference, estimate) -> float:
    r"""Signal-to-noise ratio :math:`20 \log_{10}(\lVert f\rVert_2 / \lVert f - \hat f\rVert_2)`, in dB.

    Parameters
    ----------
    reference : array_like
        Clean signal :math:`f`.
    estimate : array_like
        Estimate :math:`\hat f`, same length.

    Returns
    -------
    float
        SNR in decibels (``inf`` if the estimate is exact).
    """
    reference = np.asarray(reference)
    return float(20 * np.log10(np.linalg.norm(reference) / np.linalg.norm(reference - estimate)))
