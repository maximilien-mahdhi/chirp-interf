r"""Closed-form Gaussian STFT of linear chirps and their interference term.

Continuous-time conventions (seconds, Hz):

.. math::

    V_g f(t, \eta) = \int f(s)\, g(s - t)\, e^{-2i\pi \eta (s - t)}\, ds,
    \qquad g(u) = \frac{e^{-u^2 / (2\sigma^2)}}{\sqrt{2\pi}\,\sigma}.

For :math:`f(s) = e^{2i\pi(a s^2/2 + b s)}`, with :math:`\delta = a t + b - \eta`
(distance to the instantaneous frequency) and :math:`D = 1 + 4\pi^2 a^2 \sigma^4`,

.. math::

    V_g f(t, \eta) = \frac{e^{2i\pi(a t^2/2 + b t)}}{\sqrt{1 - 2i\pi a \sigma^2}}
        \exp\!\Big(-\frac{2\pi^2\sigma^2\delta^2\,(1 + 2i\pi a\sigma^2)}{D}\Big),
    \qquad |V_g f| = D^{-1/4} e^{-2\pi^2\sigma^2\delta^2 / D}.

For :math:`f = f_1 + f_2` the spectrogram splits as
:math:`|V f|^2 = |V f_1|^2 + |V f_2|^2 + I`, with the interference term
:math:`I = 2\,\mathrm{Re}(V f_1\, \overline{V f_2})`.

For opposite rates :math:`\pm a` and instantaneous-frequency gap :math:`\Delta(t)`,
:math:`\max_\eta 2|V f_1||V f_2| / \max_\eta(|V f_1|^2 + |V f_2|^2) \approx 2 e^{-\pi^2\sigma^2\Delta^2 / D}`
(once the components are separated), so the interference exceeds a fraction
:math:`\varepsilon` of the energy only where
:math:`\Delta(t) < \sqrt{D \ln(2/\varepsilon)} / (\pi\sigma)`.

The discrete STFT of :mod:`chirp_interf.stft` with
:func:`chirp_interf.windows.gaussian_seconds` approximates these expressions
(Riemann sum, frequency ``k fs / N``); ``tests/test_interference.py`` checks it.
"""

from __future__ import annotations

import numpy as np


def _D(a: float, sigma: float) -> float:
    r"""Spreading factor :math:`D = 1 + 4\pi^2 a^2 \sigma^4` (chirp rate ``a``, window std ``sigma``)."""
    return 1.0 + (2 * np.pi * a * sigma**2) ** 2


def chirp_stft(t, eta, a: float, b: float, sigma: float) -> np.ndarray:
    r"""Closed-form complex STFT of the linear chirp :math:`e^{2i\pi(a s^2/2 + b s)}`.

    .. math::

        V_g f(t, \eta) = \frac{e^{2i\pi(a t^2/2 + b t)}}{\sqrt{1 - 2i\pi a \sigma^2}}
            \exp\Big(-\frac{2\pi^2\sigma^2\delta^2(1 + 2i\pi a\sigma^2)}{D}\Big),
        \qquad \delta = a t + b - \eta.

    Parameters
    ----------
    t : array_like
        Times (s).
    eta : array_like
        Frequencies (Hz); broadcast against ``t``.
    a : float
        Chirp rate (Hz/s).
    b : float
        Initial frequency (Hz).
    sigma : float
        Standard deviation of the unit-mass Gaussian window (s).

    Returns
    -------
    numpy.ndarray
        Complex STFT with the broadcast shape of ``t`` and ``eta``.
    """
    t = np.asarray(t, dtype=float)
    eta = np.asarray(eta, dtype=float)
    delta = a * t + b - eta  # distance to the instantaneous frequency
    c = 2j * np.pi * a * sigma**2
    phase = np.exp(2j * np.pi * (0.5 * a * t**2 + b * t))
    return phase / np.sqrt(1.0 - c) * np.exp(-2 * np.pi**2 * sigma**2 * delta**2 * (1.0 + c) / _D(a, sigma))


def chirp_stft_modulus(t, eta, a: float, b: float, sigma: float) -> np.ndarray:
    r"""Modulus :math:`|V_g f(t, \eta)| = D^{-1/4} e^{-2\pi^2\sigma^2\delta^2 / D}` of :func:`chirp_stft`.

    Parameters
    ----------
    t, eta, a, b, sigma
        See :func:`chirp_stft`.

    Returns
    -------
    numpy.ndarray
        Real, non-negative array.
    """
    delta = a * np.asarray(t, float) + b - np.asarray(eta, float)
    D = _D(a, sigma)
    return D**-0.25 * np.exp(-2 * np.pi**2 * sigma**2 * delta**2 / D)


def decomposition(t, eta, components, sigma: float):
    r"""Closed-form decomposition :math:`|V_g f|^2 = E_1 + E_2 + I` for two linear chirps.

    Parameters
    ----------
    t : array_like
        Times (s), length ``n_t``.
    eta : array_like
        Frequencies (Hz), length ``n_eta``.
    components : sequence of two (a, b) pairs
        Rate (Hz/s) and initial frequency (Hz) of each chirp, ``[(a1, b1), (a2, b2)]``.
    sigma : float
        Window standard deviation (s).

    Returns
    -------
    E1, E2 : numpy.ndarray
        Spectrograms :math:`|V_g f_j|^2` of each component, shape ``(n_eta, n_t)``.
    I : numpy.ndarray
        Interference term :math:`2\,\mathrm{Re}(V_g f_1\,\overline{V_g f_2})`, same shape.
    """
    (a1, b1), (a2, b2) = components
    T, H = np.meshgrid(np.asarray(t, float), np.asarray(eta, float))  # rows = frequency, columns = time
    V1 = chirp_stft(T, H, a1, b1, sigma)
    V2 = chirp_stft(T, H, a2, b2, sigma)
    return np.abs(V1) ** 2, np.abs(V2) ** 2, 2.0 * np.real(V1 * np.conj(V2))


def interference_ratio(E1, E2) -> np.ndarray:
    r"""Relative interference level of each frame.

    .. math:: r(t) = \frac{\max_\eta 2|V_g f_1||V_g f_2|}{\max_\eta (E_1 + E_2)}

    The envelope :math:`2|V_g f_1||V_g f_2| = 2\sqrt{E_1 E_2}` is used instead of
    :math:`|I|`, which oscillates in time through the cosine of the phase
    difference and would split the zone into fragments.

    Parameters
    ----------
    E1, E2 : numpy.ndarray
        Component spectrograms, shape ``(n_eta, n_t)`` (see :func:`decomposition`).

    Returns
    -------
    numpy.ndarray
        Ratio ``r``, length ``n_t``, with values in ``[0, 1]``.
    """
    return np.max(2.0 * np.sqrt(E1 * E2), axis=0) / np.max(E1 + E2, axis=0)


def intervals(mask) -> list[tuple[int, int]]:
    """Maximal runs of ``True`` in a one-dimensional boolean array.

    Parameters
    ----------
    mask : array_like of bool

    Returns
    -------
    list of (int, int)
        Inclusive ``(start, end)`` indices of each run, in increasing order.

    Examples
    --------
    >>> intervals([0, 1, 1, 0, 1])
    [(1, 2), (4, 4)]
    """
    mask = np.asarray(mask, dtype=bool)
    # +1 where a run starts, -1 one past where it ends (padding handles runs touching the edges)
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1) - 1
    return list(zip(starts.tolist(), ends.tolist()))


def interference_zones(t, eta, components, sigma: float, eps: float = 0.1):
    """Time intervals where the interference exceeds ``eps`` times the component energy.

    Parameters
    ----------
    t, eta, components, sigma
        See :func:`decomposition`.
    eps : float, default 0.1
        Threshold on :func:`interference_ratio`.

    Returns
    -------
    zones : list of (float, float)
        ``(t_start, t_end)`` of each interval, in seconds.
    ratio : numpy.ndarray
        Interference ratio of each frame.
    """
    E1, E2, _ = decomposition(t, eta, components, sigma)
    ratio = interference_ratio(E1, E2)
    t = np.asarray(t, float)
    return [(float(t[i]), float(t[j])) for i, j in intervals(ratio > eps)], ratio


def symmetric_gap_threshold(a: float, sigma: float, eps: float = 0.1) -> float:
    r"""Critical frequency gap for two chirps of rates :math:`\pm a`.

    The interference exceeds ``eps`` while the gap between the instantaneous
    frequencies satisfies :math:`\Delta(t) < \sqrt{D \ln(2/\varepsilon)} / (\pi\sigma)`.

    Parameters
    ----------
    a : float
        Chirp rate (Hz/s); only ``|a|`` matters.
    sigma : float
        Window standard deviation (s).
    eps : float, default 0.1
        Relative interference level.

    Returns
    -------
    float
        Critical gap (Hz).
    """
    return float(np.sqrt(_D(a, sigma) * np.log(2.0 / eps)) / (np.pi * sigma))
