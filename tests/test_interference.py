"""Tests of the closed forms, interference zones and ridge extraction."""

import numpy as np
import pytest

import chirp_interf as ci
from chirp_interf import interference as itf
from chirp_interf import ridges, signals, windows

FS, SIGMA, A, B1, B2 = 1000.0, 0.01, -100.0, 50.0, -50.0


@pytest.fixture(scope="module")
def setup():
    """Discrete spectrogram of two crossing chirps (rates -100 / +100 Hz/s, sigma = 10 ms)."""
    t = signals.time_axis(FS, 1.0)
    g = windows.gaussian_seconds(SIGMA, FS)
    f = signals.crossing_chirps(t, A, B1, B2)
    times, freqs, V = ci.spectrogram(f, g, 512, fs=FS)
    inner = (times > 0.05) & (times < 0.95)   # away from zero-padded edges
    return t, g, times, freqs, V, inner


def test_closed_form_single_chirp(setup):
    """Discrete STFT of one chirp equals the closed form (complex values and modulus)."""
    t, g, times, freqs, _, inner = setup
    V = np.fft.fftshift(ci.stft(signals.linear_chirp(t, A, B1), g, 512), axes=0)
    T, H = np.meshgrid(times, freqs)
    np.testing.assert_allclose(V[:, inner], itf.chirp_stft(T, H, A, B1, SIGMA)[:, inner], atol=1e-4)
    np.testing.assert_allclose(np.abs(V[:, inner]),
                               itf.chirp_stft_modulus(T, H, A, B1, SIGMA)[:, inner], atol=1e-4)


def test_spectrogram_decomposition(setup):
    """Discrete |V_g f|^2 equals E1 + E2 + I computed in closed form."""
    _, _, times, freqs, V, inner = setup
    E1, E2, I = itf.decomposition(times, freqs, [(A, B1), (-A, B2)], SIGMA)
    np.testing.assert_allclose((np.abs(V) ** 2)[:, inner], (E1 + E2 + I)[:, inner], atol=1e-3)


def test_interference_zone_matches_analytic_gap(setup):
    """The zone measured on the closed form matches the analytic gap bound."""
    _, _, times, freqs, _, _ = setup
    zones, _ = itf.interference_zones(times, freqs, [(A, B1), (-A, B2)], SIGMA, eps=0.1)
    assert len(zones) == 1
    # gap Delta(t) = |2 A t + B1 - B2| = 200 |t - 0.5|
    half = itf.symmetric_gap_threshold(A, SIGMA, eps=0.1) / (2 * abs(A))
    lo, hi = zones[0]
    assert lo == pytest.approx(0.5 - half, abs=5e-3)
    assert hi == pytest.approx(0.5 + half, abs=5e-3)


def test_intervals_runs():
    """``intervals`` returns the maximal runs of True, edges included."""
    assert itf.intervals([0, 1, 1, 0, 1]) == [(1, 2), (4, 4)]
    assert itf.intervals([1, 1]) == [(0, 1)]
    assert itf.intervals([0, 0]) == []


def test_ridges_follow_separated_components():
    """On noiseless separated chirps, both ridges stay within one frequency bin."""
    fs, N = 1000.0, 256
    t = signals.time_axis(fs, 1.0)
    f = signals.linear_chirp(t, 100.0, -200.0) + signals.linear_chirp(t, 100.0, 50.0)
    times, freqs, V = ci.spectrogram(f, windows.gaussian_seconds(0.01, fs), N, fs=fs)
    jump = ridges.max_jump_for_rate(100.0, N, fs)
    est = np.sort(freqs[ridges.extract_ridges(V, 2, jump)], axis=0)
    true = np.vstack([100 * times - 200, 100 * times + 50])
    assert np.max(np.abs(est - true)) <= fs / N

