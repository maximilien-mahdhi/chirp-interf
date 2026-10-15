"""Tests of the recovery techniques RT1, RT2, RT3 and of the noise-level estimator."""

import numpy as np
import pytest

import chirp_interf as ci
from chirp_interf import reference, signals, windows

CASES = [(1, 0), (2, 1), (4, 0), (4, 3), (7, 3), (11, 5)]


@pytest.fixture
def chirp():
    """Two crossing chirps sampled at 500 Hz during 1 s."""
    t = signals.time_axis(500.0, 1.0)
    return signals.crossing_chirps(t, a=-150.0, b1=80.0, b2=-70.0)


@pytest.mark.parametrize("method", ["RT1", "RT2", "RT3"])
@pytest.mark.parametrize("R,p", CASES)
def test_exact_without_threshold(chirp, method, R, p):
    """Without thresholding, every technique recovers the signal to machine precision."""
    g = windows.gaussian(4.0, M=12)
    out = ci.recover(method, chirp, g, 64, R, p)
    np.testing.assert_allclose(out, chirp, atol=1e-12)


@pytest.mark.parametrize("method", ["rt1", "rt2", "rt3"])
@pytest.mark.parametrize("R,p", CASES)
def test_thresholded_matches_reference(chirp, method, R, p):
    """With thresholding, the C++ core matches the NumPy oracle."""
    g = windows.gaussian(4.0, M=12)
    noisy = signals.add_noise(chirp, 0.5, rng=1)
    thr = 3 * ci.noise_level(0.5, g)
    got = getattr(ci, method)(noisy, g, 64, R, p, thr)
    want = getattr(reference, method)(noisy, g, 64, R, p, thr)
    np.testing.assert_allclose(got, want, atol=1e-10)


def test_thresholding_denoises(chirp):
    """Thresholding at 3 gamma improves the SNR by more than 5 dB (RT2, RT3)."""
    g = windows.gaussian(4.0, M=12)
    noisy = signals.add_noise(chirp, 0.5, rng=2)
    thr = 3 * ci.noise_level(0.5, g)
    snr_in = ci.snr_db(chirp, noisy)
    for method in ("RT2", "RT3"):
        assert ci.snr_db(chirp, ci.recover(method, noisy, g, 64, 3, 1, thr)) > snr_in + 5


def test_noise_level_estimate():
    """The MAD estimate of gamma is within 5 % of sigma ||g||_2 on pure noise."""
    g = windows.gaussian(5.0)
    noise = signals.add_noise(np.zeros(20000), 0.7, rng=3)
    assert ci.estimate_noise_level(noise, g, 64) == pytest.approx(ci.noise_level(0.7, g), rel=0.05)


def test_rt1_rejects_large_hop(chirp):
    """RT1 refuses a hop larger than the window allows."""
    with pytest.raises(ValueError, match="RT1"):
        ci.rt1(chirp, windows.gaussian(1.0, M=2), 16, R=8, p=0)
