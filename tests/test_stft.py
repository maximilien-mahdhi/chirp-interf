"""Tests of the STFT: C++ core against the NumPy oracle, axes and parameter validation."""

import numpy as np
import pytest

import chirp_interf as ci
from chirp_interf import _core, reference, windows


@pytest.fixture
def signal():
    """Random complex test signal of length 257 (odd, not a power of two)."""
    rng = np.random.default_rng(0)
    return rng.standard_normal(257) + 1j * rng.standard_normal(257)


@pytest.mark.parametrize("R,p", [(1, 0), (2, 1), (5, 3), (9, 0)])
@pytest.mark.parametrize("N", [21, 32, 100])
def test_frames_match_numpy_reference(signal, R, p, N):
    """The C++ frames equal the FFT-based reference, frames outside the signal included."""
    g = windows.gaussian(3.0, M=10)
    got = _core.stft_frames(signal, g, N, R, p, -4, 40)
    want = reference.stft_frames(signal, g, N, R, p, -4, 40)
    np.testing.assert_allclose(got, want, atol=1e-12)


def test_frames_match_pointwise(signal):
    """Frame-wise and pointwise evaluations agree."""
    g = windows.gaussian(2.0)
    N, R, p = 64, 3, 2
    V = ci.stft(signal, g, N, R, p)
    for m, k in [(0, 0), (5, 7), (20, 63), (ci.n_frames(len(signal), R, p) - 1, 31)]:
        assert V[k, m] == pytest.approx(_core.stft_point(signal, g, m * R + p, k / N), abs=1e-12)


def test_shape_and_axes():
    """Output shape and time / frequency axes of ``spectrogram``."""
    g = windows.gaussian(2.0)
    t, freqs, V = ci.spectrogram(np.ones(100), g, 32, R=4, p=1, fs=10.0)
    assert V.shape == (32, ci.n_frames(100, 4, 1)) == (32, 25)
    assert t[0] == pytest.approx(0.1) and freqs[0] == pytest.approx(-5.0)


def test_invalid_parameters(signal):
    """Inconsistent parameters raise ``ValueError`` with an explicit message."""
    g = windows.gaussian(3.0, M=10)
    with pytest.raises(ValueError, match="N must be"):
        ci.stft(signal, g, 20)
    with pytest.raises(ValueError, match="shift p"):
        ci.stft(signal, g, 64, R=3, p=3)
    with pytest.raises(ValueError, match="odd"):
        ci.stft(signal, np.ones(4), 64)
