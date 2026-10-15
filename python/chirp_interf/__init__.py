"""chirp_interf — time-frequency analysis of multicomponent chirps.

The package is organised as follows:

* :mod:`chirp_interf.stft`         — discrete STFT on the time lattice ``{m R + p}``;
* :mod:`chirp_interf.recovery`     — recovery techniques RT1/RT2/RT3 from a thresholded STFT;
* :mod:`chirp_interf.ridges`       — greedy multi-ridge extraction;
* :mod:`chirp_interf.interference` — closed-form Gaussian STFT of linear chirps and interference term;
* :mod:`chirp_interf.signals`, :mod:`chirp_interf.windows` — test signals and analysis windows;
* :mod:`chirp_interf.reference`    — pure-NumPy oracle used by the tests;
* :mod:`chirp_interf.logger`, :mod:`chirp_interf.plotting` — logging and figure helpers.

The numerical kernels (FFT, frame analysis/synthesis, RT1–RT3) are implemented
in C++17 with OpenMP and exposed through the compiled module ``chirp_interf._core``.
"""

from . import interference, reference, ridges, signals, windows
from ._core import has_openmp
from .recovery import estimate_noise_level, noise_level, recover, rt1, rt2, rt3, snr_db
from .stft import frame_times, frequencies, n_frames, spectrogram, stft

__all__ = [
    # submodules
    "interference", "reference", "ridges", "signals", "windows",
    # build information
    "has_openmp",
    # STFT
    "stft", "spectrogram", "frequencies", "frame_times", "n_frames",
    # recovery
    "rt1", "rt2", "rt3", "recover", "snr_db", "noise_level", "estimate_noise_level",
]
__version__ = "0.2.0"
