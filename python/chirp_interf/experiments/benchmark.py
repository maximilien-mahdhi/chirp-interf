"""Time the C++/OpenMP core against the vectorised NumPy reference.

Each recovery technique is run on the same noisy two-chirp signal by both
implementations; the best of ``--repeat`` runs is reported. The number of
OpenMP threads can be set with the ``OMP_NUM_THREADS`` environment variable.

Usage
-----
    uv run chirp-benchmark                       # setting of the README
    uv run chirp-benchmark --L 100000 --N 256
    OMP_NUM_THREADS=1 uv run chirp-benchmark     # single-threaded core
"""

from __future__ import annotations

import argparse
import os
import time
from collections.abc import Callable

import numpy as np
from rich.table import Table

import chirp_interf as ci
from chirp_interf import reference, signals, windows
from chirp_interf.logger import console, get_logger, setup_logging

log = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse the command-line options (see ``--help``)."""
    ap = argparse.ArgumentParser(
        prog="chirp-benchmark", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--L", type=int, default=20000, help="signal length (default: 20000)")
    ap.add_argument("--N", type=int, default=128, help="number of frequency bins (default: 128)")
    ap.add_argument("--win", type=float, default=8.0, help="Gaussian window std in samples (default: 8)")
    ap.add_argument("--hop", type=int, default=4, help="hop R (default: 4)")
    ap.add_argument("--repeat", type=int, default=3, help="runs per measurement, best kept (default: 3)")
    ap.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                    help="verbosity (default: INFO)")
    return ap.parse_args()


def best_of(fn: Callable[[], object], repeat: int) -> float:
    """Best wall-clock time of ``repeat`` calls to ``fn``, in seconds.

    Parameters
    ----------
    fn : callable
        Function without arguments.
    repeat : int
        Number of calls.

    Returns
    -------
    float
        Minimum duration (the least perturbed by other processes).
    """
    best = np.inf
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def main() -> None:
    """Run the benchmark and print a table of timings."""
    args = parse_args()
    setup_logging(args.log_level)

    fs = 10000.0
    t = np.arange(args.L) / fs
    f = signals.add_noise(signals.crossing_chirps(t, -2000.0, 1500.0, -1500.0), 0.5, rng=55)
    g = windows.gaussian(args.win)
    thr = 3 * ci.noise_level(0.5, g)
    log.info("L=%d, N=%d, 2M+1=%d, R=%d, OpenMP=%s, threads=%s", args.L, args.N, len(g), args.hop,
             ci.has_openmp(), os.environ.get("OMP_NUM_THREADS", os.cpu_count()))

    table = Table(title=f"Recovery timings (best of {args.repeat})")
    for col in ("method", "C++ / OpenMP (ms)", "NumPy reference (ms)", "speed-up"):
        table.add_column(col, justify="right")
    for name in ("rt1", "rt2", "rt3"):
        fast, slow = getattr(ci, name), getattr(reference, name)
        cpp = best_of(lambda fn=fast: fn(f, g, args.N, args.hop, 0, thr), args.repeat)
        ref = best_of(lambda fn=slow: fn(f, g, args.N, args.hop, 0, thr), args.repeat)
        table.add_row(name.upper(), f"{1e3 * cpp:.1f}", f"{1e3 * ref:.1f}", f"{ref / cpp:.1f}x")
    console.print(table)


if __name__ == "__main__":
    main()
