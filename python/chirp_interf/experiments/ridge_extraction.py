"""Greedy ridge extraction on noisy two-component signals.

Two cases are compared, under the same noise level:

* left  : two parallel chirps (same rate, separated by 200 Hz) — both ridges are recovered;
* right : two crossing chirps (rates -200 and +200 Hz/s) — after the first ridge is
  erased, the second component is missing inside the interference zone (shaded),
  computed in closed form with :func:`chirp_interf.interference.interference_zones`.

Usage
-----
    uv run chirp-ridges               # setting of the README
    uv run chirp-ridges --sigma 2.0   # noisier
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import chirp_interf as ci
from chirp_interf import interference as itf
from chirp_interf import ridges, signals, windows
from chirp_interf.logger import get_logger, setup_logging
from chirp_interf.plotting import configure_matplotlib, save_figure

log = get_logger(__name__)

#: Chirp rate of every component (Hz/s).
RATE = 200.0


def parse_args() -> argparse.Namespace:
    """Parse the command-line options (see ``--help``)."""
    ap = argparse.ArgumentParser(
        prog="chirp-ridges", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--fs", type=float, default=1000.0, help="sampling rate in Hz (default: 1000)")
    ap.add_argument("--sigma", type=float, default=1.0, help="noise standard deviation (default: 1.0)")
    ap.add_argument("--win", type=float, default=0.01, help="Gaussian window std in seconds (default: 0.01)")
    ap.add_argument("--N", type=int, default=256, help="number of frequency bins (default: 256)")
    ap.add_argument("--hop", type=int, default=4, help="hop R between frames (default: 4)")
    ap.add_argument("--seed", type=int, default=55, help="seed of the noise generator (default: 55)")
    ap.add_argument("--output", type=Path, default=Path("figures/ridges.png"),
                    help="output figure (default: %(default)s)")
    ap.add_argument("--usetex", action=argparse.BooleanOptionalAction, default=True,
                    help="render text with LaTeX, or with matplotlib's mathtext (default: --usetex)")
    ap.add_argument("--show", action="store_true", help="open the figure in a window")
    ap.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                    help="verbosity (default: INFO)")
    return ap.parse_args()


def panel(ax, f, g, args, title: str, zone=None) -> np.ndarray:
    """Draw the spectrogram of ``f`` and its two extracted ridges on ``ax``.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    f : numpy.ndarray
        Noisy signal.
    g : numpy.ndarray
        Analysis window.
    args : argparse.Namespace
        Command-line options (``N``, ``hop``, ``fs`` and ``win`` are used).
    title : str
        Panel title.
    zone : (float, float), optional
        Interference zone to shade (s).

    Returns
    -------
    numpy.ndarray
        Ridge frequencies (Hz), shape ``(2, n_frames)``.
    """
    times, freqs, V = ci.spectrogram(f, g, args.N, R=args.hop, fs=args.fs)
    jump = ridges.max_jump_for_rate(RATE, args.N, args.fs, args.hop)
    band = ridges.band_for_window(args.win, RATE, args.N, args.fs)
    log.info("%s: max jump = %d bins, erased band = +/- %d bins", title, jump, band)
    rid = freqs[ridges.extract_ridges(V, 2, jump, band=band)]

    ax.pcolormesh(times, freqs, np.abs(V), cmap="Greys", shading="auto", rasterized=True)
    for l, color in enumerate(("tab:orange", "tab:cyan")):
        ax.plot(times, rid[l], ".", ms=3, color=color, label=rf"ridge {l + 1}")
    if zone:
        ax.axvspan(*zone, color="tab:red", alpha=0.12, label=r"interference zone")
    ax.set(title=title, xlabel=r"time (s)", ylabel=r"frequency (Hz)", ylim=(-250, 250))
    ax.legend(loc="upper right", fontsize=8)
    return rid


def main() -> None:
    """Extract the ridges of both signals, log the errors on the separated case and save the figure."""
    args = parse_args()
    setup_logging(args.log_level)
    configure_matplotlib(usetex=args.usetex)

    t = signals.time_axis(args.fs, 1.0)
    g = windows.gaussian_seconds(args.win, args.fs)
    rng = np.random.default_rng(args.seed)

    parallel = signals.linear_chirp(t, RATE, -200.0) + signals.linear_chirp(t, RATE, 0.0)
    crossing = signals.crossing_chirps(t, -RATE, 100.0, -100.0)
    zones, _ = itf.interference_zones(ci.frame_times(len(t), args.hop, 0, args.fs), ci.frequencies(args.N, args.fs),
                                      [(-RATE, 100.0), (RATE, -100.0)], args.win)
    log.info("interference zone of the crossing case: %s s", [(round(a, 3), round(b, 3)) for a, b in zones])

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    rid = panel(axes[0], signals.add_noise(parallel, args.sigma, rng=rng), g, args, r"Separated components")
    panel(axes[1], signals.add_noise(crossing, args.sigma, rng=rng), g, args, r"Crossing components",
          zone=zones[0] if zones else None)

    # Accuracy on the separated case (true instantaneous frequencies RATE t - 200 and RATE t).
    times = ci.frame_times(len(t), args.hop, 0, args.fs)
    true = np.vstack([RATE * times - 200.0, RATE * times])
    err = np.abs(np.sort(rid, axis=0) - true)
    log.info("separated case: median / max ridge error = %.2f / %.2f Hz (bin width %.2f Hz)",
             np.median(err), err.max(), args.fs / args.N)

    fig.tight_layout()
    save_figure(fig, args.output)
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
