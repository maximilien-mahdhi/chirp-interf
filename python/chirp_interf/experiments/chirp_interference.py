r"""Spectrogram of two crossing linear chirps versus its closed-form decomposition.

For f = f1 + f2 with rates -a and +a, the script computes:

* the discrete spectrogram |V_g f|^2;
* the interference term measured numerically, |V_g f|^2 - |V_g f1|^2 - |V_g f2|^2;
* the same term from the closed form 2 Re(V_g f1 conj(V_g f2));
* the relative interference level per frame and the zone where it exceeds eps,
  together with the analytic bound Delta(t) < sqrt(D ln(2/eps)) / (pi sigma).

It logs the maximal gap between the discrete and closed-form spectrograms, and
between the numerical and closed-form interference terms, on the frames whose
window lies inside the signal; then the interference zones, and saves a 2x2 figure.

Usage
-----
    uv run chirp-interference                         # setting of the README
    uv run chirp-interference --rate 300 --win 0.02   # faster chirps, longer window
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import chirp_interf as ci
from chirp_interf import interference as itf
from chirp_interf import signals, windows
from chirp_interf.logger import get_logger, setup_logging
from chirp_interf.plotting import configure_matplotlib, save_figure

log = get_logger(__name__)

#: Number of ridge standard deviations kept around the instantaneous frequencies
#: when cropping the frequency axis of the figure.
CROP_STD = 4.0


def parse_args() -> argparse.Namespace:
    """Parse the command-line options (see ``--help``)."""
    ap = argparse.ArgumentParser(
        prog="chirp-interference", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--fs", type=float, default=1000.0, help="sampling rate in Hz (default: 1000)")
    ap.add_argument("--rate", type=float, default=100.0, help="chirp rate |a| in Hz/s (default: 100)")
    ap.add_argument("--f0", type=float, default=50.0, help="initial frequencies are +f0 and -f0, in Hz (default: 50)")
    ap.add_argument("--win", type=float, default=0.01, help="Gaussian window std in seconds (default: 0.01)")
    ap.add_argument("--N", type=int, default=512, help="number of frequency bins (default: 512)")
    ap.add_argument("--eps", type=float, default=0.1, help="interference level defining the zone (default: 0.1)")
    ap.add_argument(
        "--output", type=Path, default=Path("figures/interference.png"), help="output figure (default: %(default)s)"
    )
    ap.add_argument(
        "--usetex",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="render text with LaTeX, or with matplotlib's mathtext (default: --usetex)",
    )
    ap.add_argument("--show", action="store_true", help="open the figure in a window")
    ap.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="verbosity (default: INFO)"
    )
    return ap.parse_args()


def plot(times, freqs, maps, ratio, zones, bound, args) -> plt.Figure:
    """Build the 2x2 figure.

    Parameters
    ----------
    times, freqs : numpy.ndarray
        Frame times (s) and centred frequencies (Hz).
    maps : list of (numpy.ndarray, str, str, bool)
        The three time-frequency maps to draw, as ``(values, title, colormap, symmetric)``;
        ``symmetric`` centres the colour scale on zero.
    ratio : numpy.ndarray
        Relative interference level per frame.
    zones : list of (float, float)
        Interference zones (s).
    bound : (float, float)
        Analytic interference zone (s).
    args : argparse.Namespace
        Command-line options (``f0``, ``rate`` and ``eps`` are used).

    Returns
    -------
    matplotlib.figure.Figure
    """
    # Crop empty frequencies: keep the range swept by the instantaneous frequencies
    # (f0 - rate t and -f0 + rate t for t in [0, 1]) plus CROP_STD ridge standard deviations.
    ridge_std = np.sqrt(1 + (2 * np.pi * args.rate * args.win**2) ** 2) / (2 * np.pi * args.win)
    if_max = max(abs(args.f0), abs(args.f0 - args.rate))
    sel = np.abs(freqs) <= if_max + CROP_STD * ridge_std
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.2), sharex=True)

    for ax, (Z, title, cmap, symmetric) in zip(axes.flat, maps):
        lim = {"vmin": -np.max(np.abs(Z[sel])), "vmax": np.max(np.abs(Z[sel]))} if symmetric else {}
        im = ax.pcolormesh(times, freqs[sel], Z[sel], cmap=cmap, shading="auto", rasterized=True, **lim)
        fig.colorbar(im, ax=ax)
        ax.set(title=title, ylabel=r"frequency (Hz)")

    ax = axes[1, 1]
    ax.semilogy(times, ratio, color="tab:red", label=r"closed form")
    ax.axhline(args.eps, color="k", ls="--", lw=1, label=rf"$\varepsilon = {args.eps}$")
    for lo, hi in zones:
        ax.axvspan(lo, hi, color="tab:red", alpha=0.12)
        for x in (lo, hi):
            for a_, c in zip(axes.flat[:3], ("w", "0.4", "0.4")):  # zone limits on the maps
                a_.axvline(x, color=c, ls=":", lw=1)
    for x in bound:  # draw only the limits inside the signal
        if times[0] <= x <= times[-1]:
            ax.axvline(x, color="tab:blue", lw=1)
    ax.plot([], [], color="tab:blue", lw=1, label=r"analytic bound")  # legend entry only
    ax.set(
        title=r"$\max_\eta 2|V_g f_1||V_g f_2| \,/\, \max_\eta (|V_g f_1|^2 + |V_g f_2|^2)$",
        ylim=(1e-6, 3),
        xlabel=r"time (s)",
    )
    ax.legend(loc="lower center")
    axes[1, 0].set_xlabel(r"time (s)")
    axes[0, 0].set_xlim(times[0], times[-1])  # shared by all panels
    fig.tight_layout()
    return fig


def main() -> None:
    """Compute the numerical and closed-form decompositions, log the comparison and save the figure."""
    args = parse_args()
    setup_logging(args.log_level)
    configure_matplotlib(usetex=args.usetex)

    a, b1, b2 = -args.rate, args.f0, -args.f0
    comps = [(a, b1), (-a, b2)]
    t = signals.time_axis(args.fs, 1.0)
    g = windows.gaussian_seconds(args.win, args.fs)
    log.info(
        "chirps: rates %+g / %+g Hz/s, crossing at t = %.3f s; window std %.3g s (2M+1 = %d)",
        a,
        -a,
        (b2 - b1) / (2 * a),
        args.win,
        len(g),
    )

    # Discrete spectrograms of the sum and of each component.
    times, freqs, V = ci.spectrogram(signals.crossing_chirps(t, a, b1, b2), g, args.N, fs=args.fs)
    _, _, V1 = ci.spectrogram(signals.linear_chirp(t, *comps[0]), g, args.N, fs=args.fs)
    _, _, V2 = ci.spectrogram(signals.linear_chirp(t, *comps[1]), g, args.N, fs=args.fs)
    I_num = np.abs(V) ** 2 - np.abs(V1) ** 2 - np.abs(V2) ** 2

    # Closed form and interference zones.
    E1, E2, I = itf.decomposition(times, freqs, comps, args.win)
    zones, ratio = itf.interference_zones(times, freqs, comps, args.win, args.eps)
    half = itf.symmetric_gap_threshold(a, args.win, args.eps) / (2 * args.rate)
    tc = (b2 - b1) / (2 * a)

    # Compare only the frames whose window lies entirely inside the signal: elsewhere
    # the zero padding of the discrete STFT departs from the closed form on R.
    half_window = (len(g) - 1) / 2 / args.fs
    inner = (times >= half_window) & (times <= times[-1] - half_window)
    if inner.any():
        err = np.max(np.abs(np.abs(V) ** 2 - (E1 + E2 + I))[:, inner])
        err_I = np.max(np.abs(I_num - I)[:, inner])
        log.info("on [%.3f, %.3f] s (window inside the signal):", times[inner][0], times[inner][-1])
        log.info("  max | |V_g f|^2 - (E1 + E2 + I) | = %.2e (peak %.3f)", err, np.max(np.abs(V) ** 2))
        log.info("  max | I numerical - I closed form | = %.2e (peak %.3f)", err_I, np.max(np.abs(I)))
    else:
        log.warning("window (half-length %.3f s) longer than half the signal: no frame to compare", half_window)
    log.info("interference zones (eps = %g): %s s", args.eps, [(round(lo, 3), round(hi, 3)) for lo, hi in zones])
    log.info("analytic bound: [%.3f, %.3f] s", tc - half, tc + half)

    maps = [
        (np.abs(V) ** 2, r"Discrete spectrogram $|V_g f|^2$", "magma", False),
        (I_num, r"Numerical: $|V_g f|^2 - |V_g f_1|^2 - |V_g f_2|^2$", "RdBu_r", True),
        (I, r"Closed form: $I = 2\,\mathrm{Re}\left(V_g f_1\, \overline{V_g f_2}\right)$", "RdBu_r", True),
    ]
    fig = plot(times, freqs, maps, ratio, zones, (tc - half, tc + half), args)
    save_figure(fig, args.output)
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
