"""Compare the recovery techniques RT1, RT2 and RT3 as a function of the shift p.

The signal is the sum of two crossing linear chirps corrupted by white Gaussian
noise. For every shift p in {0, ..., R-1}, the STFT on the lattice {mR + p}
is hard-thresholded at 3*gamma and inverted with each technique. The script
logs the SNR range of each technique and produces a two-panel figure:

* left  : output SNR versus p, for each technique;
* right : reconstructions at the shift where RT1 performs worst.

Usage
-----
    uv run chirp-recovery                      # default setting of the README
    uv run chirp-recovery --sigma 1.0 --hop 8  # noisier, denser lattice
    uv run chirp-recovery --estimate           # gamma estimated from the data
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import chirp_interf as ci
from chirp_interf import signals, windows
from chirp_interf.logger import get_logger, setup_logging
from chirp_interf.plotting import configure_matplotlib, save_figure

log = get_logger(__name__)

#: Colour of each technique, shared by both panels.
COLORS = {"RT1": "tab:blue", "RT2": "tab:purple", "RT3": "tab:green"}


def parse_args() -> argparse.Namespace:
    """Parse the command-line options (see ``--help``)."""
    ap = argparse.ArgumentParser(
        prog="chirp-recovery", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--fs", type=float, default=1000.0, help="sampling rate in Hz (default: 1000)")
    ap.add_argument("--sigma", type=float, default=0.6, help="noise standard deviation (default: 0.6)")
    ap.add_argument("--win", type=float, default=8.0, help="Gaussian window std in samples (default: 8)")
    ap.add_argument("--N", type=int, default=128, help="number of frequency bins, N >= 2M+1 (default: 128)")
    ap.add_argument("--hop", type=int, default=16, help="downsampling factor R (default: 16)")
    ap.add_argument("--estimate", action="store_true", help="estimate gamma from the data (MAD)")
    ap.add_argument("--seed", type=int, default=55, help="seed of the noise generator (default: 55)")
    ap.add_argument(
        "--output", type=Path, default=Path("figures/recovery.png"), help="output figure (default: %(default)s)"
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


def snr_versus_shift(f, noisy, g, N: int, R: int, threshold: float) -> dict[str, np.ndarray]:
    """Output SNR of every technique for every shift ``p = 0, ..., R-1``.

    Parameters
    ----------
    f : numpy.ndarray
        Clean signal (reference for the SNR).
    noisy : numpy.ndarray
        Observed signal.
    g : numpy.ndarray
        Analysis window.
    N, R : int
        Number of frequency bins and hop.
    threshold : float
        Hard threshold applied to the STFT.

    Returns
    -------
    dict
        ``{"RT1": snr, "RT2": snr, "RT3": snr}``, each an array of length ``R`` (dB).
    """
    return {m: np.array([ci.snr_db(f, ci.recover(m, noisy, g, N, R, p, threshold)) for p in range(R)]) for m in COLORS}


def plot(t, f, noisy, g, args, threshold: float, snr: dict[str, np.ndarray]) -> plt.Figure:
    """Build the two-panel figure (SNR versus p, reconstructions at the worst p for RT1).

    Parameters
    ----------
    t : numpy.ndarray
        Time axis (s).
    f, noisy : numpy.ndarray
        Clean and noisy signals.
    g : numpy.ndarray
        Analysis window.
    args : argparse.Namespace
        Command-line options (``N`` and ``hop`` are used).
    threshold : float
        Hard threshold applied to the STFT.
    snr : dict
        Output of :func:`snr_versus_shift`.

    Returns
    -------
    matplotlib.figure.Figure
    """
    R = args.hop
    shifts = np.arange(R)
    snr_in = ci.snr_db(f, noisy)
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13, 4.4), gridspec_kw={"width_ratios": [1, 1.6]})

    # Left: SNR versus shift.
    for m, s in snr.items():
        ax0.plot(shifts, s, "o-", color=COLORS[m], label=m, ms=4)
    ax0.axhline(snr_in, color="grey", ls="--", lw=1, label="noisy input")
    ax0.set(
        xlabel=r"shift $p$", ylabel=r"output SNR (dB)", title=rf"Recovery SNR vs shift ($R={R}$, threshold $3\gamma$)"
    )
    ax0.legend()

    # Right: zoom on the reconstructions at the worst shift for RT1.
    p_worst = int(np.argmin(snr["RT1"]))
    zoom = (t > 0.30) & (t < 0.45)
    ax1.plot(t[zoom], noisy.real[zoom], color="0.75", lw=0.8, label="noisy")
    for m, color in COLORS.items():
        rec = ci.recover(m, noisy, g, args.N, R, p_worst, threshold)
        ax1.plot(t[zoom], rec.real[zoom], color=color, lw=1.2, label=m)
    ax1.plot(t[zoom], f.real[zoom], "k", lw=1.5, label="clean")
    ax1.set(
        xlabel=r"time (s)",
        ylabel=r"$\mathrm{Re}\, f$",
        title=rf"Reconstructions at the worst shift for RT1 ($p={p_worst}$)",
    )
    ax1.legend(ncol=5, fontsize=8, loc="upper right")
    fig.tight_layout()
    return fig


def main() -> None:
    """Run the experiment, log the SNR ranges and save the figure."""
    args = parse_args()
    setup_logging(args.log_level)
    configure_matplotlib(usetex=args.usetex)

    t = signals.time_axis(args.fs, 1.0)
    f = signals.crossing_chirps(t, a=-200.0, b1=150.0, b2=-150.0)
    noisy = signals.add_noise(f, args.sigma, rng=args.seed)
    g = windows.gaussian(args.win)
    log.info("L=%d, 2M+1=%d, N=%d, R=%d, sigma=%.3g", len(t), len(g), args.N, args.hop, args.sigma)

    gamma = ci.estimate_noise_level(noisy, g, args.N, args.hop) if args.estimate else ci.noise_level(args.sigma, g)
    log.info("%s gamma = %.4g, threshold = 3 gamma", "estimated" if args.estimate else "exact", gamma)

    snr = snr_versus_shift(f, noisy, g, args.N, args.hop, 3 * gamma)
    log.info("input SNR: %.2f dB", ci.snr_db(f, noisy))
    for m, s in snr.items():
        log.info("%s: SNR in [%.2f, %.2f] dB, best p = %d", m, s.min(), s.max(), int(np.argmax(s)))

    fig = plot(t, f, noisy, g, args, 3 * gamma, snr)
    save_figure(fig, args.output)
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
