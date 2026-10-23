"""Matplotlib configuration (LaTeX rendering) and figure saving.

All figures of the repository go through :func:`configure_matplotlib`, so the
LaTeX setup lives in a single place. Rendering with ``usetex`` requires a
LaTeX distribution with ``dvipng`` (``texlive-latex-extra``, ``dvipng`` and
``cm-super`` on Debian/Ubuntu).
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

#: Preamble passed to LaTeX when ``usetex`` is enabled.
LATEX_PREAMBLE = r"\usepackage{amsmath}\usepackage{amssymb}\usepackage{amsfonts}"


def latex_available() -> bool:
    """Return ``True`` if the ``latex`` and ``dvipng`` executables are on the ``PATH``."""
    return shutil.which("latex") is not None and shutil.which("dvipng") is not None


def configure_matplotlib(usetex: bool = True) -> None:
    """Set the global Matplotlib style used by every figure.

    Parameters
    ----------
    usetex : bool, default True
        Render all text with LaTeX (Computer Modern, real math typesetting). If
        LaTeX is not installed, a warning is logged and Matplotlib's built-in
        mathtext is used instead, so the scripts still run.
    """
    if usetex and not latex_available():
        logger.warning("LaTeX (latex + dvipng) not found: falling back to mathtext")
        usetex = False

    plt.rcParams.update(
        {
            "text.usetex": usetex,
            "font.family": "serif",
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "axes.grid": False,
            "figure.dpi": 100,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
        }
    )
    if usetex:
        plt.rcParams["text.latex.preamble"] = LATEX_PREAMBLE
    else:
        plt.rcParams["mathtext.fontset"] = "cm"
    logger.debug("matplotlib configured (usetex=%s)", usetex)


def save_figure(fig: plt.Figure, path: str | Path | None) -> Path | None:
    """Save ``fig`` to ``path``, creating parent directories if needed.

    The figure is not closed, so the caller can still display it with ``plt.show()``.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save.
    path : str, Path or None
        Destination file; the extension selects the format. ``None`` skips
        saving (the figure is left open, e.g. for ``plt.show()``).

    Returns
    -------
    Path or None
        The path written, or ``None`` if nothing was saved.
    """
    if path is None:
        return None
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    logger.info("figure saved to %s", path)
    return path
