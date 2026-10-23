"""Tests of the logging and plotting helpers."""

import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from chirp_interf.logger import get_logger, setup_logging  # noqa: E402
from chirp_interf.plotting import configure_matplotlib, save_figure  # noqa: E402


def test_setup_logging_installs_single_handler():
    """``setup_logging`` replaces the root handlers by one Rich handler at the requested level."""
    setup_logging("DEBUG")
    setup_logging(logging.WARNING)
    root = logging.getLogger()
    assert len(root.handlers) == 1 and root.level == logging.WARNING
    assert get_logger("chirp_interf.test").name == "chirp_interf.test"


def test_save_figure(tmp_path):
    """``save_figure`` creates missing directories, writes the file and skips ``None``."""
    configure_matplotlib(usetex=False)
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    out = save_figure(fig, tmp_path / "sub" / "fig.png")
    assert out is not None and out.exists() and out.stat().st_size > 0
    assert save_figure(fig, None) is None
    plt.close(fig)
