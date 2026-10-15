"""Terminal logging based on Rich.

Every module of the package logs through the standard :mod:`logging` library
(``logger = logging.getLogger(__name__)``) and never configures handlers
itself. Scripts call :func:`setup_logging` once at start-up to get coloured,
timestamped output on stderr.

Example
-------
>>> from chirp_interf.logger import get_logger, setup_logging
>>> setup_logging()
>>> log = get_logger(__name__)
>>> log.info("RT2: SNR = %.2f dB", 16.4)
"""

from __future__ import annotations

import logging

from rich.console import Console
from rich.logging import RichHandler

#: Shared console writing to stderr, so that logs never mix with data printed on stdout.
console = Console(stderr=True)

_FORMAT = "%(message)s"
_DATE_FORMAT = "[%X]"


def setup_logging(level: int | str = logging.INFO) -> None:
    """Install a Rich handler on the root logger.

    Call it once, at the start of a script. Calling it again replaces the
    previous configuration (``force=True``), so it is safe in notebooks.

    Parameters
    ----------
    level : int or str, default ``logging.INFO``
        Minimum level of the messages displayed (``"DEBUG"``, ``"INFO"``, ...).
    """
    handler = RichHandler(
        console=console,
        show_time=True,
        show_level=True,
        show_path=False,
        rich_tracebacks=True,
        markup=False,
    )
    logging.basicConfig(level=level, format=_FORMAT, datefmt=_DATE_FORMAT, handlers=[handler], force=True)


def get_logger(name: str) -> logging.Logger:
    """Return the logger called ``name`` (thin wrapper around :func:`logging.getLogger`).

    Parameters
    ----------
    name : str
        Dotted logger name, typically ``__name__``.

    Returns
    -------
    logging.Logger
    """
    return logging.getLogger(name)
