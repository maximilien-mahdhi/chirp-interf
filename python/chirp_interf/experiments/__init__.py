"""Command-line experiments that reproduce the figures and the benchmark of the README.

Each module exposes a ``main()`` function registered as a console script in
``pyproject.toml``:

==========================  =======================================================
command                     module
==========================  =======================================================
``chirp-recovery``          :mod:`chirp_interf.experiments.recovery_comparison`
``chirp-interference``      :mod:`chirp_interf.experiments.chirp_interference`
``chirp-ridges``            :mod:`chirp_interf.experiments.ridge_extraction`
``chirp-benchmark``         :mod:`chirp_interf.experiments.benchmark`
==========================  =======================================================

Run them with ``uv run <command>`` (or directly after ``pip install``), or as
modules with ``python -m chirp_interf.experiments.<module>``.
"""
