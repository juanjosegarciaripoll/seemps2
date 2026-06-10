"""Run the ``>>>`` examples embedded in the library docstrings.

This module collects every :mod:`doctest` example found in the docstrings of
the :mod:`seemps` package and exposes them to ``unittest`` through the
``load_tests`` protocol.  It guarantees that the examples shown in the
documentation stay in sync with the actual behaviour of the code and do not
rot over time.

New doctests are picked up automatically: any module under :mod:`seemps` whose
docstrings contain ``>>>`` examples is discovered and executed, so there is
nothing to register here when documentation is added.
"""

from __future__ import annotations

import doctest
import importlib
import os
import pkgutil
import tempfile
import unittest

import seemps

# Doctests are matched leniently with respect to surrounding whitespace, which
# makes the expected output robust against the way NumPy aligns the rows of an
# array across versions.
OPTIONFLAGS = doctest.NORMALIZE_WHITESPACE | doctest.ELLIPSIS


def _iter_seemps_modules():
    """Yield every importable module in the :mod:`seemps` package."""
    yield seemps
    for info in pkgutil.walk_packages(seemps.__path__, prefix=seemps.__name__ + "."):
        try:
            yield importlib.import_module(info.name)
        except Exception as exc:  # pragma: no cover - surfaced as a failing test
            raise ImportError(
                f"Could not import {info.name!r} while collecting doctests"
            ) from exc


def _run_in_tmpdir(func):
    """Wrap a doctest setUp/tearDown so examples run in a scratch directory.

    Some examples (e.g. those in :mod:`seemps.hdf5`) write files to the current
    directory.  Running them inside a throw-away working directory keeps the
    repository clean.
    """
    state: dict[str, object] = {}

    def setUp(_test):
        state["cwd"] = os.getcwd()
        state["tmp"] = tempfile.TemporaryDirectory(prefix="seemps_doctest_")
        os.chdir(state["tmp"].name)

    def tearDown(_test):
        os.chdir(state["cwd"])  # type: ignore[arg-type]
        state["tmp"].cleanup()  # type: ignore[union-attr]

    return setUp, tearDown


def load_tests(loader, tests, ignore):  # noqa: ARG001 (unittest protocol)
    """Collect the doctest examples from the whole :mod:`seemps` package."""
    setUp, tearDown = _run_in_tmpdir(None)
    for module in _iter_seemps_modules():
        try:
            suite = doctest.DocTestSuite(
                module,
                optionflags=OPTIONFLAGS,
                setUp=setUp,
                tearDown=tearDown,
            )
        except ValueError:
            # Raised when the module has no docstrings with examples.
            continue
        if suite.countTestCases():
            tests.addTests(suite)
    return tests


if __name__ == "__main__":
    unittest.main()
