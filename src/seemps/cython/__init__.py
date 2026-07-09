"""SeeMPS core kernels.

By default the compiled Cython implementation (``core``) is used.  Set the
environment variable ``SEEMPS_BACKEND=python`` before importing SeeMPS to route
all numerical kernels and configuration types through the pure
NumPy/SciPy reference implementation in :mod:`._python_core` instead.
"""

import os
from importlib import import_module
from typing import Any

SEEMPS_BACKEND = os.environ.get("SEEMPS_BACKEND", "cython").lower()
if SEEMPS_BACKEND not in ("cython", "python"):
    raise ValueError(
        f"Invalid SEEMPS_BACKEND={SEEMPS_BACKEND!r}; expected 'cython' or 'python'"
    )

_impl: Any
if SEEMPS_BACKEND == "python":
    _impl = import_module("._python_core", __name__)
else:
    _impl = import_module(".core", __name__)

# Configuration types and constants
DEFAULT_STRATEGY = _impl.DEFAULT_STRATEGY
DEFAULT_TOLERANCE = _impl.DEFAULT_TOLERANCE
GemmOrder = _impl.GemmOrder
MAX_BOND_DIMENSION = _impl.MAX_BOND_DIMENSION
NO_TRUNCATION = _impl.NO_TRUNCATION
Simplification = _impl.Simplification
Strategy = _impl.Strategy
Truncation = _impl.Truncation

# Numerical kernels
_begin_environment = _impl._begin_environment
_canonicalize = _impl._canonicalize
_contract_last_and_first = _impl._contract_last_and_first
_contract_nrjl_ijk_klm = _impl._contract_nrjl_ijk_klm
_destructive_svd = _impl._destructive_svd
_end_environment = _impl._end_environment
_gemm = _impl._gemm
_join_environments = _impl._join_environments
_left_orth_2site = _impl._left_orth_2site
_right_orth_2site = _impl._right_orth_2site
_recanonicalize = _impl._recanonicalize
_select_svd_driver = _impl._select_svd_driver
_update_in_canonical_form_right = _impl._update_in_canonical_form_right
_update_in_canonical_form_left = _impl._update_in_canonical_form_left
_update_left_environment = _impl._update_left_environment
_update_right_environment = _impl._update_right_environment
destructively_truncate_vector = _impl.destructively_truncate_vector
scprod = _impl.scprod
vdot = _impl.vdot


def active_backend() -> str:
    """Return the active core backend: ``'cython'`` or ``'python'``."""
    return SEEMPS_BACKEND


__all__ = [
    "DEFAULT_STRATEGY",
    "DEFAULT_TOLERANCE",
    "GemmOrder",
    "MAX_BOND_DIMENSION",
    "NO_TRUNCATION",
    "Simplification",
    "Strategy",
    "Truncation",
    "_begin_environment",
    "_canonicalize",
    "_contract_last_and_first",
    "_contract_nrjl_ijk_klm",
    "_destructive_svd",
    "_end_environment",
    "_gemm",
    "_join_environments",
    "_left_orth_2site",
    "_right_orth_2site",
    "_recanonicalize",
    "_select_svd_driver",
    "_update_in_canonical_form_right",
    "_update_in_canonical_form_left",
    "_update_left_environment",
    "_update_right_environment",
    "destructively_truncate_vector",
    "active_backend",
    "scprod",
    "vdot",
]
