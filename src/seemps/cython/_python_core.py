"""Pure NumPy/SciPy reference implementations of the SeeMPS core kernels.

This module is a complete, self-contained reimplementation of everything the
compiled Cython ``core`` module exports: configuration types (Strategy,
Truncation, Simplification, GemmOrder, constants) and all numerical kernels
(contractions, environments, gemm, svd, schmidt, truncation).

Set ``SEEMPS_BACKEND=python`` before importing SeeMPS to route
``seemps.cython`` entirely through this module instead of ``core``.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import svd as _scipy_svd
from typing import TYPE_CHECKING, Literal, final

if TYPE_CHECKING:
    from ..state import MPS

__version__ = "python-contractions"

# Configuration types and constants (mirror truncation.pxi)

MAX_BOND_DIMENSION: int = 0x7FFFFFFF


@final
class Truncation:
    """SVD truncation algorithm when splitting tensors."""

    DO_NOT_TRUNCATE = 0
    RELATIVE_SINGULAR_VALUE = 1
    RELATIVE_NORM_SQUARED_ERROR = 2
    ABSOLUTE_SINGULAR_VALUE = 3


@final
class Simplification:
    """Tensor network simplification algorithms."""

    DO_NOT_SIMPLIFY = 0
    CANONICAL_FORM = 1
    VARIATIONAL = 2
    VARIATIONAL_EXACT_GUESS = 3


@final
class GemmOrder:
    """Matrix transpose/adjoint flags for _gemm."""

    NORMAL = 0
    TRANSPOSE = 1
    ADJOINT = 2


DEFAULT_TOLERANCE: float = float(np.finfo(np.float64).eps)


@final
class Strategy:
    """MPS and MPO simplification strategies."""

    __slots__ = (
        "_method",
        "_tolerance",
        "_simplification_tolerance",
        "_max_bond_dimension",
        "_normalize",
        "_simplify",
        "_max_sweeps",
    )

    def __init__(
        self,
        method: int = Truncation.RELATIVE_NORM_SQUARED_ERROR,
        tolerance: float = DEFAULT_TOLERANCE,
        simplification_tolerance: float = DEFAULT_TOLERANCE,
        max_bond_dimension: int = MAX_BOND_DIMENSION,
        normalize: bool = False,
        simplify: int = Simplification.VARIATIONAL,
        max_sweeps: int = 16,
    ):
        if tolerance < 0 or tolerance >= 1.0:
            raise AssertionError("Invalid tolerance argument passed to Strategy")
        if tolerance == 0 and method != Truncation.DO_NOT_TRUNCATE:
            method = Truncation.ABSOLUTE_SINGULAR_VALUE
        if max_bond_dimension <= 0 or max_bond_dimension > MAX_BOND_DIMENSION:
            raise AssertionError("Invalid bond dimension in Strategy")
        if max_sweeps < 0:
            raise AssertionError("Negative or zero number of sweeps in Strategy")
        self._method = int(method)
        self._tolerance = float(tolerance)
        self._simplification_tolerance = float(simplification_tolerance)
        self._max_bond_dimension = int(max_bond_dimension)
        self._normalize = bool(normalize)
        self._simplify = int(simplify)
        self._max_sweeps = int(max_sweeps)

    def replace(
        self,
        method: int | None = None,
        tolerance: float | None = None,
        simplification_tolerance: float | None = None,
        max_bond_dimension: int | None = None,
        normalize: bool | None = None,
        simplify: int | None = None,
        max_sweeps: int | None = None,
    ) -> Strategy:
        return Strategy(
            method=self._method if method is None else method,
            tolerance=self._tolerance if tolerance is None else tolerance,
            simplification_tolerance=(
                self._simplification_tolerance
                if simplification_tolerance is None
                else simplification_tolerance
            ),
            max_bond_dimension=(
                self._max_bond_dimension
                if max_bond_dimension is None
                else max_bond_dimension
            ),
            normalize=self._normalize if normalize is None else normalize,
            simplify=self._simplify if simplify is None else simplify,
            max_sweeps=self._max_sweeps if max_sweeps is None else max_sweeps,
        )

    def set_normalization(self, normalize: bool) -> Strategy:
        return self.replace(normalize=normalize)

    def get_method(self) -> int:
        return self._method

    def get_simplification_method(self) -> int:
        return self._simplify

    def get_tolerance(self) -> float:
        return self._tolerance

    def get_simplification_tolerance(self) -> float:
        return self._simplification_tolerance

    def get_max_bond_dimension(self) -> int:
        return self._max_bond_dimension

    def get_normalize_flag(self) -> bool:
        return self._normalize

    def get_max_sweeps(self) -> int:
        return self._max_sweeps

    def get_simplify_flag(self) -> bool:
        return self._simplify != Simplification.DO_NOT_SIMPLIFY

    def __str__(self) -> str:
        _m = {0: "None", 1: "RelativeSVD", 2: "RelativeNorm", 3: "AbsoluteSVD"}
        _s = {
            0: "None",
            1: "CanonicalForm",
            2: "Variational",
            3: "Variational (exact guess)",
        }
        return (
            f"Strategy(method={_m[self._method]}, tolerance={self._tolerance:5g}, "
            f"max_bond_dimension={self._max_bond_dimension}, normalize={self._normalize}, "
            f"simplify={_s[self._simplify]}, "
            f"simplification_tolerance={self._simplification_tolerance:5g}, "
            f"max_sweeps={self._max_sweeps})"
        )


DEFAULT_STRATEGY = Strategy(
    method=Truncation.RELATIVE_NORM_SQUARED_ERROR,
    simplify=Simplification.VARIATIONAL,
    tolerance=DEFAULT_TOLERANCE,
    simplification_tolerance=DEFAULT_TOLERANCE,
    max_bond_dimension=MAX_BOND_DIMENSION,
    normalize=False,
)

NO_TRUNCATION = DEFAULT_STRATEGY.replace(
    method=Truncation.DO_NOT_TRUNCATE,
    simplify=Simplification.DO_NOT_SIMPLIFY,
)

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
    "scprod",
    "vdot",
]


# Contractions (mirror contractions.pxi)


def _contract_last_and_first(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Contract last index of ``A`` and first from ``B``."""
    if not isinstance(A, np.ndarray) or not isinstance(B, np.ndarray):
        raise ValueError("_contract_last_and_first expects tensors")
    return np.tensordot(A, B, axes=1)


def _contract_nrjl_ijk_klm(
    U: np.ndarray, A: np.ndarray, B: np.ndarray
) -> np.ndarray:
    """Implements ``np.einsum('ijk,klm,nrjl->inrm', A, B, U)`` with ``U`` given
    as the matrix ``U[n*r, j*l]`` and the new physical dimensions equal to the
    old ones (``n == j``, ``r == l``)."""
    if (
        not isinstance(A, np.ndarray)
        or not isinstance(B, np.ndarray)
        or not isinstance(U, np.ndarray)
        or A.ndim != 3
        or B.ndim != 3
        or U.ndim != 2
    ):
        raise ValueError("Invalid arguments to _contract_nrjl_ijk_klm")
    a, d, b = A.shape
    _, e, c = B.shape
    AB = (A.reshape(a * d, b) @ B.reshape(b, e * c)).reshape(a, d * e, c)
    return (U @ AB).reshape(a, d, e, c)


# Environments (mirror environments.pxi)

_EMPTY_ENVIRONMENT = np.eye(1)


def _begin_environment(D: int = 1) -> np.ndarray:
    """Initiate the computation of a left environment from two MPS."""
    if D == 1:
        return _EMPTY_ENVIRONMENT
    return np.eye(D)


def _update_left_environment(
    B: np.ndarray, A: np.ndarray, rho: np.ndarray
) -> np.ndarray:
    """Extend the left environment with bra tensor ``B`` and ket tensor ``A``."""
    if (
        not isinstance(A, np.ndarray)
        or not isinstance(B, np.ndarray)
        or not isinstance(rho, np.ndarray)
        or A.ndim != 3
        or B.ndim != 3
        or rho.ndim != 2
    ):
        raise ValueError("Invalid arguments to _update_left_environment")
    return np.einsum("ljn,li,ijk->nk", B.conj(), rho, A)


def _update_right_environment(
    B: np.ndarray, A: np.ndarray, rho: np.ndarray
) -> np.ndarray:
    """Extend the right environment with bra tensor ``B`` and ket tensor ``A``."""
    if (
        not isinstance(A, np.ndarray)
        or not isinstance(B, np.ndarray)
        or not isinstance(rho, np.ndarray)
        or A.ndim != 3
        or B.ndim != 3
        or rho.ndim != 2
    ):
        raise ValueError("Invalid arguments to _update_right_environment")
    return np.einsum("ijk,kn,ljn->il", A, rho, B.conj())


def _end_environment(rho: np.ndarray) -> complex:
    """Extract the scalar product from the last environment."""
    return rho.flat[0]


def _join_environments(rhoL: np.ndarray, rhoR: np.ndarray) -> complex:
    """Join left and right environments to produce a scalar."""
    if rhoL.shape[0] == 1:
        return _end_environment(rhoL) * _end_environment(rhoR)
    return np.dot(rhoL.ravel(order="C"), rhoR.swapaxes(0, 1).ravel(order="C"))


def scprod(bra: MPS, ket: MPS) -> complex:
    """Compute the scalar product ``<bra|ket>`` between two MPS."""
    A = bra._data  # type: ignore[attr-defined]
    B = ket._data  # type: ignore[attr-defined]
    if len(A) != len(B):
        raise ValueError("Invalid arguments to scprod")
    rho = _EMPTY_ENVIRONMENT
    for Ai, Bi in zip(A, B):
        rho = _update_left_environment(Ai, Bi, rho)
    return _end_environment(rho)


def vdot(bra: MPS, ket: MPS) -> complex:
    """Alias for :func:`scprod`."""
    return scprod(bra, ket)


# ---------------------------------------------------------------------------
# GEMM (mirror gemm.pxi)
# ---------------------------------------------------------------------------


def _gemm_op(M: np.ndarray, flag: int) -> np.ndarray:
    if flag == 0:
        return M
    if flag == 1:
        return M.T
    if flag == 2:
        return M.conj().T
    raise ValueError(f"Invalid GEMM order flag {flag}")


def _gemm(B: np.ndarray, BT: int, A: np.ndarray, AT: int) -> np.ndarray:
    """Return ``op(B, BT) @ op(A, AT)``."""
    if (
        not isinstance(A, np.ndarray)
        or not isinstance(B, np.ndarray)
        or A.ndim != 2
        or B.ndim != 2
    ):
        raise ValueError()
    return _gemm_op(B, BT) @ _gemm_op(A, AT)


# SVD (mirror svd.pxi)

_svd_driver: Literal["gesdd", "gesvd"] = "gesdd"


def _select_svd_driver(name: str) -> None:
    global _svd_driver
    if name in ("gesvd", "gesdd"):
        _svd_driver = name  # type: ignore[assignment]
    else:
        raise ValueError(f"Invalid LAPACK SVD driver name: {name}")


def _destructive_svd(A: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Economy SVD ``A = U @ diag(s) @ Vh`` (same return order as numpy)."""
    if not isinstance(A, np.ndarray) or A.ndim != 2:
        raise ValueError("Invalid argument to SVD")
    if A.dtype == np.complex64:
        A = A.astype(np.complex128)
    elif A.dtype != np.float64 and A.dtype != np.complex128:
        A = A.astype(np.float64)
    return _scipy_svd(
        A,
        full_matrices=False,
        overwrite_a=True,
        check_finite=False,
        lapack_driver=_svd_driver,
    )


# Truncation (mirror truncation.pxi)


def _truncate(s: np.ndarray, strategy: Strategy) -> tuple[int, float]:
    """Return ``(final_size, squared_error)`` for truncating singular values ``s``."""
    method = strategy.get_method()
    N = s.shape[0]
    if method == Truncation.DO_NOT_TRUNCATE:
        return N, 0.0

    max_bond = strategy.get_max_bond_dimension()
    tol = strategy.get_tolerance()

    if method == Truncation.RELATIVE_NORM_SQUARED_ERROR:
        errors = np.empty(N + 1)
        errors[0] = 0.0
        np.cumsum((s * s)[::-1], out=errors[1:])
        total = errors[N]
        max_error = float(total * tol)
        idx = int(np.searchsorted(errors, max_error, side="right"))
        drop = min(idx - 1, N - 1)
        final_size = min(N - drop, max_bond)
        return final_size, float(errors[N - final_size])

    if method == Truncation.RELATIVE_SINGULAR_VALUE:
        max_error = float(tol * s[0])
    elif method == Truncation.ABSOLUTE_SINGULAR_VALUE:
        max_error = tol
    else:
        raise AssertionError("Invalid truncation method")

    final_size = min(N, max_bond)
    for i in range(1, final_size):
        if s[i] <= max_error:
            final_size = i
            break
    dropped = s[final_size:]
    return final_size, float(np.dot(dropped, dropped))


def destructively_truncate_vector(s: np.ndarray, strategy: Strategy) -> float:
    """Truncate ``s`` in place and return the squared truncation error."""
    final_size, error = _truncate(s, strategy)
    if final_size != s.shape[0]:
        s.resize(final_size, refcheck=False)
    return error


# Schmidt / canonical form (mirror schmidt.pxi)


def __update_in_canonical_form_right(
    state: list[np.ndarray], someA: np.ndarray, site: int, truncation: Strategy
) -> float:
    A = np.ascontiguousarray(someA).copy()
    a, i, b = A.shape
    U, s, V = _destructive_svd(A.reshape(a * i, b))
    D, error = _truncate(s, truncation)
    s, U, V = s[:D], U[:, :D], V[:D, :]
    state[site] = U.reshape(a, i, D)
    site += 1
    state[site] = _contract_last_and_first(s.reshape(D, 1) * V, state[site])
    return np.sqrt(error)


def _update_in_canonical_form_right(
    state: list[np.ndarray], A: np.ndarray, site: int, truncation: Strategy
) -> tuple[int, float]:
    """Insert tensor ``A`` in canonical form, updating the right neighbor."""
    if site + 1 == len(state):
        state[site] = A
        return site, 0.0
    return site + 1, __update_in_canonical_form_right(state, A, site, truncation)


def __update_in_canonical_form_left(
    state: list[np.ndarray], someA: np.ndarray, site: int, truncation: Strategy
) -> float:
    A = np.ascontiguousarray(someA).copy()
    a, i, b = A.shape
    U, s, V = _destructive_svd(A.reshape(a, i * b))
    D, error = _truncate(s, truncation)
    s, U, V = s[:D], U[:, :D], V[:D, :]
    state[site] = V.reshape(D, i, b)
    site -= 1
    state[site] = _contract_last_and_first(state[site], U * s)
    return np.sqrt(error)


def _update_in_canonical_form_left(
    state: list[np.ndarray], A: np.ndarray, site: int, truncation: Strategy
) -> tuple[int, float]:
    """Insert tensor ``A`` in canonical form, updating the left neighbor."""
    if site == 0:
        state[0] = A
        return 0, 0.0
    return site - 1, __update_in_canonical_form_left(state, A, site, truncation)


def _recanonicalize(
    state: list[np.ndarray], oldcenter: int, newcenter: int, truncation: Strategy
) -> float:
    err = 0.0
    while oldcenter > newcenter:
        err += __update_in_canonical_form_left(
            state, state[oldcenter], oldcenter, truncation
        )
        oldcenter -= 1
    while oldcenter < newcenter:
        err += __update_in_canonical_form_right(
            state, state[oldcenter], oldcenter, truncation
        )
        oldcenter += 1
    return err


def _canonicalize(
    state: list[np.ndarray], center: int, truncation: Strategy
) -> float:
    """Bring ``state`` (a list of rank-3 tensors) into canonical form."""
    L = len(state)
    err = 0.0
    for i in range(0, center):
        err += __update_in_canonical_form_right(state, state[i], i, truncation)
    for i in range(L - 1, center, -1):
        err += __update_in_canonical_form_left(state, state[i], i, truncation)
    return err


def _left_orth_2site(
    AA: np.ndarray, strategy: Strategy
) -> tuple[np.ndarray, np.ndarray, float]:
    """Split ``AA[a,b,c,d]`` into a left-isometry ``B[a,b,r]`` and ``C[r,c,d]``."""
    a, d1, d2, b = AA.shape
    U, s, V = _destructive_svd(AA.reshape(a * d1, d2 * b))
    D, error = _truncate(s, strategy)
    s, U, V = s[:D], U[:, :D], V[:D, :]
    return (
        U.reshape(a, d1, D),
        (s.reshape(D, 1) * V).reshape(D, d2, b),
        np.sqrt(error),
    )


def _right_orth_2site(
    AA: np.ndarray, strategy: Strategy
) -> tuple[np.ndarray, np.ndarray, float]:
    """Split ``AA[a,b,c,d]`` into ``B[a,b,r]`` and a right-isometry ``C[r,c,d]``."""
    a, d1, d2, b = AA.shape
    U, s, V = _destructive_svd(AA.reshape(a * d1, d2 * b))
    D, error = _truncate(s, strategy)
    s, U, V = s[:D], U[:, :D], V[:D, :]
    return (
        (U * s).reshape(a, d1, D),
        V.reshape(D, d2, b),
        np.sqrt(error),
    )
