from __future__ import annotations

import numpy as np
import scipy.linalg
import warnings

from ..cython import _contract_last_and_first
from ..operators import MPO
from ..operators.quadratic import QuadraticForm
from ..state import DEFAULT_STRATEGY, NO_TRUNCATION, MPS, CanonicalMPS, Strategy
from ..state import random_mps
from ..state.antilinear import AntilinearForm
from ..tools import make_logger
from ..typing import Tensor3
from .dmrg import _SOLVERS, SolverFn


class _Basis:
    """Environments that project `A x` and `b` onto the basis that the MPS `y`
    spans to the left and to the right of its center."""

    Ax: QuadraticForm
    b: AntilinearForm

    def __init__(self, y: CanonicalMPS, A: MPO, x: CanonicalMPS, b: MPS):
        self.Ax = QuadraticForm(A, y, ket=x)
        self.b = AntilinearForm(y, b)

    def move(self, direction: int) -> None:
        """Follow the center of `y` one site in the given `direction`."""
        site = self.Ax.site + direction
        (self.Ax.move_right if direction > 0 else self.Ax.move_left)(site)
        self.b.update(direction)


def _solve_local(basis: _Basis, i: int, solver: SolverFn, atol: float) -> Tensor3:
    """Solve `A x = b` for the tensor of `x` at site `i`, in the basis of `x`."""
    A_i = basis.Ax.one_site_operator(i)
    b_i = basis.b.tensor1site()
    x_i = basis.Ax.ket[i].reshape(-1)
    x_i, _ = solver(A_i, b_i.reshape(-1), x0=x_i, atol=atol, rtol=0)
    return x_i.reshape(b_i.shape)


def _residual(i: int, left: _Basis, right: _Basis) -> Tensor3:
    """Tensor of `b - A x` at site `i`, in the `left` basis for the sites
    before `i` and in the `right` basis for the sites after it."""
    A, x, b = left.Ax.O, left.Ax.ket, left.b.ket
    b_i = _contract_last_and_first(
        left.b.L[i], _contract_last_and_first(b[i], right.b.R[i])
    )
    # L[a,c,b] x[b,j,f] -> Ax[j,f,a,c]
    Ax_i = np.tensordot(x[i], left.Ax.left_env[i], (0, 2))
    # Ax[j,f,a,c] A[c,i,j,d] -> Ax[f,a,i,d]
    Ax_i = np.tensordot(Ax_i, A[i], ([0, 3], [2, 0]))
    # Ax[f,a,i,d] R[e,d,f] -> Ax[a,i,e]
    Ax_i = np.tensordot(Ax_i, right.Ax.right_env[i], ([0, 3], [2, 1]))
    return b_i - Ax_i


def _enrich(x: MPS, i: int, direction: int, extra: Tensor3, max_rank: int) -> None:
    """Add the directions `extra` to the orthonormal basis of `x` at site `i`.
    The state does not change: the neighbor gives zero weight to them."""
    j = i + direction
    U, B = x[i], x[j]
    if direction < 0:
        # Mirror the tensors, so that the bond to enlarge is always the right one.
        U, B, extra = (t.transpose(2, 1, 0) for t in (U, B, extra))
    rank = U.shape[2]
    U = np.concatenate((U, extra[:, :, : max_rank - rank]), axis=2)
    Q, R = scipy.linalg.qr(U.reshape(-1, U.shape[2]), mode="economic")
    U, B = Q.reshape(*U.shape[:2], -1), np.tensordot(R[:, :rank], B, 1)
    if direction < 0:
        U, B = U.transpose(2, 1, 0), B.transpose(2, 1, 0)
    x[i], x[j] = U, B


def _residual_norm(A: MPO, x: MPS, b: MPS) -> float:
    """Norm of `A x - b`, from a canonical form to avoid cancellations."""
    r = (A.apply(x, simplify=False) - b).join()
    return CanonicalMPS(r, strategy=NO_TRUNCATION, normalize=False).norm()


def amen_solve(
    A: MPO,
    b: MPS,
    guess: MPS | None = None,
    maxiter: int = 20,
    atol: float = 0,
    rtol: float = 1e-8,
    strategy: Strategy = DEFAULT_STRATEGY,
    method: str = "cg",
    residual_rank: int = 8,
) -> CanonicalMPS:
    r"""Solve :math:`A x = b` for an MPO `A` and an MPS `b` using the
    alternating minimal energy method (Dolgov and Savostyanov, 2014).

    AMEn sweeps over the sites as a one-site DMRG would, solving at each site
    the equation projected onto the basis that the other tensors of `x` span.
    Such a sweep alone cannot change the bond dimensions of `x`. AMEn also
    keeps an approximation `z` to the residual :math:`b - A x` with a small
    bond dimension, and before leaving a site adds the basis of `z` to that
    of `x`. The next site can then correct `x` along the residual, as in a
    step of steepest descent.

    Parameters
    ----------
    A : MPO
        Linear operator on the left-hand side.
    b : MPS
        Right-hand side.
    guess : MPS | None, default = None
        Initial guess (defaults to `b`).
    maxiter : int, default = 20
        Maximum number of sweeps, each one going right and then left.
    atol, rtol : float
        Convergence requires ``norm(A@x - b) <= max(rtol * norm(b), atol)``.
        Defaults are `rtol=1e-8` and `atol=0`.
    strategy : Strategy, default = DEFAULT_STRATEGY
        Truncation strategy for the bond dimension of the solution.
    method : str, default = 'cg'
        Iterative solver for the local problems: ``'cg'``, ``'bicg'``,
        ``'bicgstab'``, ``'gmres'``, or ``'lgmres'``.
    residual_rank : int, default = 8
        Bond dimension of the residual approximation, which is the number
        of directions that each bond of the solution gains per sweep.

    Returns
    -------
    CanonicalMPS
        The solution `x`.
    """
    if (solver := _SOLVERS.get(method)) is None:
        raise ValueError(f'Unknown solver "{method}"')
    if maxiter < 1 or residual_rank < 1:
        raise ValueError("maxiter and residual_rank must be positive")
    tol = max(atol, rtol * b.norm())
    atol_local = 0.05 * tol / np.sqrt(b.size)
    strategy = strategy.replace(normalize=False)
    max_rank = strategy.get_max_bond_dimension()

    x = CanonicalMPS(b if guess is None else guess, center=0, strategy=strategy)
    is_complex = any(np.iscomplexobj(t) for t in (*A, *b, *x))
    z = CanonicalMPS(
        random_mps(
            b.physical_dimensions(),
            residual_rank,
            complex=is_complex,
            rng=np.random.default_rng(0),
        )
    )
    x_basis, z_basis = _Basis(x, A, x, b), _Basis(z, A, x, b)
    residual = np.inf

    with make_logger() as logger:
        for sweep in range(2 * maxiter):
            direction = +1 if sweep % 2 == 0 else -1
            sites = range(b.size)[::direction]
            for i in sites:
                # 1. Solve A x = b for this site only, as in one-site DMRG.
                x[i] = _solve_local(x_basis, i, solver, atol_local)
                # 2. Keep a low-rank copy `z` of the residual b - A x.
                z[i] = _residual(i, z_basis, z_basis)
                if i == sites[-1]:
                    break
                # 3. Give the next site of `x` the directions of that error.
                if direction > 0:
                    extra = _residual(i, x_basis, z_basis)
                else:
                    extra = _residual(i, z_basis, x_basis)
                x.update_canonical(x[i], direction, strategy)
                _enrich(x, i, direction, extra, max_rank)
                z.update_canonical(z[i], direction, NO_TRUNCATION)
                x_basis.move(direction)
                z_basis.move(direction)

            # The norm of `z` is a cheap lower bound of the true residual.
            estimate = np.linalg.norm(z[sites[-1]])
            if estimate <= tol or sweep == 2 * maxiter - 1:
                residual = _residual_norm(A, x, b)
            logger(f"AMEn {sweep=} D={x.max_bond_dimension()} {estimate=} {residual=}")
            if residual <= tol:
                break
        else:
            warnings.warn(f"AMEn did not converge: {residual} > {tol}", RuntimeWarning)
    return x
