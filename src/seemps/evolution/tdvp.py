from __future__ import annotations

import numpy as np
from scipy.sparse.linalg import expm_multiply
from seemps.state import MPS, CanonicalMPS, Strategy, DEFAULT_STRATEGY
from seemps.cython import _contract_last_and_first
from seemps.optimization.dmrg import (
    QuadraticForm,
    DMRGMatrixOperator,
    OneSiteDMRGOperator,
)
from seemps.operators import MPO
from seemps.evolution.common import ode_solver, ODECallback, TimeSpan


class TDVPForm(QuadraticForm):
    def one_site_Hamiltonian(self, i: int) -> OneSiteDMRGOperator:
        return OneSiteDMRGOperator(  # pyright: ignore[reportCallIssue]
            self.left_env[i],  # pyright: ignore[reportArgumentType]
            self.H[i],  # pyright: ignore[reportArgumentType]
            self.right_env[i],  # pyright: ignore[reportArgumentType]
        )

    def two_site_Hamiltonian(self, i: int) -> DMRGMatrixOperator:
        assert i == self.site
        return DMRGMatrixOperator(  # pyright: ignore[reportCallIssue]
            self.left_env[i],  # pyright: ignore[reportArgumentType]
            _contract_last_and_first(self.H[i], self.H[i + 1]),  # pyright: ignore[reportArgumentType]
            self.right_env[i + 1],  # pyright: ignore[reportArgumentType]
        )


def _evolve(
    operator: OneSiteDMRGOperator | DMRGMatrixOperator,
    tensor: np.ndarray,
    factor: float | complex,
    normalize: bool = False,
) -> np.ndarray:
    """Apply time evolution operator to tensor."""
    shape = tensor.shape
    v = expm_multiply(
        factor * operator, tensor.ravel(), traceA=factor * operator.trace()
    )
    if normalize:
        v = v / np.linalg.norm(v)

    return v.reshape(shape)


def tdvp_step(
    H: MPO, state: MPS, dt: float | complex, strategy: Strategy = DEFAULT_STRATEGY
) -> CanonicalMPS:
    if not isinstance(state, CanonicalMPS):
        state = CanonicalMPS(state, center=0, strategy=strategy)

    QF = TDVPForm(H, state, start=0)
    normalize = strategy.get_normalize_flag()

    # Sweep Right
    for i in range(H.size - 1):
        # Evolve 2-site
        Op2 = QF.two_site_Hamiltonian(i)
        A2 = _contract_last_and_first(QF.state[i], QF.state[i + 1])
        A2 = _evolve(Op2, A2, -0.5 * dt, normalize)
        QF.update_2site_right(A2, i, strategy)

        # Evolve 1-site backward
        if i < H.size - 2:
            Op1 = QF.one_site_Hamiltonian(i + 1)
            QF.state[i + 1] = _evolve(Op1, QF.state[i + 1], 0.5 * dt, normalize)

    # Sweep Left
    for i in range(H.size - 2, -1, -1):
        # Evolve 2-site
        Op2 = QF.two_site_Hamiltonian(i)
        A2 = _contract_last_and_first(QF.state[i], QF.state[i + 1])
        A2 = _evolve(Op2, A2, -0.5 * dt, normalize)
        QF.update_2site_left(A2, i, strategy)

        # Evolve 1-site backward
        if i > 0:
            Op1 = QF.one_site_Hamiltonian(i)
            QF.state[i] = _evolve(Op1, QF.state[i], 0.5 * dt, normalize)

    return QF.state


def tdvp(
    H: MPO,
    time: TimeSpan,
    state: MPS,
    steps: int = 1000,
    strategy: Strategy = DEFAULT_STRATEGY,
    callback: ODECallback | None = None,
    itime: bool = False,
):
    r"""Solve a Schrodinger equation using the Time Dependent Variational Principle
    (TDVP) algorithm.

    Parameters
    ----------
    H : MPO
        Hamiltonian in MPO form.
    time : Real | tuple[Real, Real] | Sequence[Real]
        Integration interval, or sequence of time steps.
    state : MPS
        Initial guess of the ground state.
    steps : int, default = 1000
        Integration steps, if not defined by `t_span`.
    strategy : Strategy, default = DEFAULT_STRATEGY
        Truncation strategy for MPO and MPS algebra.
    callback : Callable[[float, MPS], Any] | None
        A callable called after each iteration (defaults to None).
    itime : bool, default = False
        Whether to solve the imaginary time evolution problem.

    Returns
    -------
    result : MPS | list[Any]
        Final state after evolution or values collected by callback
    """

    def evolve_for_dt(
        t: float, state: MPS, factor: complex | float, dt: float, strategy: Strategy
    ) -> MPS:
        return tdvp_step(H, state, factor * dt, strategy=strategy)

    return ode_solver(evolve_for_dt, time, state, steps, strategy, callback, itime)
