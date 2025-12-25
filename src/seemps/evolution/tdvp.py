import numpy as np
from typing import Any
import scipy.sparse.linalg
from scipy.sparse.linalg import expm_multiply
from seemps.state import MPS, CanonicalMPS, Strategy, DEFAULT_STRATEGY
from seemps.cython import _contract_last_and_first
from seemps.optimization.dmrg import QuadraticForm, DMRGMatrixOperator
from seemps.operators import MPO
from seemps.evolution.common import ode_solver, ODECallback, TimeSpan


class OneSiteTDVPOperator(scipy.sparse.linalg.LinearOperator):
    L: np.ndarray
    H_mpo: np.ndarray
    R: np.ndarray
    v_shape: tuple[int, int, int]

    def __new__(cls, *args: Any, **kwargs: Any):
        return object.__new__(cls)

    def __init__(self, L: np.ndarray, H: np.ndarray, R: np.ndarray):
        self.L = L
        self.H_mpo = H
        self.R = R
        _, _, b = L.shape
        _, _, f = R.shape
        _, _, k, _ = H.shape
        self.v_shape = (b, k, f)

        super().__init__(dtype=L.dtype, shape=(b * k * f, b * k * f))  # type: ignore[call-arg] # pyright: ignore[reportCallIssue]

    def _matvec(self, v: np.ndarray) -> np.ndarray:
        v = v.reshape(self.v_shape)

        aux = np.tensordot(v, self.L, axes=(0, 2))
        aux = np.tensordot(aux, self.H_mpo, axes=([0, 3], [2, 0]))
        aux = np.tensordot(aux, self.R, axes=([0, 3], [2, 1]))

        return aux.reshape(-1)

    def _rmatvec(self, v: np.ndarray) -> np.ndarray:
        a = self.L.shape[0]
        g = self.H_mpo.shape[1]
        d = self.R.shape[0]
        v = v.reshape(a, g, d)

        aux = np.tensordot(v, self.L.conj(), axes=(0, 0))
        aux = np.tensordot(aux, self.H_mpo.conj(), axes=([0, 2], [1, 0]))
        aux = np.tensordot(aux, self.R.conj(), axes=([0, 3], [0, 1]))

        return aux.reshape(-1)

    def trace(self) -> complex:
        l_c = np.trace(self.L, axis1=0, axis2=2)
        w_ce = np.trace(self.H_mpo, axis1=1, axis2=2)
        r_e = np.trace(self.R, axis1=0, axis2=2)
        return np.dot(l_c, np.dot(w_ce, r_e))


class TwoSiteTDVPOperator(DMRGMatrixOperator):
    def __new__(cls, *args: Any, **kwargs: Any):
        return object.__new__(cls)

    def __init__(self, L: np.ndarray, H12: np.ndarray, R: np.ndarray):
        super().__init__(L, H12, R)

    def _rmatvec(self, v: np.ndarray) -> np.ndarray:
        a = self.L.shape[0]
        i = self.H12.shape[1]
        j = self.H12.shape[3]
        e = self.R.shape[0]

        v = v.reshape(a, i, j, e)
        aux = np.tensordot(v, self.L.conj(), axes=(0, 0))
        aux = np.tensordot(aux, self.H12.conj(), axes=([3, 0, 1], [0, 1, 3]))
        aux = np.tensordot(aux, self.R.conj(), axes=([0, 4], [0, 1]))

        return aux.reshape(-1)

    def trace(self) -> complex:
        l_c = np.trace(self.L, axis1=0, axis2=2)
        tmp = np.trace(self.H12, axis1=1, axis2=2)
        w_ce = np.trace(tmp, axis1=1, axis2=2)
        r_e = np.trace(self.R, axis1=0, axis2=2)
        return np.dot(l_c, np.dot(w_ce, r_e))


class TDVPForm(QuadraticForm):
    def one_site_Hamiltonian(self, i: int) -> OneSiteTDVPOperator:
        return OneSiteTDVPOperator(self.left_env[i], self.H[i], self.right_env[i])

    def two_site_Hamiltonian(self, i: int) -> TwoSiteTDVPOperator:
        assert i == self.site
        return TwoSiteTDVPOperator(
            self.left_env[i],
            _contract_last_and_first(self.H[i], self.H[i + 1]),
            self.right_env[i + 1],
        )


def _evolve(
    operator: OneSiteTDVPOperator | TwoSiteTDVPOperator,
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
