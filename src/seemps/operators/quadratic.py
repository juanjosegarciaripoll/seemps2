from __future__ import annotations

from typing import Callable, Any
import numpy as np
from scipy.sparse.linalg import LinearOperator

from .mpo import MPO
from ..typing import Tensor3, Tensor4, Weight
from ..state import MPS, CanonicalMPS, Strategy
from ..cython import _contract_last_and_first
from ..state.environments import (
    MPOEnvironment,
    begin_mpo_environment,
    update_left_mpo_environment,
    update_right_mpo_environment,
)


def _make_operator(
    shape: tuple[int, int],
    dtype: Any,
    matvec: Callable[[np.ndarray], np.ndarray],
    rmatvec: Callable[[np.ndarray], np.ndarray],
    trace: Callable[[], Weight],
) -> LinearOperator:
    op = LinearOperator(shape=shape, dtype=dtype, matvec=matvec, rmatvec=rmatvec)
    setattr(op, "trace", trace)
    return op


class QuadraticForm:
    O: MPO
    state: CanonicalMPS
    ket: MPS
    size: int
    left_env: list[MPOEnvironment]
    right_env: list[MPOEnvironment]
    site: int

    def __init__(
        self, O: MPO, state: CanonicalMPS, start: int = 0, ket: MPS | None = None
    ):
        # Initialize self.O, self.state and self.ket by reference
        self.O = O
        if not isinstance(state, CanonicalMPS):
            raise Exception("QuadraticForm: state must be a CanonicalMPS")
        if state.center not in [start, start + 1]:
            raise Exception(
                f"QuadraticForm: state must be centered at start={start} or {start + 1}"
            )
        self.state = state
        self.ket = ket if ket is not None else state
        self.size = size = state.size
        if size != O.size:
            raise Exception("QuadraticForm: MPO and MPS do not have the same size")
        if any(Op.shape[1] != A.shape[1] for Op, A in zip(O, state)):
            raise Exception("QuadraticForm: MPO and MPS dimensions do not match")

        left_env: list[MPOEnvironment] = [begin_mpo_environment()] * size
        right_env: list[MPOEnvironment] = left_env.copy()

        # Build right environments
        env = right_env[-1]
        for i in range(size - 1, max(0, start - 1), -1):
            right_env[i - 1] = env = update_right_mpo_environment(
                env, state[i], O[i], self.ket[i]
            )

        # Build left environments
        env = left_env[0]
        for i in range(0, start):
            left_env[i + 1] = env = update_left_mpo_environment(
                env, state[i], O[i], self.ket[i]
            )

        self.left_env = left_env
        self.right_env = right_env
        self.site = start

    # Move optimization center and update environments
    def move_right(self, i: int):
        j = self.site
        self.left_env[i] = update_left_mpo_environment(
            self.left_env[j], self.state[j], self.O[j], self.ket[j]
        )
        self.site = i

    def move_left(self, i: int):
        j = self.site
        self.right_env[i] = update_right_mpo_environment(
            self.right_env[j], self.state[j], self.O[j], self.ket[j]
        )
        self.site = i

    # Effective 0/1/2-site blocks
    def zero_site_block(self, i: int):
        return self.left_env[i], self.right_env[i - 1]

    def one_site_block(self, i: int):
        return self.left_env[i], self.O[i], self.right_env[i]

    def two_site_block(self, i: int):
        H12 = _contract_last_and_first(self.O[i], self.O[i + 1])
        return self.left_env[i], H12, self.right_env[i + 1]

    # Effective 0-site operator
    def zero_site_operator(self, i: int) -> LinearOperator:
        L, R = self.zero_site_block(i)
        b = L.shape[2]
        f = R.shape[2]
        v_shape = (b, f)
        n = b * f
        dtype = np.result_type(L.dtype, R.dtype)

        def _matvec(
            v: np.ndarray,
            _L: np.ndarray = L,
            _R: np.ndarray = R,
            _vs: tuple[int, ...] = v_shape,
        ) -> np.ndarray:
            v = v.reshape(_vs)
            aux = np.tensordot(v, _L, axes=(0, 2))
            return np.tensordot(aux, _R, axes=([0, 2], [2, 1])).reshape(-1)

        def _rmatvec(
            v: np.ndarray,
            _L: np.ndarray = L,
            _R: np.ndarray = R,
            _vs: tuple[int, ...] = v_shape,
        ) -> np.ndarray:
            v = v.reshape(_vs)
            aux = np.tensordot(v, _L.conj(), axes=(0, 0))
            return np.tensordot(aux, _R.conj(), axes=([0, 1], [0, 1])).reshape(-1)

        def _trace(_L: np.ndarray = L, _R: np.ndarray = R) -> Weight:
            l_c = np.trace(_L, axis1=0, axis2=2)
            r_e = np.trace(_R, axis1=0, axis2=2)
            return np.vdot(l_c, r_e)

        return _make_operator((n, n), dtype, _matvec, _rmatvec, _trace)

    # Effective 1-site operator
    def one_site_operator(self, i: int) -> LinearOperator:
        L, H, R = self.one_site_block(i)
        b = L.shape[2]
        s = H.shape[2]
        f = R.shape[2]
        v_shape = (b, s, f)
        n = b * s * f
        dtype = np.result_type(L.dtype, R.dtype, H.dtype)

        def _matvec(
            v: np.ndarray,
            _L: np.ndarray = L,
            _H: np.ndarray = H,
            _R: np.ndarray = R,
            _vs: tuple[int, ...] = v_shape,
        ) -> np.ndarray:
            v = v.reshape(_vs)
            aux = np.tensordot(v, _L, axes=(0, 2))
            aux = np.tensordot(aux, _H, axes=([0, 3], [2, 0]))
            return np.tensordot(aux, _R, axes=([0, 3], [2, 1])).reshape(-1)

        def _rmatvec(
            v: np.ndarray,
            _L: np.ndarray = L,
            _H: np.ndarray = H,
            _R: np.ndarray = R,
            _vs: tuple[int, ...] = v_shape,
        ) -> np.ndarray:
            v = v.reshape(_vs)
            aux = np.tensordot(v, _L.conj(), axes=(0, 0))
            aux = np.tensordot(aux, _H.conj(), axes=([0, 2], [1, 0]))
            return np.tensordot(aux, _R.conj(), axes=([0, 3], [0, 1])).reshape(-1)

        def _trace(
            _L: np.ndarray = L, _H: np.ndarray = H, _R: np.ndarray = R
        ) -> Weight:
            l_c = np.trace(_L, axis1=0, axis2=2)
            w_ce = np.trace(_H, axis1=1, axis2=2)
            r_e = np.trace(_R, axis1=0, axis2=2)
            return np.dot(l_c, np.dot(w_ce, r_e))

        return _make_operator((n, n), dtype, _matvec, _rmatvec, _trace)

    # Effective 2-site operator
    def two_site_operator(self, i: int) -> LinearOperator:
        L, H, R = self.two_site_block(i)
        k = H.shape[2]
        l = H.shape[4]
        b = L.shape[2]
        f = R.shape[2]
        v_shape = (b, k, l, f)
        n = b * k * l * f
        dtype = np.result_type(L.dtype, R.dtype, H.dtype)

        def _matvec(
            v: np.ndarray,
            _L: np.ndarray = L,
            _H: np.ndarray = H,
            _R: np.ndarray = R,
            _vs: tuple[int, ...] = v_shape,
        ) -> np.ndarray:
            v = v.reshape(_vs)
            aux = np.tensordot(v, _L, ((0,), (2,)))
            aux = np.tensordot(aux, _H, ((0, 1, 4), (2, 4, 0)))
            return np.tensordot(aux, _R, ((0, 4), (2, 1))).reshape(-1)

        def _rmatvec(
            v: np.ndarray,
            _L: np.ndarray = L,
            _H: np.ndarray = H,
            _R: np.ndarray = R,
            _vs: tuple[int, ...] = v_shape,
        ) -> np.ndarray:
            v = v.reshape(_vs)
            aux = np.tensordot(v, _L.conj(), axes=(0, 0))
            aux = np.tensordot(aux, _H.conj(), axes=([3, 0, 1], [0, 1, 3]))
            return np.tensordot(aux, _R.conj(), axes=([0, 4], [0, 1])).reshape(-1)

        def _trace(
            _L: np.ndarray = L, _H: np.ndarray = H, _R: np.ndarray = R
        ) -> Weight:
            l_c = np.trace(_L, axis1=0, axis2=2)
            tmp = np.trace(_H, axis1=1, axis2=2)
            w_ce = np.trace(tmp, axis1=1, axis2=2)
            r_e = np.trace(_R, axis1=0, axis2=2)
            return np.dot(l_c, np.dot(w_ce, r_e))

        return _make_operator((n, n), dtype, _matvec, _rmatvec, _trace)

    def gradient_1site(self) -> Tensor3:
        """Return the gradient tensor d<state|O|ket>/dstate* at the current
        center site."""
        c = self.site
        L = self.left_env[c]  # (a, c, b)
        Oc = self.O[c]  # (c, j, i, e)
        A = self.ket[c]  # (b, i, f)
        R = self.right_env[c]  # (d, e, f)

        a, c_, b = L.shape
        _, j, i, e = Oc.shape
        _, _, f = A.shape
        d = R.shape[0]

        # L(a,c,b) * Oc(c,j,i,e) -> aux(a,b,j,i,e)
        aux = np.matmul(
            L.transpose(0, 2, 1).reshape(a * b, c_),
            Oc.reshape(c_, j * i * e),
        ).reshape(a, b, j, i, e)

        # aux(a,b,j,i,e) * A(b,i,f) -> aux(a,j,e,f)
        aux = np.matmul(
            aux.transpose(0, 2, 4, 1, 3).reshape(a * j * e, b * i),
            A.reshape(b * i, f),
        ).reshape(a, j, e, f)

        # aux(a,j,e,f) * R(d,e,f) -> result(a,j,d)
        result = np.matmul(
            aux.reshape(a * j, e * f),
            R.reshape(d, e * f).T,
        ).reshape(a, j, d)

        return result

    def gradient_2site(self, direction: int) -> Tensor4:
        """Return the gradient tensor for two sites.

        Parameters
        ----------
        direction : {+1, -1}
            If positive, acts on (site, site+1).
            Otherwise on (site-1, site).
        """
        if direction > 0:
            i = self.site
            j = i + 1
        else:
            j = self.site
            i = j - 1

        L = self.left_env[i]  # (a, c, b)
        Oi = self.O[i]  # (c, j, i, e)
        A = self.ket[i]  # (b, i, f)
        Oj = self.O[j]  # (e, l, k, d)
        B = self.ket[j]  # (f, k, g)
        R = self.right_env[j]  # (h, d, g)
        a, c_, b = L.shape
        _, j_, i_, e = Oi.shape
        _, _, f = A.shape
        _, l_, k_, d_ = Oj.shape
        _, _, g = B.shape
        h = R.shape[0]
        # L(a,c,b) * Oi(c,j,i,e) -> aux(a,b,j,i,e)
        aux = np.matmul(
            L.transpose(0, 2, 1).reshape(a * b, c_),
            Oi.reshape(c_, j_ * i_ * e),
        ).reshape(a, b, j_, i_, e)
        # aux(a,b,j,i,e) * A(b,i,f) -> aux(a,j,e,f)
        aux = np.matmul(
            aux.transpose(0, 2, 4, 1, 3).reshape(a * j_ * e, b * i_),
            A.reshape(b * i_, f),
        ).reshape(a, j_, e, f)
        # Oj(e,l,k,d) * B(f,k,g) -> aux2(e,l,d,f,g)
        aux2 = np.matmul(
            Oj.transpose(0, 1, 3, 2).reshape(e * l_ * d_, k_),
            B.transpose(1, 0, 2).reshape(k_, f * g),
        ).reshape(e, l_, d_, f, g)
        # aux2(e,l,d,f,g) * R(h,d,g) -> aux2(e,l,f,h)
        aux2 = np.matmul(
            aux2.transpose(0, 1, 3, 2, 4).reshape(e * l_ * f, d_ * g),
            R.reshape(h, d_ * g).T,
        ).reshape(e, l_, f, h)
        # aux(a,j,e,f) * aux2(e,l,f,h) -> result(a,j,l,h)
        return np.matmul(
            aux.transpose(0, 1, 3, 2).reshape(a * j_, f * e),
            aux2.transpose(2, 0, 1, 3).reshape(f * e, l_ * h),
        ).reshape(a, j_, l_, h)

    def ket_2site(self, i: int):
        A = self.ket[i]
        B = self.ket[i + 1]
        return _contract_last_and_first(A, B)

    # Update sites and the corresponding environment
    def update_2site_right(self, AB: Tensor4, i: int, strategy: Strategy) -> None:
        self.state.update_2site_right(AB, i, strategy)
        if i < self.size - 2:
            self.site = j = i + 1
            self.left_env[j] = update_left_mpo_environment(
                self.left_env[i], self.state[i], self.O[i], self.ket[i]
            )

    def update_2site_left(self, AB: Tensor4, i: int, strategy: Strategy) -> None:
        self.state.update_2site_left(AB, i, strategy)
        if i > 0:
            j = i + 1
            self.right_env[i] = update_right_mpo_environment(
                self.right_env[j], self.state[j], self.O[j], self.ket[j]
            )
            self.site = i - 1
