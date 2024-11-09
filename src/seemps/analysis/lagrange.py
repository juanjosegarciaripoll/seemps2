import numpy as np
from scipy.sparse import dok_matrix, csc_array  # type: ignore
from typing import Callable, Optional
from functools import lru_cache

from ..state import MPS, Strategy, DEFAULT_STRATEGY
from ..state.schmidt import _destructive_svd
from ..state._contractions import _contract_last_and_first
from ..state.core import destructively_truncate_vector
from ..truncate import simplify
from .mesh import array_affine


# TODO: Implement multivariate Lagrange interpolation and multirresolution constructions


def lagrange_basic(
    func: Callable,
    order: int,
    sites: int,
    start: float = -1.0,
    stop: float = 1.0,
    strategy: Strategy = DEFAULT_STRATEGY,
    use_logs: bool = True,
) -> MPS:
    """
    Performs a basic Lagrange MPS Chebyshev interpolation of a function.

    Parameters
    ----------
    func : Callable
        The function to interpolate.
    order : int
        The order of the Chebyshev interpolation.
    sites : int
        The number of qubits of the MPS.
    start : float, default=-1.0
        The starting point of the function's domain.
    stop : float, default=1.0
        The end point of the function's domain.
    strategy : Strategy, default=DEFAULT_STRATEGY
        The MPS simplification strategy.
    use_logs : bool, default=True
        Whether to compute the Chebyshev cardinal function using
        logarithms to avoid overflow.

    Returns
    -------
    mps : MPS
        The MPS corresponding to the naive Chebyshev interpolation.
    """
    builder = LagrangeBuilder(order)
    Al = builder.A_L(func, start, stop)
    Ac = builder.A_C(use_logs)
    Ar = builder.A_R(use_logs)

    mps = MPS([Al] + [Ac for _ in range(sites - 2)] + [Ar])
    return simplify(mps, strategy=strategy)


def lagrange_rank_revealing(
    func: Callable,
    order: int,
    sites: int,
    start: float = -1.0,
    stop: float = 1.0,
    strategy: Strategy = DEFAULT_STRATEGY,
    use_logs: bool = True,
) -> MPS:
    """
    Performs a Lagrange rank-revealing MPS Chebyshev interpolation of a function.

    Parameters
    ----------
    func : Callable
        The function to interpolate.
    order : int
        The order of the Chebyshev interpolation.
    sites : int
        The number of qubits of the MPS.
    start : float, default=-1.0
        The starting point of the function's domain.
    stop : float, default=1.0
        The end point of the function's domain.
    strategy : Strategy, default=DEFAULT_STRATEGY
        The MPS simplification strategy.
    use_logs : bool, default=True
        Whether to compute the Chebyshev cardinal function using
        logarithms to avoid overflow.

    Returns
    -------
    mps : MPS
        The MPS corresponding to the rank-revealing Chebyshev interpolation.
    """
    builder = LagrangeBuilder(order)
    Al = builder.A_L(func, start, stop)
    Ac = builder.A_C(use_logs)
    Ar = builder.A_R(use_logs)

    U_L, R = np.linalg.qr(Al.reshape((2, order + 1)))
    tensors = [U_L.reshape(1, 2, 2)]
    for _ in range(sites - 2):
        B = _contract_last_and_first(R, Ac)
        r1, s, r2 = B.shape
        ## SVD
        U, S, V = _destructive_svd(B.reshape(r1 * s, r2))
        destructively_truncate_vector(S, strategy)
        D = S.size
        U = U[:, :D]
        R = S.reshape(D, 1) * V[:D, :]
        ##
        tensors.append(U.reshape(r1, s, -1))
    U_R = _contract_last_and_first(R, Ar)
    tensors.append(U_R)
    return MPS(tensors)


def lagrange_local_rank_revealing(
    func: Callable,
    order: int,
    local_order: int,
    sites: int,
    start: float = -1.0,
    stop: float = 1.0,
    strategy: Strategy = DEFAULT_STRATEGY,
) -> MPS:
    """
    Performs a local rank-revealing Lagrange MPS Chebyshev interpolation of a function.
    The intermediate tensors are now sparse, with a number of non-zero elements that
    is proportional to `local_order`, increasing the efficiency of the interpolation.

    Parameters
    ----------
    func : Callable
        The function to interpolate.
    order : int
        The order of the Chebyshev interpolation.
    local_order : int
        The local order of the Chebyshev interpolation.
    sites : int
        The number of qubits of the MPS.
    start : float, default=-1.0
        The starting point of the function's domain.
    stop : float, default=1.0
        The end point of the function's domain.
    strategy : Strategy, default=DEFAULT_STRATEGY
        The MPS simplification strategy.

    Returns
    -------
    mps : MPS
        The MPS corresponding to the local rank-revealing Chebyshev interpolation.
    """
    # TODO: Optimize matrix multiplications and SVD considering sparsity
    builder = LagrangeBuilder(order, local_order)
    Al = builder.A_L(func, start, stop)
    Ac = builder.A_C_sparse()
    Ar = builder.A_R_sparse()

    U_L, R = np.linalg.qr(Al.reshape((2, order + 1)))
    tensors = [U_L.reshape(1, 2, 2)]
    for _ in range(sites - 2):
        B = R @ Ac
        r1 = B.shape[0]
        ## SVD
        U, S, V = _destructive_svd(B.reshape(r1 * 2, order + 1))
        destructively_truncate_vector(S, strategy)
        D = S.size
        U = U[:, :D]
        R = S.reshape(D, 1) * V[:D, :]
        ##
        tensors.append(U.reshape(r1, 2, -1))
    U_R = R @ Ar
    tensors.append(U_R.reshape(-1, 2, 1))
    return MPS(tensors)


class LagrangeBuilder:
    """
    Auxiliar class used to build the tensors required for MPS Lagrange interpolation.
    """

    def __init__(
        self,
        order: int,
        local_order: Optional[int] = None,
    ):
        self.d = order
        self.m = local_order if local_order else order
        self.D = order + 1
        self.c = np.array(
            [0.5 * (np.cos(np.pi * i / self.d) + 1) for i in range(self.d + 1)]
        )
        self.angular_grid = np.array([i * np.pi / self.d for i in range(self.d + 1)])
        if local_order is not None:
            self.extended_grid = np.array(
                [(i * np.pi) / self.d for i in range(-self.d, 2 * self.d + 1)]
            )
        # Precompute cardinal terms
        self.den = self.c[:, np.newaxis] - self.c
        np.fill_diagonal(self.den, 1)
        self.log_den = np.log(abs(self.den))
        self.sign_den = np.sign(self.den)

    @lru_cache(maxsize=None)  # Unbound cache
    def angular_index(self, theta: float) -> int:
        """
        Returns the index of the closest point of theta to an equispaced angular grid
        defined in [0, ..., π].
        """
        return int(np.argmin(abs(theta - self.angular_grid)))

    def chebyshev_cardinal(self, x: np.ndarray, j: int, use_logs: bool) -> float:
        """
        Evaluates the j-th Chebyshev cardinal function (the Lagrange interpolating
        polynomial for the Chebyshev-Lobatto nodes) at a given point x.
        """
        # TODO: Vectorize for the j index a numpy array
        num = np.delete(x[:, np.newaxis] - self.c, j, axis=1)
        if use_logs:  # Prevents overflow
            with np.errstate(divide="ignore"):  # Ignore warning of log(0)
                log_num = np.log(abs(num))
            log_den = np.delete(self.log_den[j], j)
            log_div = np.sum(log_num - log_den, axis=1)
            sign_num = np.sign(num)
            sign_den = np.delete(self.sign_den[j], j)
            sign_div = np.prod(sign_num * sign_den, axis=1)
            return sign_div * np.exp(log_div)
        else:
            den = np.delete(self.den[j], j)
            return np.prod(num / den, axis=1)

    def local_chebyshev_cardinal(self, x: float, j: int) -> float:
        """
        Evaluates the j-th local Chebyshev cardinal function at a given point x
        by means of a local angular Lagrange interpolation on an extended angular grid
        defined in [-π, ..., 2*π]
        """
        # TODO: Vectorize for x a numpy array
        # TODO: Vectorize for the j index a numpy array
        theta = np.arccos(2 * x - 1)
        idx = self.angular_index(theta)

        P = 0.0
        for gamma in range(idx - self.m, idx + self.m + 1):
            # Find the unique representative of gamma in [0, ..., d]
            gamma_res = (
                -gamma
                if gamma < 0
                else self.d - (gamma - self.d) if gamma > self.d else gamma
            )
            if j == gamma_res:
                P += self.local_angular_cardinal(theta, gamma)
        return P

    def local_angular_cardinal(self, theta: float, gamma: int) -> float:
        """
        Evaluates the gamma-th angular Lagrange interpolating polynomial at a given point theta
        on an extended angular grid defined in [-π, ..., 2*π].
        """
        idx = self.angular_index(theta)
        L = 1
        for beta in range(idx - self.m, idx + self.m + 1):
            if beta != gamma:
                L *= (theta - self.extended_grid[self.d + beta]) / (
                    self.extended_grid[self.d + gamma]
                    - self.extended_grid[self.d + beta]
                )
        return L

    def A_L(self, func: Callable, start: float, stop: float) -> np.ndarray:
        """
        Returns the left-most MPS tensor required for Chebyshev interpolation.
        """
        A = np.zeros((1, 2, self.D))
        for s in range(2):
            A[0, s, :] = func(
                array_affine(0.5 * (s + self.c), orig=(0, 1), dest=(start, stop))
            )
        return A

    def A_C(self, use_logs: bool = True) -> np.ndarray:
        """
        Returns the central MPS tensor required for Chebyshev interpolation.
        """
        A = np.zeros((self.D, 2, self.D))
        for s in range(2):
            for i in range(self.D):
                A[i, s, :] = self.chebyshev_cardinal(0.5 * (s + self.c), i, use_logs)
        return A

    def A_R(self, use_logs: bool = True) -> np.ndarray:
        """
        Returns the right-most MPS tensor required for Chebyshev interpolation.
        """
        A = np.zeros((self.D, 2, 1))
        for s in range(2):
            for i in range(self.D):
                A[i, s, 0] = self.chebyshev_cardinal(np.array([0.5 * s]), i, use_logs)
        return A

    def A_C_sparse(self) -> csc_array:
        """
        Returns the central MPS tensor required for local Chebyshev interpolation.
        For efficiency, it is represented as a (d+1, 2*(d+1)) sparse matrix (CSR).
        """
        A = dok_matrix((self.D, 2 * self.D), dtype=np.float64)
        for s in range(2):
            for i in range(self.D):
                # TODO: Vectorize this loop
                for j, c_j in enumerate(self.c):
                    val = self.local_chebyshev_cardinal(0.5 * (s + c_j), i)
                    if val != 0:
                        A[i, s * self.D + j] = val
        return A.tocsc()

    def A_R_sparse(self) -> csc_array:
        """
        Returns the right-most MPS tensor required for local Chebyshev interpolation.
        For efficiency, it is represented as a (d+1, 2) sparse matrix (CSR).
        """
        A = dok_matrix((self.D, 2), dtype=np.float64)
        for s in range(2):
            for i in range(self.D):
                val = self.local_chebyshev_cardinal(0.5 * s, i)
                if val != 0:
                    A[i, s] = val
        return A.tocsc()
