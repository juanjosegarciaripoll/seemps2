from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
import scipy.sparse
import functools
from scipy.sparse import dok_array, csr_array
from typing import Callable

from ..state import MPS, Strategy, DEFAULT_STRATEGY, simplify
from ..state.schmidt import _destructive_svd
from ..cython import _contract_last_and_first
from ..cython.core import destructively_truncate_vector
from ..typing import Tensor3, MPSOrder
from .mesh import Interval, ArrayInterval, Mesh, array_affine


def mps_lagrange_chebyshev_basic(
    func: Callable,
    domain: Interval | Mesh,
    order: int,
    use_logs: bool = True,
    mps_order: MPSOrder = "A",
    strategy: Strategy = DEFAULT_STRATEGY,
) -> MPS:
    """
    Performs a "basic" Lagrange MPS Chebyshev interpolation of a function.

    Parameters
    ----------
    func : Callable
        The function to interpolate.
    domain : Interval | Mesh
        The domain where the function is defined.
    order : int
        The order of the Chebyshev interpolation.
    use_logs : bool, default=True
        Whether to compute the Chebyshev cardinal function using
        logarithms to avoid overflow.
    mps_order : MPSOrder, default='A'
        The order of the MPS cores, either "A" (serial) or "B" (interleaved).
    strategy : Strategy, default=DEFAULT_STRATEGY
        The MPS simplification strategy.

    Returns
    -------
    mps : MPS
        The MPS corresponding to the naive Chebyshev interpolation.
    """
    mesh = Mesh([domain]) if isinstance(domain, Interval) else domain
    _validate_mesh(mesh)

    builder = LagrangeBuilder(order)
    A_L = builder.build_left_core(func, mesh)
    A_C = builder.build_center_core(use_logs)
    A_R = builder.build_right_core(use_logs)
    cores = [A_L] + builder.build_dense_cores(A_C, A_R, mesh, mps_order)[1:]

    mps = MPS(cores)
    return simplify(mps, strategy=strategy)


def mps_lagrange_chebyshev_rr(
    func: Callable,
    domain: Interval | Mesh,
    order: int,
    use_logs: bool = True,
    mps_order: MPSOrder = "A",
    strategy: Strategy = DEFAULT_STRATEGY,
) -> MPS:
    """
    Performs a Lagrange rank-revealing MPS Chebyshev interpolation of a function.

    Parameters
    ----------
    func : Callable
        The function to interpolate.
    domain : Interval | Mesh
        The domain where the function is defined.
    order : int
        The order of the Chebyshev interpolation.
    use_logs : bool, default=True
        Whether to compute the Chebyshev cardinal function using
        logarithms to avoid overflow.
    mps_order : MPSOrder, default='A'
        The order of the MPS cores, either "A" (serial) or "B" (interleaved).
    strategy : Strategy, default=DEFAULT_STRATEGY
        The MPS simplification strategy.

    Returns
    -------
    mps : MPS
        The MPS corresponding to the rank-revealing Chebyshev interpolation.
    """
    mesh = Mesh([domain]) if isinstance(domain, Interval) else domain
    _validate_mesh(mesh)

    builder = LagrangeBuilder(order)
    A_L = builder.build_left_core(func, mesh)
    A_C = builder.build_center_core(use_logs)
    A_R = builder.build_right_core(use_logs)
    cores = builder.build_dense_cores(A_C, A_R, mesh, mps_order)

    U_L, R = np.linalg.qr(A_L.reshape((2, -1)))
    tensors: list[NDArray] = [U_L.reshape(1, 2, 2)]
    for core in cores[1:-1]:
        B = _contract_last_and_first(R, core)
        r1, _, r2 = B.shape
        ## SVD
        U, S, V = _destructive_svd(B.reshape(r1 * 2, r2))
        destructively_truncate_vector(S, strategy)
        D = S.size
        U = U[:, :D]
        R = S.reshape(D, 1) * V[:D, :]
        ##
        tensors.append(U.reshape(r1, 2, -1))
    U_R = _contract_last_and_first(R, cores[-1])
    tensors.append(U_R)
    return MPS(tensors)


def mps_lagrange_chebyshev_lrr(
    func: Callable,
    domain: Interval | Mesh,
    order: int,
    local_order: int,
    mps_order: MPSOrder = "A",
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
    domain : Interval | Mesh
        The domain where the function is defined.
    order : int
        The order of the Chebyshev interpolation.
    local_order : int
        The local order of the Chebyshev interpolation.
    mps_order : MPSOrder, default='A'
        The order of the MPS cores, either "A" (serial) or "B" (interleaved).
    strategy : Strategy, default=DEFAULT_STRATEGY
        The MPS simplification strategy.

    Returns
    -------
    mps : MPS
        The MPS corresponding to the local rank-revealing Chebyshev interpolation.
    """
    # TODO: Perform sparse matrix multiplications
    mesh = Mesh([domain]) if isinstance(domain, Interval) else domain
    _validate_mesh(mesh)

    builder = LagrangeBuilder(order, local_order)
    A_L = builder.build_left_core(func, mesh)
    A_C = builder.build_center_sparse_core()
    A_R = builder.build_right_sparse_core()
    cores = builder.build_sparse_cores(A_C, A_R, mesh, mps_order)

    U_L, R = np.linalg.qr(A_L.reshape((2, -1)))
    tensors: list[NDArray] = [U_L.reshape(1, 2, 2)]
    for core in cores[1:-1]:
        B = R @ core
        r1 = B.shape[0]
        ## SVD
        U, S, V = _destructive_svd(B.reshape(r1 * 2, -1))
        destructively_truncate_vector(S, strategy)
        D = S.size
        U = U[:, :D]
        R = S.reshape(D, 1) * V[:D, :]
        ##
        tensors.append(U.reshape(r1, 2, -1))
    U_R = R @ cores[-1]
    tensors.append(U_R.reshape(-1, 2, 1))
    return MPS(tensors)


def _validate_mesh(mesh: Mesh):
    num_qubits = [int(np.log2(N)) for N in mesh.dimensions]
    if not all(2**n == N for n, N in zip(num_qubits, mesh.dimensions)):
        raise ValueError("The mesh must be quantizable in qubits.")
    if len(set(num_qubits)) != 1:
        raise ValueError("The qubits per dimension must be constant.")


class LagrangeBuilder:
    """Auxiliar class used to build the tensors required for MPS Lagrange interpolation."""

    def __init__(
        self,
        order: int,
        local_order: int | None = None,
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

    @functools.lru_cache(maxsize=None)  # Unbound cache
    def angular_index(self, theta: float) -> int:
        """
        Returns the index of the closest point of theta to an equispaced angular grid
        defined in [0, ..., π].
        """
        return int(np.argmin(abs(theta - self.angular_grid)))

    def chebyshev_cardinal(self, x: np.ndarray, j: int, use_logs: bool) -> np.ndarray:
        """
        Evaluates the j-th Chebyshev cardinal function (the Lagrange interpolating
        polynomial for the Chebyshev-Lobatto nodes) at a given point x.
        """
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
        θ = np.arccos(2 * x - 1)
        idx = self.angular_index(θ)

        P = 0.0
        for γ in range(idx - self.m, idx + self.m + 1):
            γ_rep = -γ if γ < 0 else self.d - (γ - self.d) if γ > self.d else γ
            if j == γ_rep:
                P += self.local_angular_cardinal(θ, γ)
        return P

    def local_angular_cardinal(self, θ: float, γ: int) -> float:
        """
        Evaluates the γ-th angular Lagrange interpolating polynomial at a given point θ
        on an extended angular grid defined in [-π, ..., 2*π].
        """
        idx = self.angular_index(θ)
        L = 1
        for β in range(idx - self.m, idx + self.m + 1):
            if β != γ:
                L *= (θ - self.extended_grid[self.d + β]) / (
                    self.extended_grid[self.d + γ] - self.extended_grid[self.d + β]
                )
        return L

    def build_left_core(
        self, func: Callable, mesh: Mesh, channels_first: bool = True
    ) -> Tensor3:
        """
        Returns the left-most MPS core required for Chebyshev interpolation.
        """
        m = mesh.dimension
        A = np.zeros((1, 2, self.D**m))
        for σ in [0, 1]:
            intervals: list[Interval] = []
            for i in range(m):
                a, b = mesh.intervals[i].start, mesh.intervals[i].stop
                c = (σ + self.c) / 2 if i == 0 else self.c
                arr = array_affine(c, (0, 1), (a, b))
                intervals.append(ArrayInterval(arr))
            c_mesh = Mesh(intervals)
            tensor = c_mesh.to_tensor(channels_first)
            A[0, σ, :] = func(tensor).reshape(-1)
        return A

    def build_center_core(self, use_logs: bool) -> Tensor3:
        """
        Returns the central MPS tensor required for Chebyshev interpolation.
        """
        A = np.zeros((self.D, 2, self.D))
        for σ in range(2):
            for i in range(self.D):
                A[i, σ, :] = self.chebyshev_cardinal(0.5 * (σ + self.c), i, use_logs)
        return A

    def build_right_core(self, use_logs: bool) -> Tensor3:
        """
        Returns the right-most MPS tensor required for Chebyshev interpolation.
        """
        A = np.zeros((self.D, 2, 1))
        for σ in range(2):
            for i in range(self.D):
                A[i, σ, :] = self.chebyshev_cardinal(np.array([0.5 * σ]), i, use_logs)
        return A

    def build_center_sparse_core(self) -> csr_array:
        """
        Returns the central MPS tensor required for local Chebyshev interpolation.
        For efficiency, it is represented as a (d+1, 2*(d+1)) sparse matrix (CSR).
        """
        A = dok_array((self.D, 2 * self.D), dtype=np.float64)
        for σ in range(2):
            for i in range(self.D):
                for j, c_j in enumerate(self.c):
                    A[i, σ * self.D + j] = self.local_chebyshev_cardinal(
                        0.5 * (σ + c_j), i
                    )
        return A.tocsr()

    def build_right_sparse_core(self) -> csr_array:
        """
        Returns the right-most MPS tensor required for local Chebyshev interpolation.
        For efficiency, it is represented as a (d+1, 2) sparse matrix (CSR).
        """
        A = dok_array((self.D, 2), dtype=np.float64)
        for σ in range(2):
            for i in range(self.D):
                A[i, σ] = self.local_chebyshev_cardinal(0.5 * σ, i)
        return A.tocsr()

    @staticmethod
    def build_dense_cores(
        A_C: Tensor3, A_R: Tensor3, mesh: Mesh, mps_order: MPSOrder
    ) -> list[Tensor3]:
        """
        Builds the multidimensional cores on the given mesh and mps_order.
        """
        m = mesh.dimension
        n = int(np.log2(mesh.dimensions[0]))

        A_R_kron = [_kron_dense(A_R, m - i, 0) for i in range(m)]
        if mps_order == "A":
            A_C_kron = [_kron_dense(A_C, m - i, 0) for i in range(m)]
            cores = []
            for A_C, A_R in zip(A_C_kron, A_R_kron):
                cores.extend([A_C] * (n - 1) + [A_R])
        elif mps_order == "B":
            A_C_kron = [_kron_dense(A_C, m, i) for i in range(m)]
            cores = A_C_kron * (n - 1) + A_R_kron

        return cores

    @staticmethod
    def build_sparse_cores(
        A_C: csr_array, A_R: csr_array, mesh: Mesh, mps_order: MPSOrder
    ) -> list[csr_array]:
        """
        Builds the multidimensional sparse cores on the given mesh and mps_order.
        """
        m = mesh.dimension
        n = int(np.log2(mesh.dimensions[0]))

        A_R_kron = [_kron_sparse(A_R, m - i, 0) for i in range(m)]
        if mps_order == "A":
            A_C_kron = [_kron_sparse(A_C, m - i, 0) for i in range(m)]
            cores = []
            for A_C, A_R in zip(A_C_kron, A_R_kron):
                cores.extend([A_C] * (n - 1) + [A_R])
        elif mps_order == "B":
            A_C_kron = [_kron_sparse(A_C, m, i) for i in range(m)]
            cores = A_C_kron * (n - 1) + A_R_kron

        return cores


def _kron_dense(A: Tensor3, m: int, i: int) -> Tensor3:
    """
    Take the Kronecker product of the tensor A with identity matrices along m dimensions.
    The function reshapes A from (i, s, j) to (s, i, j) and back to (i*i, s, j*i) after the operation.
    """
    I = np.eye(A.shape[0])
    tensors = [np.swapaxes(A, 0, 1) if j == i else I for j in range(m)]
    B = tensors[0]
    for tensor in tensors[1:]:
        B = np.kron(B, tensor)
    return np.swapaxes(B, 0, 1)


def _kron_sparse(A: csr_array, m: int, i: int) -> csr_array:
    """
    Take the Kronecker product of the sparse tensor A with identity matrices along m dimensions.
    This operation is implemented converting the CSR matrix to a dense format as a temporary workaround.
    """
    # TODO: Fix without transforming to dense matrices.
    A_dense = A.toarray().reshape(A.shape[0], 2, A.shape[1] // 2)
    B = _kron_dense(A_dense, m, i)
    return scipy.sparse.csr_array(B.reshape(B.shape[0], 2 * B.shape[2]))
