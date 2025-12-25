# mypy: ignore-errors

# TODO: This module is full of type errors because it is a patch that reuses TensorArray
# for code reuse, which assumes dense cores. It should be redone.

from __future__ import annotations
import numpy as np
from scipy.sparse import csr_array
from typing import Iterable
from ...state import MPS, TensorArray
from ...cython import _contract_last_and_first
from ...typing import Vector, Matrix, Tensor3, NDArray


class SparseCore:
    """
    Sparse rank-3 MPS tensor core.

    Stores a physical slice for each physical index as a CSR matrix, reducing memory and
    accelerating contractions when the structure is sparse. All slices share the same (r_L, r_R) shape.
    """

    def __init__(self, data: list[csr_array]):
        r_L, r_R = data[0].shape
        s = len(data)
        for matrix in data:
            if matrix.shape != (r_L, r_R):
                raise ValueError("All tensor slices must be of the same shape.")

        self.data = data
        self.shape = (r_L, s, r_R)

    def conj(self) -> SparseCore:
        return SparseCore([matrix.conj() for matrix in self.data])

    def transpose(self) -> SparseCore:
        return SparseCore([A.T.tocsr() for A in self.data])

    def to_dense(self) -> Tensor3:
        core = np.zeros(self.shape, dtype=np.float64)
        for idx, matrix in enumerate(self.data):
            core[:, idx, :] = matrix.toarray()
        return core


class SparseMPS(TensorArray):
    """
    MPS whose cores are stored in sparse form.

    A `SparseMPS` is a sequence of `SparseCore` objects (or dense cores for compatibility).
    It mirrors the interface of a dense `MPS` while enabling efficient contractions for sparsely
    structured models, such as computational tree MPS approximations.
    """

    __data: list[SparseCore | Tensor3]  # pyright: ignore

    def __init__(self, data: Iterable[SparseCore | Tensor3]):
        super().__init__(data)  # pyright: ignore

    def physical_dimensions(self) -> list[int]:
        return list(a.shape[1] for a in self._data)

    def bond_dimensions(self) -> list[int]:
        return list(a.shape[0] for a in self._data) + [self._data[-1].shape[-1]]

    def max_bond_dimension(self) -> int:
        return max(self.bond_dimensions())

    def to_dense(self) -> MPS:
        return MPS([core.to_dense() for core in self._data])  # pyright: ignore

    def to_vector(self) -> Vector:
        Ψ = np.ones((1, 1))
        for core in self._data:
            Ψ = _contract_left(Ψ, core).reshape(-1, core.shape[2])
        return Ψ.reshape(-1)


def _contract_left(tensor: NDArray, core: SparseCore | Tensor3) -> Tensor3:
    """Contracts the tensor's last index with the sparse core's first index."""
    r_L, s, r_R = core.shape
    if tensor.shape[-1] != r_L:
        raise ValueError("Invalid dimensions.")

    if isinstance(core, SparseCore):
        A = tensor.reshape(-1, r_L)
        new_tensor = np.zeros(tensor.shape[:-1] + (s, r_R), dtype=np.float64)
        for s_idx, B in enumerate(core.data):
            C = A @ B
            new_tensor[:, s_idx, :] = C.reshape(tensor.shape[:-1] + (r_R,))
    else:
        new_tensor = _contract_last_and_first(tensor, core)
    return new_tensor


def _contract_right(core: SparseCore | Tensor3, tensor: NDArray) -> Tensor3:
    """Contracts the sparse core's last index with the tensor's first index."""
    r_L, s, r_R = core.shape
    if tensor.shape[0] != r_R:
        raise ValueError("Invalid dimensions.")

    if isinstance(core, SparseCore):
        new_tensor = np.zeros((r_L, s) + tensor.shape[1:], dtype=np.float64)
        A = tensor.reshape(r_R, -1)
        for s_idx, B in enumerate(core.data):
            C = B @ A
            new_tensor[:, s_idx, :] = C.reshape((r_L,) + tensor.shape[1:])
    else:
        new_tensor = _contract_last_and_first(core, tensor)
    return new_tensor


def _contract_left_environment(
    environment: Tensor3, core: SparseCore | Tensor3
) -> NDArray:
    """Contracts the environment with the sparse core's first and physical indices."""
    if isinstance(core, SparseCore):
        r_L_core, s_core, r_R_core = core.shape
        r_L_env, s_env, r_R_env = environment.shape

        if s_core != s_env or r_L_core != r_L_env:
            raise ValueError("Invalid dimensions")

        ρ = np.zeros((r_R_core, r_R_env), dtype=np.float64)
        for s_idx, matrix in enumerate(core.data):
            env_slice = environment[:, s_idx, :]
            ρ += matrix.T @ env_slice
    else:
        ρ = np.einsum("ijn,ijk->kn", environment, core)
    return ρ


def _contract_right_environment(
    core: SparseCore | Tensor3, environment: Tensor3
) -> NDArray:
    """Contracts the environment with the sparse core's physical and last indices."""
    raise NotImplementedError()


def _update_left_environment(B: SparseCore, A: SparseCore, ρ: Matrix) -> Matrix:
    return _contract_left_environment(_contract_left(ρ, A), B.conj())


def _update_right_environment(B: SparseCore, A: SparseCore, ρ: Matrix) -> Matrix:
    return _contract_right_environment(B.conj(), _contract_right(A, ρ))


def get_environment(
    bra: SparseMPS, ket: SparseMPS, left_to_right: bool = False
) -> Matrix:
    """
    Build the contraction environment between two SparseMPS objects.

    Sweeps left-to-right or right-to-left, accumulating the effective
    operator acting between corresponding cores.
    """

    if len(bra) != len(ket):
        raise ValueError("The bra and ket must be of the same length.")

    ρ = np.eye(1)
    if left_to_right:
        for B, A in zip(bra, ket):
            ρ = _update_left_environment(B, A, ρ)  # pyright: ignore
    else:
        for B, A in zip(bra[::-1], ket[::-1]):
            ρ = _update_right_environment(B, A, ρ)  # pyright: ignore
    return ρ


def scprod_filter(filter: SparseMPS, distribution: MPS) -> Vector:
    ρ = get_environment(filter[:-1], distribution, left_to_right=True)  # pyright: ignore
    return _contract_last_and_first(ρ.T, filter[-1]).reshape(-1)
