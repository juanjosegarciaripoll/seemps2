from __future__ import annotations
import numpy as np
from typing import Iterable
from ...state import MPS
from ...cython import _contract_last_and_first
from ...typing import Vector, Matrix, Tensor3, NDArray
from .sparse_array import SparseMPSTensor, SparseTensorArray, SparseCore

# TODO: We should be using sparse.sparray


class SparseMPS(SparseTensorArray):
    """
    MPS whose cores are stored in sparse form.

    A `SparseMPS` is a sequence of `SparseCore` objects (or dense cores for compatibility).
    It mirrors the interface of a dense `MPS` while enabling efficient contractions for sparsely
    structured models, such as computational tree MPS approximations.
    """

    def __init__(self, data: Iterable[SparseMPSTensor]):
        super().__init__(data)

    def physical_dimensions(self) -> list[int]:
        return list(a.shape[1] for a in self._data)

    def bond_dimensions(self) -> list[int]:
        return list(a.shape[0] for a in self._data) + [self._data[-1].shape[-1]]

    def max_bond_dimension(self) -> int:
        return max(self.bond_dimensions())

    def to_dense(self) -> MPS:
        return MPS(
            [
                core.to_dense() if isinstance(core, SparseCore) else core
                for core in self._data
            ]
        )

    def to_vector(self) -> Vector:
        Ψ = np.ones((1, 1))
        for core in self._data:
            Ψ = _contract_left(Ψ, core).reshape(-1, core.shape[2])
        return Ψ.reshape(-1)


def _contract_left(tensor: NDArray, core: SparseMPSTensor) -> Tensor3:
    """Contracts the tensor's last index with the sparse core's first index."""

    if isinstance(core, SparseCore):
        r_L, s, r_R = core.shape
        if tensor.shape[-1] != r_L:
            raise ValueError("Invalid dimensions.")
        A = tensor.reshape(-1, r_L)
        new_tensor = np.zeros(tensor.shape[:-1] + (s, r_R), dtype=np.float64)
        for s_idx, B in enumerate(core.data):
            C = A @ B
            new_tensor[:, s_idx, :] = C.reshape(tensor.shape[:-1] + (r_R,))
    else:
        new_tensor = _contract_last_and_first(tensor, core)
    return new_tensor


def _contract_right(core: SparseMPSTensor, tensor: NDArray) -> Tensor3:
    """Contracts the sparse core's last index with the tensor's first index."""

    if isinstance(core, SparseCore):
        r_L, s, r_R = core.shape
        if tensor.shape[0] != r_R:
            raise ValueError("Invalid dimensions.")
        new_tensor = np.zeros((r_L, s) + tensor.shape[1:], dtype=np.float64)
        A = tensor.reshape(r_R, -1)
        for s_idx, B in enumerate(core.data):
            C = B @ A
            new_tensor[:, s_idx, :] = C.reshape((r_L,) + tensor.shape[1:])
    else:
        new_tensor = _contract_last_and_first(core, tensor)
    return new_tensor


def _contract_left_environment(environment: Tensor3, core: SparseMPSTensor) -> NDArray:
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


def _contract_right_environment(core: SparseMPSTensor, environment: Tensor3) -> NDArray:
    """Contracts the environment with the sparse core's physical and last indices."""
    raise NotImplementedError()


def _update_left_environment(
    B: SparseMPSTensor, A: SparseMPSTensor, ρ: Matrix
) -> Matrix:
    return _contract_left_environment(_contract_left(ρ, A), B.conj())


def _update_right_environment(
    B: SparseMPSTensor, A: SparseMPSTensor, ρ: Matrix
) -> Matrix:
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
            ρ = _update_left_environment(B, A, ρ)
    else:
        for B, A in zip(bra[::-1], ket[::-1]):
            ρ = _update_right_environment(B, A, ρ)
    return ρ


def scprod_filter(filter: SparseMPS, distribution: MPS) -> Vector:
    ρ = get_environment(filter[:-1], distribution, left_to_right=True)  # pyright: ignore
    return _contract_last_and_first(ρ.T, filter[-1]).reshape(-1)  # pyright: ignore
