from __future__ import annotations
import numpy as np
from ..state import MPS
from ..typing import Vector, Matrix


def evaluate_mps(mps: MPS, mps_indices: Matrix) -> Vector:
    """
    Evaluates a collection of MPS indices by contracting the MPS tensors.

    Parameters
    ----------
    mps : MPS
        The MPS to evaluate.
    mps_indices : Matrix
        An array of indices to be evaluated on the MPS.

    Returns
    -------
    Vector
        The vector of evaluations corresponding to the provided indices.
    """
    if mps_indices.ndim == 1:
        mps_indices = mps_indices.reshape(1, -1)
    A = mps[0][:, mps_indices[:, 0], :]
    for idx, site in enumerate(mps[1:]):
        B = site[:, mps_indices[:, idx + 1], :]
        # np.einsum("kq,qkr->kr", A, B)
        A = np.matmul(A.transpose(1, 0, 2), B.transpose(1, 0, 2)).transpose(1, 0, 2)
    return A.reshape(-1)


def random_mps_indices(
    physical_dimensions: list[int],
    num_indices: int = 1000,
    allowed_indices: list[int] | None = None,
    rng: np.random.Generator = np.random.default_rng(),
) -> Vector:
    """
    Generates random indices for sampling a MPS.

    Parameters
    ----------
    physical_dimensions : list[int]
        The physical dimensions of the MPS.
    num_indices : int, default=1000
        The number of random indices to generate.
    allowed_indices : list[int], optional
        An optional list with allowed values for the random indices.
    rng : np.random.Generator, default=`numpy.random.default_rng()`
        The random number generator to be used. If None, uses Numpy's
        default random number generator without any predefined seed.

    Returns
    -------
    Vector
        An array of random MPS indices.
    """
    mps_indices = []
    for k in physical_dimensions:
        indices = list(range(k))
        if allowed_indices is not None:
            indices = list(set(indices) & set(allowed_indices))
        mps_indices.append(rng.choice(indices, num_indices))
    return np.vstack(mps_indices).T
