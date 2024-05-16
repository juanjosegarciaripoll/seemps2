from __future__ import annotations
import numpy as np
from typing import Optional

from ..state import MPS


def evaluate_mps(mps: MPS, mps_indices: np.ndarray) -> np.ndarray:
    """
    Evaluates a collection of MPS indices by contracting the MPS tensors.

    Parameters
    ----------
    mps : MPS
        The MPS to evaluate.
    mps_indices : np.ndarray
        An array of indices to be evaluated on the MPS.

    Returns
    -------
    np.ndarray
        The array of evaluations corresponding to the provided indices."""
    if mps_indices.ndim == 1:
        mps_indices = mps_indices[np.newaxis, :]
    A = mps[0][:, mps_indices[:, 0], :]
    for idx, site in enumerate(mps[1:]):
        B = site[:, mps_indices[:, idx + 1], :]
        # np.einsum("kq,qkr->kr", A, B)
        A = np.matmul(A.transpose(1, 0, 2), B.transpose(1, 0, 2)).transpose(1, 0, 2)
    return A.reshape(-1)


def random_mps_indices(
    mps: MPS,
    num_indices: int = 1000,
    allowed_indices: Optional[list[int]] = None,
    rng: np.random.Generator = np.random.default_rng(),
):
    """
    Generates random indices for sampling a MPS.

    Parameters
    ----------
    mps : MPS
        The matrix product state to sample from.
    num_indices : int, default 1000
        The number of random indices to generate.
    rng : np.random.Generator, default=`numpy.random.default_rng()`
        The random number generator to be used. If None, uses Numpy's
        default random number generator without any predefined seed.
    allowed_indices : Optional[tuple], default=None
        An optional tuple with allowed values for the random indices.
    Returns
    -------
    indices : np.ndarray
        An array of random MPS indices."""
    mps_indices = []
    for k in mps.physical_dimensions():
        indices = list(range(k))
        if allowed_indices is not None:
            indices = list(set(indices) & set(allowed_indices))
        mps_indices.append(rng.choice(indices, num_indices))
    return np.vstack(mps_indices).T
