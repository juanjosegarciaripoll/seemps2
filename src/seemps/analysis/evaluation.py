from __future__ import annotations
import numpy as np
from ..cython import _evaluate_mps
from ..tools import DEFAULT_RNG
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
    return _evaluate_mps(mps, mps_indices)


def random_mps_indices(
    physical_dimensions: list[int],
    num_indices: int = 1000,
    allowed_indices: list[int] | None = None,
    rng: np.random.Generator = DEFAULT_RNG,
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
    rng : np.random.Generator, default=`seemps.tools.DEFAULT_RNG`
        The random number generator to be used. If None, uses SeeMPS
        common, preseeded generator.

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
