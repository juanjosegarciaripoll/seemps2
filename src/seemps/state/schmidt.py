from __future__ import annotations
import numpy as np
import math
from math import sqrt
from typing import Sequence
from numpy.typing import NDArray
from ..typing import VectorLike, Tensor3
from .core import (
    Strategy,
    DEFAULT_STRATEGY,
    _destructive_svd,
    _schmidt_weights,
    _left_orth_2site,
    _right_orth_2site,
)


def _vector2mps(
    state: VectorLike,
    dimensions: Sequence[int],
    strategy: Strategy = DEFAULT_STRATEGY,
    normalize: bool = True,
    center: int = -1,
) -> tuple[list[Tensor3], float]:
    """Construct a list of tensors for an MPS that approximates the state ψ
    represented as a complex vector in a Hilbert space.

    Parameters
    ----------
    ψ         -- wavefunction with \\prod_i dimensions[i] elements
    dimensions -- list of dimensions of the Hilbert spaces that build ψ
    tolerance -- truncation criterion for dropping Schmidt numbers
    normalize -- boolean to determine if the MPS is normalized
    """
    ψ: NDArray = np.asarray(state).copy().reshape(1, -1, 1)
    L = len(dimensions)
    if math.prod(dimensions) != ψ.size:
        raise Exception("Wrong dimensions specified when converting a vector to MPS")
    output = [ψ] * L
    if center < 0:
        center = L + center
    if center < 0 or center >= L:
        raise Exception("Invalid value of center in _vector2mps")
    err = 0.0
    for i in range(center):
        output[i], ψ, new_err = _left_orth_2site(
            ψ.reshape(ψ.shape[0], dimensions[i], -1, ψ.shape[-1]), strategy
        )
        err += sqrt(new_err)
    for i in range(L - 1, center, -1):
        ψ, output[i], new_err = _right_orth_2site(
            ψ.reshape(ψ.shape[0], -1, dimensions[i], ψ.shape[-1]), strategy
        )
        err += sqrt(new_err)
    if normalize:
        N: float = np.linalg.norm(ψ.reshape(-1))  # type: ignore
        ψ /= N
        err /= N
    output[center] = ψ
    return output, err * err


__all__ = ["_destructive_svd", "_schmidt_weights", "_vector2mps"]
