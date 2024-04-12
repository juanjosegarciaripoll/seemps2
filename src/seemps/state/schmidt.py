from __future__ import annotations
import numpy as np
import math
from math import sqrt
from typing import Sequence, Any, Callable
from numpy.typing import NDArray
from ..typing import VectorLike, Tensor3, Vector
from .core import (
    Strategy,
    destructively_truncate_vector,
    DEFAULT_STRATEGY,
    _destructive_svd,
    schmidt_weights,
)
from scipy.linalg import svd, LinAlgError  # type: ignore
from scipy.linalg.lapack import get_lapack_funcs  # type: ignore


def left_orth_2site(AA, strategy: Strategy):
    """Split a tensor AA[a,b,c,d] into B[a,b,r] and C[r,c,d] such
    that 'B' is a left-isometry, truncating the size 'r' according
    to the given 'strategy'. Tensor 'AA' may be overwritten."""
    α, d1, d2, β = AA.shape
    U, S, V = _destructive_svd(AA.reshape(α * d1, β * d2))
    err = destructively_truncate_vector(S, strategy)
    D = S.size
    return (
        U[:, :D].reshape(α, d1, D),
        (S.reshape(D, 1) * V[:D, :]).reshape(D, d2, β),
        err,
    )


def right_orth_2site(AA, strategy: Strategy):
    """Split a tensor AA[a,b,c,d] into B[a,b,r] and C[r,c,d] such
    that 'C' is a right-isometry, truncating the size 'r' according
    to the given 'strategy'. Tensor 'AA' may be overwritten."""
    α, d1, d2, β = AA.shape
    U, S, V = _destructive_svd(AA.reshape(α * d1, β * d2))
    err = destructively_truncate_vector(S, strategy)
    D = S.size
    return (U[:, :D] * S).reshape(α, d1, D), V[:D, :].reshape(D, d2, β), err


def vector2mps(
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
        raise Exception("Invalid value of center in vector2mps")
    err = 0.0
    for i in range(center):
        output[i], ψ, new_err = left_orth_2site(
            ψ.reshape(ψ.shape[0], dimensions[i], -1, ψ.shape[-1]), strategy
        )
        err += sqrt(new_err)
    for i in range(L - 1, center, -1):
        ψ, output[i], new_err = right_orth_2site(
            ψ.reshape(ψ.shape[0], -1, dimensions[i], ψ.shape[-1]), strategy
        )
        err += sqrt(new_err)
    if normalize:
        N: float = np.linalg.norm(ψ.reshape(-1))  # type: ignore
        ψ /= N
        err /= N
    output[center] = ψ
    return output, err * err
