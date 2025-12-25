from __future__ import annotations
import numpy as np
import math
from collections.abc import Sequence
from numpy.typing import NDArray
from typing import Literal
from ..typing import VectorLike, Tensor3, Vector
from ..cython.core import Strategy, DEFAULT_STRATEGY
from scipy.linalg import svd as _scipy_svd
from ..cython.core import _destructive_svd, _left_orth_2site, _right_orth_2site

#
# Type of LAPACK driver used for solving singular value decompositions.
# The "gesdd" algorithm is the default in Python and is faster, but it
# may produced wrong results, specially in ill-conditioned matrices.
#
SVD_LAPACK_DRIVER: Literal["gesvd", "gesdd"] = "gesvd"


def _schmidt_weights(A: Tensor3) -> Vector:
    d1, d2, d3 = A.shape
    s: Vector = _scipy_svd(
        A.reshape(d1 * d2, d3),
        full_matrices=False,
        compute_uv=False,
        check_finite=False,
        lapack_driver=SVD_LAPACK_DRIVER,
    )
    s *= s
    s /= np.sum(s)
    return s


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
    err: float = 0.0
    for i in range(center):
        output[i], ψ, new_err = _left_orth_2site(
            ψ.reshape(ψ.shape[0], dimensions[i], -1, ψ.shape[-1]), strategy
        )
        err += new_err
    for i in range(L - 1, center, -1):
        ψ, output[i], new_err = _right_orth_2site(
            ψ.reshape(ψ.shape[0], -1, dimensions[i], ψ.shape[-1]), strategy
        )
        err += new_err
    if normalize:
        N = np.linalg.norm(ψ.reshape(-1))
        ψ /= N
        err /= float(N)
    output[center] = ψ
    return output, err


__all__ = ["_destructive_svd", "_schmidt_weights", "_vector2mps"]
