from __future__ import annotations
import numpy as np
import math
from math import sqrt
from typing import Sequence, Any, Callable
from numpy.typing import NDArray
from ..typing import VectorLike, Tensor3, Vector
from .core import Strategy, truncate_vector, DEFAULT_STRATEGY
from scipy.linalg import svd, LinAlgError  # type: ignore
from scipy.linalg.lapack import get_lapack_funcs  # type: ignore

#
# Type of LAPACK driver used for solving singular value decompositions.
# The "gesdd" algorithm is the default in Python and is faster, but it
# may produced wrong results, specially in ill-conditioned matrices.
#
SVD_LAPACK_DRIVER = "gesdd"

_lapack_svd_driver: dict[Any, tuple[Callable, Callable]] = dict()


def set_svd_driver(driver: str) -> None:
    global _lapack_svd_driver, SVD_LAPACK_DRIVER
    SVD_LAPACK_DRIVER = driver
    _lapack_svd_driver = {
        dtype: get_lapack_funcs(
            (driver, driver + "_lwork"),
            [np.zeros((10, 10), dtype=dtype)],
            ilp64="preferred",
        )
        for dtype in (np.float64, np.complex128)
    }


set_svd_driver("gesdd")


def _our_svd(A: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    _lapack_svd, _lapack_svd_lwork = _lapack_svd_driver[type(A[0, 0])]

    # compute optimal lwork
    lwork, flag = _lapack_svd_lwork(
        A.shape[0],
        A.shape[1],
        compute_uv=True,
        full_matrices=False,
    )
    if flag != 0:
        raise ValueError("Internal work array size computation failed: %d" % flag)

    # perform decomposition
    u, s, v, info = _lapack_svd(
        A,
        compute_uv=True,
        lwork=int(lwork.real),
        full_matrices=False,
        overwrite_a=True,
    )
    if info == 0:
        return u, s, v
    elif info > 0:
        raise LinAlgError("SVD did not converge")
    else:
        raise ValueError("illegal value in %dth argument of internal gesdd" % -info)


def schmidt_weights(A: Tensor3) -> Vector:
    d1, d2, d3 = A.shape
    s = svd(
        A.reshape(d1 * d2, d3),
        full_matrices=False,
        compute_uv=False,
        check_finite=False,
        lapack_driver=SVD_LAPACK_DRIVER,
    )
    s *= s
    s /= np.sum(s)
    return s


def ortho_right(A, strategy: Strategy):
    α, i, β = A.shape
    U, s, V = _our_svd(A.reshape(α * i, β).copy())
    s, err = truncate_vector(s, strategy)
    D = s.size
    return U[:, :D].reshape(α, i, D), s.reshape(D, 1) * V[:D, :], err


def ortho_left(A, strategy: Strategy):
    α, i, β = A.shape
    U, s, V = _our_svd(A.reshape(α, i * β).copy())
    s, err = truncate_vector(s, strategy)
    D = s.size
    return V[:D, :].reshape(D, i, β), U[:, :D] * s.reshape(1, D), err


def left_orth_2site(AA, strategy: Strategy):
    """Split a tensor AA[a,b,c,d] into B[a,b,r] and C[r,c,d] such
    that 'B' is a left-isometry, truncating the size 'r' according
    to the given 'strategy'. Tensor 'AA' may be overwritten."""
    α, d1, d2, β = AA.shape
    U, S, V = _our_svd(AA.reshape(α * d1, β * d2))
    S, err = truncate_vector(S, strategy)
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
    U, S, V = _our_svd(AA.reshape(α * d1, β * d2))
    S, err = truncate_vector(S, strategy)
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
