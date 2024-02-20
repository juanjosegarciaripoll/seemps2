import numpy as np
import math
from typing import Sequence
from numpy.typing import NDArray
from ..typing import VectorLike, Tensor3
from . import core
from .core import Strategy, truncate_vector, DEFAULT_STRATEGY
from scipy.linalg import svd  # type: ignore

#
# Type of LAPACK driver used for solving singular value decompositions.
# The "gesdd" algorithm is the default in Python and is faster, but it
# may produced wrong results, specially in ill-conditioned matrices.
#
SVD_LAPACK_DRIVER = "gesdd"


def _schmidt_split(ψ, strategy, overwrite):
    U, s, V = svd(
        ψ,
        full_matrices=False,
        overwrite_a=overwrite,
        check_finite=False,
        lapack_driver=SVD_LAPACK_DRIVER,
    )
    s, _ = core.truncate_vector(s, strategy)
    D = s.size
    return U[:, :D], s.reshape(D, 1) * V[:D, :]


def ortho_right(A, strategy: Strategy):
    α, i, β = A.shape
    U, s, V = svd(
        A.reshape(α * i, β),
        full_matrices=False,
        check_finite=False,
        lapack_driver=SVD_LAPACK_DRIVER,
    )
    s, err = truncate_vector(s, strategy)
    D = s.size
    return U[:, :D].reshape(α, i, D), s.reshape(D, 1) * V[:D, :], err


def ortho_left(A, strategy: Strategy):
    α, i, β = A.shape
    U, s, V = svd(
        A.reshape(α, i * β),
        full_matrices=False,
        check_finite=False,
        lapack_driver=SVD_LAPACK_DRIVER,
    )
    s, err = truncate_vector(s, strategy)
    D = s.size
    return V[:D, :].reshape(D, i, β), U[:, :D] * s.reshape(1, D), err


def left_orth_2site(AA, strategy: Strategy):
    """Split a tensor AA[a,b,c,d] into B[a,b,r] and C[r,c,d] such
    that 'B' is a left-isometry, truncating the size 'r' according
    to the given 'strategy'. Tensor 'AA' may be overwritten."""
    α, d1, d2, β = AA.shape
    U, S, V = svd(
        AA.reshape(α * d1, β * d2),
        full_matrices=False,
        overwrite_a=True,
        check_finite=False,
        lapack_driver=SVD_LAPACK_DRIVER,
    )
    S, err = core.truncate_vector(S, strategy)
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
    U, S, V = svd(
        AA.reshape(α * d1, β * d2),
        full_matrices=False,
        overwrite_a=True,
        lapack_driver=SVD_LAPACK_DRIVER,
        check_finite=False,
    )
    S, err = truncate_vector(S, strategy)
    D = S.size
    return (U[:, :D] * S).reshape(α, d1, D), V[:D, :].reshape(D, d2, β), err


def vector2mps(
    state: VectorLike,
    dimensions: Sequence[int],
    strategy: Strategy = DEFAULT_STRATEGY,
    normalize: bool = True,
) -> list[Tensor3]:
    """Construct a list of tensors for an MPS that approximates the state ψ
    represented as a complex vector in a Hilbert space.

    Parameters
    ----------
    ψ         -- wavefunction with \\prod_i dimensions[i] elements
    dimensions -- list of dimensions of the Hilbert spaces that build ψ
    tolerance -- truncation criterion for dropping Schmidt numbers
    normalize -- boolean to determine if the MPS is normalized
    """
    ψ: NDArray = np.asarray(state)
    if math.prod(dimensions) != ψ.size:
        raise Exception("Wrong dimensions specified when converting a vector to MPS")
    output = [ψ] * len(dimensions)
    Da = 1
    for i, d in enumerate(dimensions[:-1]):
        # We split a new subsystem and group the left bond dimension
        # and the physical index into a large index.
        # We then split the state using the Schmidt decomposition. This
        # produces a tensor for the site we are looking at and leaves
        # us with a (hopefully) smaller state for the rest
        A, ψ = _schmidt_split(ψ.reshape(Da * d, -1), strategy, overwrite=(i > 0))
        output[i] = A.reshape(Da, d, -1)
        Da = ψ.shape[0]

    if normalize is True:
        ψ /= np.linalg.norm(ψ)
    output[-1] = ψ.reshape(Da, dimensions[-1], 1)

    return output
