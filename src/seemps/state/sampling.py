from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from numpy.random import Generator, default_rng
from .mps import MPS
from .canonical_mps import CanonicalMPS


def sample_mps(mps: MPS, size: int = 1, rng: Generator = default_rng()) -> NDArray:
    """Generate configurations by sampling a matrix-product state.

    This function samples quantum states according to the probability
    distribution encoded in an MPS. If we label :math:`|i_1i_2\\ldots i_N\\rangle`
    the computational basis on which the MPS is defined, then

    .. math::
        p(i_1,i_2,\\ldots,i_N) = |\\langle{i_1i_2\\ldots i_N}|\\psi\rangle|^2

    is a multivariate probability distribution for generating configurations of
    the physical variables :math:`i_k` simultaneously. Instead of constructing
    this distribution (which takes an exponentially large space), this routine
    generates instances of the integers :math:`i_k` in a method that reproduces
    the same distribution.

    Parameters
    ----------
    mps : MPS
        Normalized matrix product state.
    size : int
        Number of samples to generate, defaults to 1.

    Returns
    -------
    ArrayLike
        A list of configurations sampled according to the above distribution,
        each represented by a Numpy vector.
    """
    if not isinstance(mps, CanonicalMPS):
        mps = CanonicalMPS(mps, center=0)
    L = mps.size
    output = np.empty((size, L), dtype=int)
    for state in output:
        if mps.center == L - 1:
            i = 0
            for n, A in enumerate(reversed(mps)):
                A = A[:, :, i]
                p = np.cumsum(np.abs(A.reshape(-1)) ** 2)
                z = np.searchsorted(p, p[-1] * rng.random())
                i = z // A.shape[1]
                state[n] = z % A.shape[1]
        else:
            if mps.center != 0:
                mps = CanonicalMPS(mps, center=0)
            i = 0
            for n, A in enumerate(mps):
                A = A[i, :, :]
                p = np.cumsum(np.abs(A.reshape(-1)) ** 2)
                z = np.searchsorted(p, p[-1] * rng.random())
                i = z // A.shape[0]
                state[n] = z % A.shape[0]
    return output
