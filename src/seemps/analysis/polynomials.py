from __future__ import annotations
import numpy as np
from numpy.polynomial.polynomial import Polynomial
import scipy.special  # type: ignore
from ..state import MPS, Strategy, DEFAULT_STRATEGY
from ..truncate import simplify
from typing import Optional
from .factories import mps_interval
from .mesh import Interval, RegularHalfOpenInterval, RegularClosedInterval


def _mps_x_tensor(
    degree: int,
    domain: Interval,
    first: bool = False,
) -> MPS:
    r"""
    Construct a tensor representation of a collection of monomials.

    This function creates the tensor :math:`|x_N^m\rangle`, of all possible
    monomials for `m` from 0 up to L-1. This collection of  monomials is
    stored as an MPS where the last tensor has one extra index that is no
    longer of size 1.

    Parameters
    ----------
    degree: int
        Maximum degree of the monomials (must be >= 0)
    domain :Interval
        Interval of definition for the monomial variable.
    first: bool, default = False
        Where to place the extra index: at the first (True) or last (False) site.

    Returns
    -------
    xL : MPS
        MPS representation of the monomials collection.
    """
    if not isinstance(domain, (RegularHalfOpenInterval, RegularClosedInterval)):
        raise ValueError("Unable to construct polyomial for non-regular intervals")
    L = degree + 1
    x_mps: MPS = mps_interval(domain)
    N = len(x_mps)
    for n in range(N):
        # This is the operator with the information about
        # position carried by this qubit (see `mps_equispaced()`)
        On = x_mps[n]
        On = On[0, :, min(1, On.shape[2] - 1)]
        d = len(On)
        An = np.zeros((L, d, L))
        for m in range(L):
            for r in range(m):
                An[r, :, m] = scipy.special.binom(m, r) * On ** (m - r)
            An[m, :, m] = 1.0
        x_mps[n] = An
    if first:
        x_mps[-1] = x_mps[-1][:, :, [0]]
    else:
        x_mps[0] = x_mps[0][[0], :, :]
    return x_mps


def mps_from_polynomial(
    p: Polynomial | np.ndarray,
    domain: Interval,
    first: bool = False,
    strategy: Strategy = DEFAULT_STRATEGY,
):
    r"""
    Construct a tensor representation of a polynomial.

    This function creates the MPS representation of a polynomial `p`,
    whose variable runs over the given `domain`.

    Parameters
    ----------
    p : numpy.polynomial.polynomial.Polynomial | numpy.ndarray
        Coefficients of the polynomial, or Numpy object encoding it.
    domain : Interval
        Interval of definition for the monomial variable.
    first : bool, default = False
        Where to contract the coefficients (beginning or end).
    strategy : Strategy, default = DEFAULT_STRATEGY
        Simplification strategy of the MPS, if desired.

    Returns
    -------
    p : MPS
        MPS encoding of the polynomial function
    """
    if not isinstance(p, Polynomial):
        p = Polynomial(p)
    xm_mps = _mps_x_tensor(p.degree(), domain, first)
    if first:
        A = xm_mps[0]
        A = np.einsum("a,aib", p.coef, A).reshape(1, A.shape[1], A.shape[2])
        xm_mps[0] = A
    else:
        A = xm_mps[-1]
        A = np.einsum("aib,b", A, p.coef).reshape(A.shape[0], A.shape[1], 1)
        xm_mps[-1] = A
    if strategy.get_simplify_flag():
        return simplify(xm_mps, strategy=strategy)
    return xm_mps
