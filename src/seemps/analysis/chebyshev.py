from __future__ import annotations
from typing import Callable, Optional

import numpy as np
from scipy.fft import dct  # type: ignore

from .mesh import ChebyshevZerosInterval, Interval
from .factories import mps_interval
from ..operators import MPO
from ..state import MPS, Strategy, Truncation, Simplification
from ..truncate import simplify


def chebyshev_coefficients(
    f: Callable,
    order: int,
    start: float = -1,
    stop: float = +1,
    domain: Optional[Interval] = None,
) -> np.polynomial.Chebyshev:
    """
    Returns the Chebyshev coefficients for a given function on a specified
    interval using the Discrete Cosine Transform (DCT II).

    The error of the Chebyshev approximation is related to the magnitude of the
    last (or second-to-last) coefficient of the series. As a rule of thumb, the
    error is given by the sum of all the neglected coefficients, which can be
    close to the magnitude of the last coefficient if the series decays sufficiently
    fast (for example, exponentially fast for analytical functions).

    Parameters
    ----------
    f : Callable
        The target function to approximate with Chebyshev polynomials.
    order : int
        The number of Chebyshev coefficients to compute.
    domain : Optional[Interval], default = None
        The domain on which the function is defined and in which the approximation
        is desired.
    start : float, default = -1
    stop : float, default = +1
        Alternative way to specify the function's domain.

    Returns
    -------
    coefficients : `numpy.polynomial.Chebyshev`
        An array of Chebyshev coefficients scaled to the specified interval.
    """
    if domain is not None:
        start, stop = domain.start, domain.stop
    chebyshev_zeros = np.flip(ChebyshevZerosInterval(start, stop, order).to_vector())
    coefficients = dct(f(chebyshev_zeros), type=2) / order
    coefficients[0] /= 2
    return np.polynomial.Chebyshev(coefficients, domain=(start, stop))


# TODO: All the tests have been done using the `RELATIVE_SINGULAR_VALUE`
# truncation method, which is kind of flunky, because it does not measure
# the actual error. This should be migrated to DEFAULT_STRATEGY, maybe
# strengthening the tolerance
DEFAULT_CHEBYSHEV_STRATEGY = Strategy(
    method=Truncation.RELATIVE_SINGULAR_VALUE,
    tolerance=1e-8,
    simplify=Simplification.VARIATIONAL,
    simplification_tolerance=1e-8,
    normalize=False,
)


def _interval_map(orig, dest):
    """Compute the affine transformation between two intervals.

    Map the original domain [x0, x1] to [u0, u1]
    This is a transformation u = a * x + b
    with u0 = a * x0 + b
    and  u1 = a * x1 + b
    Hence a = (u1 - u0) / (x1 - x0)
          b = ((u1 + u0) - a * (x0 + x1)) / 2
    """
    x0, x1 = orig
    u0, u1 = dest
    a = (u1 - u0) / (x1 - x0)
    b = 0.5 * ((u1 + u0) - a * (x0 + x1))
    return a, b


def cheb2mps(
    c: np.polynomial.Chebyshev,
    domain: Optional[Interval] = None,
    x: Optional[MPS] = None,
    strategy: Strategy = DEFAULT_CHEBYSHEV_STRATEGY,
) -> MPS:
    """
    Construct an MPS representation of a function, from a Chebyshev expansion.

    This function takes as input an MPS representation of the first order
    polynomial `x` in a given `domain`, with values `[x0,x1]`. It also takes
    a Chebyshev expansion `c` of a function `c(x)` defined in a domain that
    contains this interval `[x0,x1]`. With this information, it constructs
    the MPS that approximates `c(x)`.

    Parameters
    ----------
    c : `numpy.polynomial.Chebyshev`
        Chebyshev expansion over a given domain.
    domain : Optional[Interval], default = None
        Interval of definition for the function, which must be contained in the
        Chebyshev's series domain.
    x : Optional[MPS], default = None
        MPS representation of the `x` function in the series' domain. It will
        be computed from `domain` if not provided.
    strategy : Strategy, default = DEFAULT_CHEBYSHEV_STRATEGY
        Simplification strategy for operations between MPS.

    Returns
    -------
    f : MPS
        MPS representation of the polynomial expansion.
    """
    x_mps: MPS
    if domain is not None:
        x_mps = mps_interval(domain.map_to(-1, 1))
        L = len(x_mps)
        I = MPS([np.ones((1, 2, 1))] * L)
    elif isinstance(x, MPS):
        # We assume that `x` is contained in the domain (x0, x1)
        # of the Chebyshev expansion
        (x0, x1), _ = c.linspace(2)
        # We map the MPS operator to the (-1, 1) domain using
        # a linear transformation.
        a, b = _interval_map((x0, x1), (-1, 1))
        x_mps = a * x
        L = len(x_mps)
        I = MPS([np.ones((1, 2, 1))] * L)
        if np.abs(b) > np.finfo(np.float64).eps:
            x_mps = (x_mps + b * I).join()
    else:
        raise Exception("In cheb2mps, either domain or an MPS must be provided.")

    coef = c.coef
    y = [MPS([np.zeros((1, 2, 1))] * len(x_mps))] * (len(coef) + 2)
    for i in range(len(y) - 3, -1, -1):
        y[i] = simplify(
            coef[i] * I - y[i + 2] + 2 * x_mps * y[i + 1],
            strategy=strategy,
        )
    return simplify(y[0] - x_mps * y[1], strategy=strategy)


def cheb2mpo(
    c: np.polynomial.Chebyshev,
    domain: Optional[Interval] = None,
    x: Optional[MPO] = None,
    strategy: Strategy = DEFAULT_CHEBYSHEV_STRATEGY,
) -> MPO:
    """
    *NOT IMPLEMENTED*
    Construct an MPO representation of a function, from a Chebyshev expansion.

    This function takes as input an MPO representation of the first order
    polynomial `x` in a given `domain`, with values `[x0,x1]`. It also takes
    a Chebyshev expansion `c` of a function `c(x)` defined in a domain that
    contains this interval `[x0,x1]`. With this information, it constructs
    the MPO that approximates `c(x)`.

    Parameters
    ----------
    c : `numpy.polynomial.Chebyshev`
        Chebyshev expansion over a given domain.
    domain : Optional[Interval], default = None
        Interval of definition for the function, whose domain must be included
        in that of `c`.
    x : Optional[MPO], default = None
        MPS representation of the `x` function in the series' domain. It will
        be computed from `domain` if not provided.
    strategy : Strategy, default = DEFAULT_CHEBYSHEV_STRATEGY
        Simplification strategy for operations between MPOs.

    Returns
    -------
    f : MPO
        MPO representation of the polynomial expansion.
    """
    raise Exception("cheb2mpo not implemented")


def chebyshev_approximation(
    f: Callable,
    order: int,
    domain: Interval,
    differentiation_order: int = 0,
    strategy: Strategy = DEFAULT_CHEBYSHEV_STRATEGY,
) -> MPS:
    """
    Load a function as an MPS using Chebyshev expansions.

    This function constructs a Chebyshev series that approximates `f` over
    the given `domain`, and uses that expansion to construct an MPS
    representation via `cheb2mps`.

    Parameters
    ----------
    f : Callable
        A univariate scalar function.
    order : int
        Order of the Chebyshev expansion
    domain : Interval
        The domain over which the function is to be approximated.
    differentiation_order : int, default = 0
        If positive or negative value `N`, integrate or differentiate the
        function a total of `abs(N)` times prior to computing the expansion.
    strategy : Strategy, default Strategy()
        The strategy used for simplifying the MPS or MPO during computation.

    Returns
    -------
    mps : MPS
        MPS approximation to the function.
    """
    c = chebyshev_coefficients(f, order, domain.start, domain.stop)
    if differentiation_order < 0:
        c = c.integ(-differentiation_order, lbnd=domain.start)
    elif differentiation_order > 0:
        c = c.deriv(differentiation_order)
    return cheb2mps(c, domain, strategy=strategy)
