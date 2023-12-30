from typing import Callable, Optional, Union

import numpy as np
from scipy.fft import dct  # type: ignore
from typing import Callable, Optional

from seemps.analysis.mesh import ChebyshevZerosInterval
from seemps.analysis.sampling import infinity_norm
from seemps.operators import MPO
from seemps.state import MPS, Strategy, DEFAULT_STRATEGY, Truncation, Simplification
from seemps.truncate import simplify


def chebyshev_coefficients(
    f: Callable, order: int, start: float, stop: float
) -> np.ndarray:
    """
    Returns the Chebyshev coefficients for a given function on a specified interval using
    the Discrete Cosine Transform (DCT II).

    The accuracy of the Chebyshev approximation is correlated to the magnitude of the last few coefficients
    in the series (depending on their periodicity), with smaller absolute values typically indicating
    a better approximation of the function.

    Parameters
    ----------
    f : Callable
        The target function to approximate with Chebyshev polynomials.
    start : float
        The starting point of the domain of the function.
    stop : float
        The ending point of the domain of the function.
    order : int
        The number of Chebyshev coefficients to compute.

    Returns
    -------
    coefficients : np.ndarray
        An array of Chebyshev coefficients scaled to the specified interval.
    """
    chebyshev_zeros = np.flip(ChebyshevZerosInterval(start, stop, order).to_vector())
    coefficients = dct(f(chebyshev_zeros), type=2) / order
    coefficients[0] /= 2
    return coefficients


def differentiate_chebyshev_coefficients(
    chebyshev_coefficients: np.ndarray, start: float, stop: float
) -> np.ndarray:
    """
    Returns the Chebyshev coefficients of the derivative of a function
    whose Chebyshev coefficients are given.

    Parameters
    ----------
    chebyshev_coefficients : np.ndarray
        Chebyshev coefficients of the original function.
    start : float
        The starting point of the domain of the function.
    stop : float
        The ending point of the domain of the function.

    Returns
    -------
    np.ndarray
        Chebyshev coefficients of the derivative of the original function.
    """
    c = chebyshev_coefficients  # Shorter alias
    N = len(c) - 1
    c_diff = np.zeros_like(c)
    c_diff[N] = 0
    c_diff[N - 1] = 0
    for i in range(N - 2, 0, -1):
        c_diff[i] = 2 * (i + 1) * c[i + 1] + c_diff[i + 2]
    c_diff[0] = (2 * c[1] + c_diff[2]) / 2
    return c_diff * 2 / (stop - start)


def integrate_chebyshev_coefficients(
    chebyshev_coefficients: np.ndarray,
    start: float,
    stop: float,
    integration_constant: Optional[float] = None,
) -> np.ndarray:
    """
    Returns the Chebyshev coefficients of the integral of a function
    whose Chebyshev coefficients are given.

    Parameters
    ----------
    chebyshev_coefficients : np.ndarray
        Chebyshev coefficients of the original function.
    start : float
        The starting point of the domain of the function.
    stop : float
        The ending point of the domain of the function.
    integration_constant : float, optional
        The constant of integration to be added to the zeroth coefficient.

    Returns
    -------
    np.ndarray
        Chebyshev coefficients of the integral of the original function.
    """
    c = chebyshev_coefficients  # Shorter alias
    N = len(c) - 1
    c_intg = np.zeros_like(c)
    c_intg[1] = (2 * c[0] - c[2]) / 2
    for i in range(2, N + 1):
        c_intg[i] = (c[i - 1] - c[i + 1]) / (2 * i) if i < N else c[i - 1] / (2 * i)
    c_intg[0] = c[0] if integration_constant is None else integration_constant
    return c_intg * (stop - start) / 2


DEFAULT_CHEBYSHEV_STRATEGY = Strategy(
    method=Truncation.RELATIVE_SINGULAR_VALUE,
    tolerance=1e-8,
    simplify=Simplification.VARIATIONAL,
    simplification_tolerance=1e-8,
    normalize=False,
)


def chebyshev_approximation(
    f: Callable,
    order: int,
    domain: Union[MPS, MPO],
    domain_norm_inf: Optional[float] = None,
    differentiation_order: int = 0,
    strategy: Strategy = DEFAULT_CHEBYSHEV_STRATEGY,
) -> Union[MPS, MPO]:
    """
    Returns the MPS representation of a function or one of its integrals or derivatives
    by means of the Chebyshev approximation using the Clenshaw algorithm.

    Parameters
    ----------
    f : Callable
        A univariate scalar target function to approximate.
    order : int
        The order of the Chebyshev approximation.
    domain : Union[MPS, MPO]
        The domain over which the function is to be approximated, represented as a MPS or MPO.
    domain_norm_inf : Optional[float], default None
        The infinity norm of the domain, used for scaling the domain and the Chebyshev coefficients.
        If None, it is calculated from the domain.
    differentiation_order : int, default 0
        A parameter n which sets the target function as the n-th derivative (if positive) or
        integral (if negative) of the original function f.
    strategy : Strategy, default Strategy()
        The strategy used for simplifying the MPS or MPO during computation.

    Returns
    -------
    chebyshev_approximation : Union[MPS, MPO]
        The Chebyshev approximation of the function on the domain using the Clenshaw algorithm.
    """
    # Scale domain and coefficients with the infinity norm of the domain.
    if domain_norm_inf is None:
        domain_norm_inf = infinity_norm(domain)
    domain = domain * (1 / domain_norm_inf)
    coefficients = chebyshev_coefficients(f, order, -domain_norm_inf, domain_norm_inf)

    # Differentiate or integrate the coefficients
    for _ in range(abs(differentiation_order)):
        if differentiation_order > 0:
            coefficients = differentiate_chebyshev_coefficients(
                coefficients, -domain_norm_inf, domain_norm_inf
            )
        else:
            coefficients = integrate_chebyshev_coefficients(
                coefficients, -domain_norm_inf, domain_norm_inf
            )

    # Run the Clenshaw algorithm.
    I: Union[MPS, MPO]
    if isinstance(domain, MPS):
        I = MPS([np.ones((1, 2, 1))] * len(domain))
        y = [MPS([np.zeros((1, 2, 1))] * len(domain))] * (len(coefficients) + 2)
        for i in range(len(y) - 3, -1, -1):
            y[i] = simplify(
                coefficients[i] * I - y[i + 2] + 2 * domain * y[i + 1],
                strategy=strategy,
            )
        return simplify(y[0] - domain * y[1], strategy=strategy)
    elif isinstance(domain, MPO):
        raise Exception("chebyshev_approximation not implemented for MPOs")
    else:
        raise Exception("Invalid argument to chebyshev_approximation")
    return chebyshev_approximation
