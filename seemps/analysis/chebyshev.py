import numpy as np
from scipy.fft import dct
from typing import Callable, Optional

from seemps.state import MPS, Strategy
from seemps.truncate import simplify
from .mesh import ChebyshevZerosInterval


def chebyshev_coefficients(
    f: Callable, start: float, stop: float, order: int
) -> np.ndarray:
    """
    Calculate the Chebyshev coefficients for a given function on a specified interval using
    the Discrete Cosine Transform (DCT II).

    The accuracy of the Chebyshev approximation is correlated to the magnitude of the last few coefficients
    in the series (depending on their periodicity), with smaller absolute values typically indicating
    a better approximation of the function.

    Parameters
    ----------
    f : Callable
        The target function to approximate with Chebyshev polynomials.
    start : float
        The start of the interval over which to compute the coefficients.
    stop : float
        The end of the interval over which to compute the coefficients.
    order : int
        The number of Chebyshev coefficients to compute.

    Returns
    -------
    np.ndarray
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
    Compute the Chebyshev coefficients of the derivative of a function
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
    Compute the Chebyshev coefficients of the integral of a function
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


def chebyshev_approximation_clenshaw(
    chebyshev_coefficients: np.ndarray, mps_0: MPS, strategy: Strategy = Strategy()
) -> MPS:
    """
    Evaluate a Chebyshev MPS series using Clenshaw's algorithm. This method assumes
    that the support of the starting mps_0 is within the range [-1, 1].
    Evaluations outside this range may yield incorrect results due to the properties
    of Chebyshev polynomials which are orthogonal only inside this interval.

    Parameters
    ----------
    chebyshev_coefficients : np.ndarray
        Array of coefficients for the Chebyshev series.
    mps_0
        Initial MPS for the Clenshaw algorithm.
    strategy
        Strategy to be used in the MPS simplification routine.

    Returns
    -------
    MPS
        Resulting MPS of the Chebyshev series evaluation using the Clenshaw algorithm.
    """
    c = np.flip(chebyshev_coefficients)
    I = MPS([np.ones((1, 2, 1))] * len(mps_0))
    y = [MPS([np.zeros((1, 2, 1))] * len(mps_0))] * (len(c) + 2)
    for i in range(2, len(y)):
        y[i] = simplify(
            c[i - 2] * I - y[i - 2] + 2 * mps_0 * y[i - 1], strategy=strategy
        )
    mps = y[-1] - mps_0 * y[-2]
    return simplify(mps, strategy=strategy)
