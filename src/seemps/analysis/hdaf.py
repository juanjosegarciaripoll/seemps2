from math import sqrt, log, ceil
import numpy as np
from scipy.special import loggamma  # type: ignore
from typing import cast
from collections.abc import Iterator
from ..operators import MPO
from ..state import Strategy, DEFAULT_STRATEGY
from ..register.transforms import mpo_weighted_shifts
from ..typing import FloatOrArray, Float


def auto_sigma(
    M: int, dx: Float, time: Float = 0.0, lower_bound: Float | None = None
) -> Float:
    if lower_bound is None:
        lower_bound = 3 * dx
    s0: Float = hdaf_kernel(0, dx=dx, s0=1.0, M=M)
    return max(float(lower_bound), sqrt(time), s0)  # type: ignore


def _asymptotic_factor(M: int, l: int, c: Float) -> Float:
    """
    Equivalent to sqrt(factorial(M + l)) * c ** (M // 2) / factorial(M // 2).
    This is a helper to allow computing 'width_bound' even for very large M.
    """
    Mhalf = M // 2
    return np.exp(0.5 * loggamma(M + l + 1) + Mhalf * np.log(c) - loggamma(Mhalf + 1))


def width_bound(s0: Float, M: int, l: int, time: Float, eps: Float = 1e-16) -> Float:
    """
    Analytic upper bound for the width of the HDAF with the same given
    parameters.

    Returns a value xb such that |HDAF(x)| < eps, for |x| >= xb.
    """
    abs_st = (s0**4 + time**2) ** 0.25

    A = 0.5 / (s0**2 + (time / s0) ** 2)
    B = -sqrt(M + l) / abs_st

    chi = _asymptotic_factor(M, l, c=0.5 * (s0 / abs_st) ** 2)
    chi /= sqrt(2 * np.pi) * abs_st ** (l + 1)

    C = log(eps / chi)

    return 0.5 * (-B + sqrt(B**2 - 4 * A * C)) / A


def _hnl(
    x: np.ndarray,
    c: Float = 1.0,
    l: int = 0,
    d: Float = 1.0,
) -> Iterator[np.ndarray]:
    """
    Generator for H(2n + l, x) * c**n * d / factorial(n), without explicitly
    computing powers nor factorials, where H(n, x) is the n-th Hermite
    polynomial on x.
    """

    n = 0
    hn = np.ones(x.shape) * d  # H0 with appropriate shape
    gn = 2 * x * d  # H1

    # Compute initial values
    for k in range(1, l + 1):
        temp = 2 * x * gn - 2 * k * hn
        gn, hn = temp, gn  # H(l+1, x), H(l, x)

    yield hn

    # Main recurrence relations
    while True:
        n += 1

        hn = (x * gn - (2 * n - 1 + l) * hn) * 2 * c / n  # ~ H(2n + l, x)
        gn = 2 * x * hn - 2 * c * (2 + l / n) * gn  # ~ H(2n + 1 + l, x)

        yield hn


def hdaf_kernel(
    x: FloatOrArray,
    dx: Float,
    s0: Float,
    M: int,
    time: Float = 0.0,
    derivative: int = 0,
) -> FloatOrArray:
    if time == 0:  # Spread under the free propagator
        st = s0
    else:
        st = np.sqrt(s0**2 + 1j * time)

    y = np.asarray(x) / (st * np.sqrt(2))

    const = np.exp(-(y**2)) * dx
    const /= np.sqrt(2 * np.pi) * st * (-1 * np.sqrt(2) * st) ** derivative
    gen = _hnl(y, c=-((0.5 * s0 / st) ** 2), l=derivative, d=const)

    return cast(FloatOrArray, sum(next(gen) for _ in range(int(M) // 2 + 1)))


def hdaf_mpo(
    num_qubits: int,
    dx: Float,
    M: int,
    s0: Float | None = None,
    time: Float = 0.0,
    derivative: int = 0,
    periodic: bool = True,
    strategy: Strategy = DEFAULT_STRATEGY,
) -> MPO:
    """
    Constructs a Matrix Product Operator (MPO) of Hermite Distributed
    Approximating Functionals (HDAFs). The operator may approximate the
    identity, a derivative or the free propagator, depending on the values
    of the `derivative` and `time` parameters.

    Parameters
    ----------
    num_qubits : int
        The number of qubits to discretize the system.
    dx : Float
        The grid stepsize.
    M : int
        The order of the highest Hermite polynomial (must be an even integer).
    s0 : Float | None, default=None
        The width of the HDAF Gaussian weight. If not provided, a suitable
        width will be computed based on `M` and `dx`.
    time : Float, default=0.0
        The evolution time of the Free Propagator to approximate.
    derivative : int, default=0
        The order of the derivative to approximate.
    periodic : bool, default=True
        Whether the grid follows perioidic boundary conditions.
    strategy : Strategy, default=DEFAULT_STRATEGY
        The strategy for the returned MPO. Values of the HDAF below the
        simplification tolerance of the strategy will be discarded.

    Returns
    -------
    mpo: MPO
        The HDAF approximation to an operator specified by the input parameters.
    """

    # Compute width if not provided
    if s0 is None:
        s0 = auto_sigma(M=M, dx=dx, time=time)

    # Threshold of values to discard
    tol = strategy.get_simplification_tolerance()

    # Make kernel vector
    num_elems = ceil(width_bound(s0=s0, M=M, l=derivative, time=time, eps=tol) / dx)
    num_elems = min(num_elems, 2 ** (num_qubits - 1))
    pos_half = dx * np.arange(num_elems)
    hdaf_vec_pos = hdaf_kernel(pos_half, dx, s0, M, time, derivative)
    hdaf_vec_neg = hdaf_vec_pos[:0:-1] * (-1) ** derivative

    # Diagonals and values
    shifts_pos = np.where(np.abs(hdaf_vec_pos) > tol)[0]
    shifts_neg = np.where(np.abs(hdaf_vec_neg) > tol)[0]

    shifts = np.r_[shifts_neg - len(hdaf_vec_neg), shifts_pos]
    weights = np.r_[hdaf_vec_neg[shifts_neg], hdaf_vec_pos[shifts_pos]]

    # Form MPO
    mpo = mpo_weighted_shifts(num_qubits, weights, shifts, periodic=periodic)
    mpo.strategy = strategy

    return mpo
