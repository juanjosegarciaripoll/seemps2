from __future__ import annotations
from typing import Callable, Literal
from math import sqrt
import numpy as np
from numpy.typing import NDArray
from scipy.fft import dct  # type: ignore

from ..tools import make_logger
from ..state import CanonicalMPS, MPS, MPSSum, Strategy, DEFAULT_STRATEGY
from ..truncate import simplify
from ..truncate.simplify_mpo import simplify_mpo
from ..operators import MPO, MPOList, MPOSum
from .mesh import (
    Interval,
    ChebyshevInterval,
    array_affine,
)
from .operators import mpo_affine
from .factories import mps_interval, mps_affine


def interpolation_coefficients(
    func: Callable[[NDArray], NDArray],
    order: int | None = None,
    start: float = -1.0,
    stop: float = +1.0,
    domain: Interval | None = None,
    interpolated_nodes: Literal["zeros", "extrema"] = "zeros",
) -> np.polynomial.Chebyshev:
    """
    Returns the coefficients for the Chebyshev interpolation of a function on a given set
    of nodes and on a specified interval.

    Parameters
    ----------
    func : Callable
        The target function to approximate with Chebyshev polynomials.
    order : int, optional
        The number of Chebyshev coefficients to compute.
        If None, estimates an order that results in an error below machine precision.
    domain : Interval, optional
        The domain on which the function is defined and in which the approximation
        is desired.
    start : float, default=-1.0
    stop : float, default=+1.0
        Alternative way to specify the function's domain.
    interpolated_nodes : str, default = "zeros"
        The nodes on which the function is interpolated. Use "zeros" for
        Chebyshev zeros or "extrema" for Chebyshev extrema.

    Returns
    -------
    coefficients : `numpy.polynomial.Chebyshev`
        An array of Chebyshev coefficients scaled to the specified interval.
    """
    if order is None:
        order = estimate_order(func, start, stop, domain)
    if domain is not None:
        start, stop = domain.start, domain.stop
    match interpolated_nodes:
        case "zeros":
            nodes: NDArray = ChebyshevInterval(start, stop, order).to_vector()
            coefficients = (1 / order) * dct(np.flip(func(nodes)), type=2)
        case "extrema":
            nodes = ChebyshevInterval(start, stop, order, endpoints=True).to_vector()
            coefficients = 2.0 * dct(np.flip(func(nodes)), type=1, norm="forward")
        case _:
            raise TypeError("interpolated_nodes is not one of zeros | extrema")
    coefficients[0] /= 2
    return np.polynomial.Chebyshev(coefficients, domain=(start, stop))


def projection_coefficients(
    func: Callable,
    order: int | None = None,
    start: float = -1.0,
    stop: float = +1.0,
    domain: Interval | None = None,
) -> np.polynomial.Chebyshev:
    """
    Returns the coefficients for the Chebyshev projection of a function using
    Chebyshev-Gauss integration.

    Parameters
    ----------
    func : Callable
        The target function to approximate with Chebyshev polynomials.
    order : int, optional
        The number of Chebyshev projection coefficients to compute.
        If None, estimates an order that results in an error below machine precision.
    start : float, default=-1.0
    stop : float, default=+1.0
        The domain on which the function is defined and in which the approximation is desired.
    domain : Interval, optional
        Alternative way to specify the function's domain.

    Returns
    -------
    coefficients : `numpy.polynomial.Chebyshev`
    An array of Chebyshev coefficients scaled to the specified interval.
    """
    if order is None:
        order = estimate_order(func, start, stop, domain)
    if domain is not None:
        start, stop = domain.start, domain.stop
    quad_order = order  # TODO: Check if this order integrates to machine precision
    nodes = np.cos(np.pi * np.arange(1, 2 * quad_order, 2) / (2.0 * quad_order))
    nodes_affine = array_affine(nodes, orig=(-1, 1), dest=(start, stop))
    weights = np.ones(quad_order) * (np.pi / quad_order)
    T_matrix = np.cos(np.outer(np.arange(order), np.arccos(nodes)))
    coefficients = (2 / np.pi) * (T_matrix * func(nodes_affine)) @ weights
    coefficients[0] /= 2
    return np.polynomial.Chebyshev(coefficients, domain=(start, stop))


def estimate_order(
    func: Callable,
    start: float = -1,
    stop: float = +1,
    domain: Interval | None = None,
    tolerance: float = float(np.finfo(np.float64).eps),
    initial_order: int = 2,
    max_order: int = 2**12,  # 4096
) -> int:
    """
    Returns an estimation of the number of Chebyshev coefficients required to achieve a
    given accuracy such that the last pair of coefficients fall below a given tolerance,
    as they theoretically bound the maximum error of the expansion.

    Notes
    -----
    - The coefficients are evaluated in pairs because even and odd functions respectively
    have vanishing even and odd coefficients.
    """
    if domain is not None:
        start, stop = domain.start, domain.stop
    order = initial_order
    while order <= max_order:
        c = projection_coefficients(func, order, start, stop).coef
        max_c_in_pairs = np.maximum(np.abs(c[::2]), np.abs(c[1::2]))
        c_below_tolerance = np.where(max_c_in_pairs < tolerance)[0]
        if c_below_tolerance.size > 0 and c_below_tolerance[0] != 0:
            return 2 * c_below_tolerance[0] + 1
        order *= 2
    raise ValueError("Order exceeds max_order without achieving tolerance.")


def cheb2mps(
    coefficients: np.polynomial.Chebyshev,
    initial_mps: MPS | None = None,
    domain: Interval | None = None,
    strategy: Strategy = DEFAULT_STRATEGY,
    clenshaw: bool = True,
    rescale: bool = True,
) -> MPS:
    """
    Composes a function on an initial MPS by expanding it on the basis of Chebyshev polynomials.
    Allows to load functions on MPS by providing a suitable initial MPS for a given interval.
    Takes as input the Chebyshev coefficients of a function `f(x)` defined in an interval `[a, b]`
    and, optionally, an initial MPS representing a function `g(x)` that is taken as the first order
    polynomial of the expansion. With this information, it constructs the MPS that approximates `f(g(x))`.

    Parameters
    ----------
    coefficients : np.polynomial.Chebyshev
        The Chebyshev expansion coefficients representing the target function that
        is defined on a given interval `[a, b]`.
    initial_mps : MPS, optional
        The initial MPS on which to apply the expansion.
        By default (if ``rescale`` is ``True``), it must have a support inside the domain of
        definition of the function `[a, b]`.
        If ``rescale`` is ``False``, it must have a support inside `[-1, 1]`.
    domain : Interval, optional
        An alternative way to specify the initial MPS by constructing it from the given Interval.
    strategy : Strategy, default=DEFAULT_STRATEGY
        The simplification strategy for operations between MPS.
    clenshaw : bool, default=True
        Whether to use the Clenshaw algorithm for polynomial evaluation.
    rescale : bool, default=True
        Whether to perform an affine transformation of the initial MPS from the domain
        `[a, b]` of the Chebyshev coefficients to the canonical Chebyshev interval `[-1, 1]`.

    Returns
    -------
    f_mps : MPS
        MPS representation of the polynomial expansion.

    Notes
    -----
    - The computational complexity of the expansion depends on bond dimensions of the intermediate
    states. For the case of loading univariate functions on `RegularInterval` domains, these are
    bounded by the polynomial order of each intermediate step. In general, these are determined by
    the function, the bond dimensions of the initial state and the simplification strategy used.

    - The Clenshaw evaluation method has a better performance overall, but performs worse when
    the expansion order is overestimated. This can be avoided using the `estimate_order` method.

    Examples
    --------
    .. code-block:: python

        # Load an univariate Gaussian in an equispaced domain.
        start, stop = -1, 1
        n_qubits = 10
        func = lambda x: np.exp(-x**2)
        coefficients = interpolation_coefficients(func, start=start, stop=stop)
        domain = RegularInterval(start, stop, 2**n_qubits)
        mps = cheb2mps(coefficients, domain=domain)
    """
    if isinstance(initial_mps, MPS):
        pass
    elif isinstance(domain, Interval):
        initial_mps = mps_interval(domain)
    else:
        raise ValueError("Either a domain or an initial MPS must be provided.")
    if rescale:
        orig = tuple(coefficients.linspace(2)[0])
        initial_mps = mps_affine(initial_mps, orig, (-1, 1))

    c = coefficients.coef
    I_norm = 2 ** (initial_mps.size / 2)
    normalized_I = CanonicalMPS(
        [np.ones((1, 2, 1)) / sqrt(2.0)] * initial_mps.size,
        center=0,
        is_canonical=True,
    )
    x_norm = initial_mps.norm()
    normalized_x = CanonicalMPS(
        initial_mps, center=0, normalize=True, strategy=strategy
    )
    logger = make_logger(1)
    if clenshaw:
        steps = len(c)
        logger("MPS Clenshaw evaluation started")
        y_i = y_i_plus_1 = normalized_I.zero_state()
        for i, c_i in enumerate(reversed(c)):
            y_i_plus_1, y_i_plus_2 = y_i, y_i_plus_1
            y_i = simplify(
                # coef[i] * I - y[i + 2] + (2 * x_mps) * y[i + 1],
                MPSSum(
                    weights=[c_i * I_norm, -1, 2 * x_norm],
                    states=[normalized_I, y_i_plus_2, normalized_x * y_i_plus_1],
                    check_args=False,
                ),
                strategy=strategy,
            )
            logger(
                f"MPS Clenshaw step {i + 1}/{steps}, maxbond={y_i.max_bond_dimension()}, error={y_i.error():6e}"
            )
        f_mps = simplify(
            MPSSum(
                weights=[1, -x_norm],
                states=[y_i, normalized_x * y_i_plus_1],
                check_args=False,
            ),
            strategy=strategy,
        )
    else:
        steps = len(c)
        logger("MPS Chebyshev expansion started")
        f_mps = simplify(
            MPSSum(
                weights=[c[0] * I_norm, c[1] * x_norm],
                states=[normalized_I, normalized_x],
                check_args=False,
            ),
            strategy=strategy,
        )
        T_i, T_i_plus_1 = I_norm * normalized_I, x_norm * normalized_x
        for i, c_i in enumerate(c[2:], start=2):
            T_i_plus_2 = simplify(
                MPSSum(
                    weights=[2 * x_norm, -1],
                    states=[normalized_x * T_i_plus_1, T_i],
                    check_args=False,
                ),
                strategy=strategy,
            )
            f_mps = simplify(
                MPSSum(weights=[1, c_i], states=[f_mps, T_i_plus_2], check_args=False),
                strategy=strategy,
            )
            logger(
                f"MPS expansion step {i + 1}/{steps}, maxbond={f_mps.max_bond_dimension()}, error={f_mps.error():6e}"
            )
            T_i, T_i_plus_1 = T_i_plus_1, T_i_plus_2
    logger.close()
    return f_mps


def cheb2mpo(
    coefficients: np.polynomial.Chebyshev,
    initial_mpo: MPO,
    strategy: Strategy = DEFAULT_STRATEGY,
    clenshaw: bool = True,
    rescale: bool = True,
) -> MPO:
    """
    Composes a function on an initial MPO by expanding it on the basis of Chebyshev polynomials.

    Parameters
    ----------
    coefficients : np.polynomial.Chebyshev
        The Chebyshev expansion coefficients representing the target function that
        is defined on a given interval `[a, b]`.
    initial_mpo : MPO
        The initial MPO on which to apply the expansion.
        By default (if ``rescale`` is ``True``), it must have a support inside the domain of
        definition of the function `[a, b]`.
        If ``rescale`` is ``False``, it must have a support inside `[-1, 1]`.
    strategy : Strategy, default=DEFAULT_STRATEGY
        The simplification strategy for operations between MPS.
    clenshaw : bool, default=True
        Whether to use the Clenshaw algorithm for polynomial evaluation.
    rescale : bool, default=True
        Whether to perform an affine transformation of the initial MPO from the domain
        `[a, b]` of the Chebyshev coefficients to the canonical Chebyshev interval `[-1, 1]`.

    Returns
    -------
    f_mpo : MPO
        MPO representation of the polynomial expansion.
    """
    if rescale:
        orig = tuple(coefficients.linspace(2)[0])
        initial_mpo = mpo_affine(initial_mpo, orig, (-1, 1))
    c = coefficients.coef
    I = MPO([np.eye(2).reshape(1, 2, 2, 1)] * len(initial_mpo))
    logger = make_logger(1)
    if clenshaw:
        steps = len(c)
        logger("MPO Clenshaw evaluation started")
        y_i = y_i_plus_1 = MPO([np.zeros((1, 2, 2, 1))] * len(initial_mpo))
        for i, c_i in enumerate(reversed(coefficients.coef)):
            y_i_plus_1, y_i_plus_2 = y_i, y_i_plus_1
            y_i = simplify_mpo(
                MPOSum(
                    mpos=[I, y_i_plus_2, MPOList([initial_mpo, y_i_plus_1])],
                    weights=[c_i, -1, 2],
                ),
                strategy=strategy,
            )
            logger(
                f"MPO Clenshaw step {i + 1}/{steps}, maxbond={y_i.max_bond_dimension()}"
            )
        f_mpo = simplify_mpo(
            MPOSum([y_i, MPOList([initial_mpo, y_i_plus_1])], weights=[1, -1]),
            strategy=strategy,
        )
    else:
        steps = len(c)
        logger("MPO Chebyshev expansion started")
        T_i, T_i_plus_1 = I, initial_mpo
        f_mpo = simplify_mpo(
            MPOSum(mpos=[T_i, T_i_plus_1], weights=[c[0], c[1]]),
            strategy=strategy,
        )
        for i, c_i in enumerate(c[2:], start=2):
            T_i_plus_2 = simplify_mpo(
                MPOSum(mpos=[MPOList([initial_mpo, T_i_plus_1]), T_i], weights=[2, -1]),
                strategy=strategy,
            )
            f_mpo = simplify_mpo(
                MPOSum(mpos=[f_mpo, T_i_plus_2], weights=[1, c_i]),
                strategy=strategy,
            )
            logger(
                f"MPO expansion step {i + 1}/{steps}, maxbond={f_mpo.max_bond_dimension()}"
            )
            T_i, T_i_plus_1 = T_i_plus_1, T_i_plus_2
    return f_mpo
