from __future__ import annotations
import numpy as np
from typing import Callable, Optional
from abc import ABC, abstractmethod

from ...state import MPS, MPSSum, CanonicalMPS, Strategy, DEFAULT_STRATEGY, simplify
from ...operators import MPO, MPOList, MPOSum, simplify_mpo
from ...typing import Vector
from ...tools import make_logger
from ..mesh import Interval
from ..factories import mps_interval, mps_affine
from ..operators import mpo_affine


ScalarFunction = Callable[[Vector], float]


class PolynomialExpansion(ABC):
    """Abstract base class for polynomial expansions of a function f(x).

    A polynomial expansion is defined by coefficients in a chosen basis
    {P_k(x)} and by the recurrence relation that generates the basis.
    Subclasses must provide:

    - the canonical domain of the basis (e.g. `(-1, 1)` for Chebyshev/Legendre),

    - the three-term recurrence coefficients (α_k, β_k, γ_k),

    - the scaling factor κ for P₁(x) = κ·x.

    Attributes
    ----------
    coeffs : Vector
        Expansion coefficients of f(x) in the chosen basis.
    domain : tuple[float, float]
        Interval [a, b] where the expansion is defined.
    canonical_domain : tuple[float, float]
        Canonical interval of the basis polynomials.
    """

    canonical_domain: tuple[float, float]

    def __init__(self, coeffs: Vector, domain: tuple[float, float]):
        self.coeffs = coeffs
        self.domain = domain

    @abstractmethod
    def get_recurrence(self, k: int) -> tuple[float, float, float]:
        """
        Return the three-term recurrence coefficients (α_k, β_k, γ_k) for
        P_{k+1}(x) = (α_k x + β_k) P_k(x) - γ_k P_{k-1}(x).
        """
        ...

    @property
    @abstractmethod
    def p1_factor(self) -> float:
        """
        Return the scalar κ such that the first-degree basis polynomial satisfies
        P_1(x) = κ·x. Used to correctly seed the three-term recurrence relation.
        """
        ...

    def to_mps(
        self,
        initial: Interval | MPS,
        clenshaw: bool = True,
        strategy: Strategy = DEFAULT_STRATEGY,
        rescale: bool = True,
    ) -> MPS:
        """
        Construct the MPS representation of a composed function using a general
        orthogonal polynomial expansion.

        Given a orthogonal polynomial expansion of a function `f(x)` (e.g. in a basis of
        Chebyshev, Legendre, or other orthogonal polynomials), and an initial representation
        of `g(x)` as an `Interval` or `MPS`, this routine builds an MPS approximation of the
        composition `f(g(x))`.

        The construction can be performed either via the Clenshaw recurrence or
        by direct evaluation of the polynomial series. If `rescale=True`, the
        input `initial` is mapped to the canonical domain of the polynomial
        family (e.g. `[-1, 1]` for Chebyshev/Legendre) before applying the
        expansion.

        Parameters
        ----------
        expansion : PolynomialExpansion
            The polynomial expansion object (e.g. Power series, Chebyshev,
            Legendre, etc.) encoding the coefficients of `f(x)`.
        initial : Interval or MPS
            The initial function `g(x)`, given either as an interval (from which
            an MPS is built) or as an existing MPS.
        clenshaw : bool, default=True
            Whether to use the Clenshaw recurrence for polynomial evaluation
            (recommended for stability).
        strategy : Strategy, default=DEFAULT_STRATEGY
            Simplification strategy for intermediate MPS operations.
        rescale : bool, default=True
            Whether to rescale `initial` to the canonical domain of the chosen
            polynomial basis.

        Returns
        -------
        MPS
            An MPS approximation of the composed function `f(g(x))`.

        Notes
        -----
        - Efficiency depends on the bond dimensions of the intermediate MPS
          states and the chosen simplification strategy.

        - Clenshaw recurrence is generally more efficient and numerically stable,
          though overestimating the expansion order can degrade performance.

        Examples
        --------
        .. code-block:: python

            # Expand a Gaussian using Chebyshev polynomials and load it into an MPS
            func = lambda x: np.exp(-x**2)
            coeffs = interpolation_coefficients(func, start=-1, stop=1)
            expansion = Chebyshev(coeffs)
            domain = RegularInterval(-1, 1, 2**10)
            mps = expansion.to_mps(domain)
        """
        return _mps_polynomial_expansion(self, initial, clenshaw, strategy, rescale)

    def to_mpo(
        self,
        initial: MPO,
        clenshaw: bool = True,
        strategy: Strategy = DEFAULT_STRATEGY,
        rescale: bool = True,
    ) -> MPO:
        """
        Construct the MPO representation of a composed operator using a general
        orthogonal polynomial expansion.

        Given a orthogonal polynomial expansion of a function `f(x)` (in Chebyshev, Legendre,
        Hermite, or another polynomial basis) and an initial operator `A` represented
        as an MPO, this routine builds an MPO approximation of `f(A)`.

        The expansion can be evaluated using the Clenshaw recurrence (recommended
        for stability) or by direct series evaluation. If `rescale=True`, the input
        MPO is mapped to the canonical domain of the polynomial family (e.g. `[-1, 1]`
        for Chebyshev/Legendre) before applying the expansion.

        Parameters
        ----------
        expansion : PolynomialExpansion
            The polynomial expansion object (Chebyshev, Legendre, etc.)
            encoding the coefficients of `f(x)`.
        initial : MPO
            The operator `A` to which the expansion is applied, given as an MPO.
        clenshaw : bool, default=True
            Whether to use the Clenshaw recurrence for polynomial evaluation.
        strategy : Strategy, default=DEFAULT_STRATEGY
            Simplification strategy for intermediate MPO operations.
        rescale : bool, default=True
            Whether to rescale the initial MPO to the canonical domain of the
            chosen polynomial basis.

        Returns
        -------
        MPO
            An MPO approximation of the operator function `f(A)`.
        """
        return _mpo_polynomial_expansion(self, initial, clenshaw, strategy, rescale)


class PowerExpansion(PolynomialExpansion):
    """
    Polynomial expansion in the power basis {1, x, x^2, ...}.

    The canonical domain is [-1, 1]. The recurrence relation is trivial:

        P_{k+1}(x) = x · P_k(x).

    This is equivalent to a standard Taylor/power series expansion. When combined with
    Clenshaw evaluations, this enables the evaluation following Horner's method, which
    is more efficient and stable than evaluating naive monomials x^i directly.
    """

    canonical_domain = (-1, 1)

    def get_recurrence(self, k: int) -> tuple[float, float, float]:
        return (1.0, 0.0, 0.0)

    @property
    def p1_factor(self):
        return 1.0


class OrthogonalExpansion(PolynomialExpansion, ABC):
    """
    Polynomial expansion in an orthogonal polynomial basis.

    Represents expansions in families such as Chebyshev, Legendre,
    Hermite, or Gegenbauer, where the recurrence coefficients are
    determined by orthogonality relations on the canonical domain.
    """

    @classmethod
    @abstractmethod
    def project(
        cls,
        func: ScalarFunction,
        start: float,
        stop: float,
        order: Optional[int] = None,
    ) -> PolynomialExpansion:
        """Project `func` defined on (`start`, `stop`) onto the orthogonal polynomial basis up to the given `order`."""
        ...

    @classmethod
    def estimate_order(
        cls,
        func: ScalarFunction,
        start: float,
        stop: float,
        tol: float = 100 * float(np.finfo(np.float64).eps),
        initial_order: int = 2,
        max_order: int = 2**12,  # 4096
        **kwargs,
    ) -> int:
        """Estimate order of orthogonal polynomial expansion.

        Relies on `cls.project` to build the coefficients, iterating
        until the absolute value of the next coefficient lays below
        the given tolerance.
        """
        order = initial_order
        while order <= max_order:
            # Build the expansion at this trial order
            expansion = cls.project(func, start, stop, order, **kwargs)
            c = expansion.coeffs
            pairs = np.maximum(np.abs(c[0::2]), np.abs(c[1::2]))
            idx = np.where(pairs < tol)[0]
            if idx.size > 0 and idx[0] != 0:
                return 2 * idx[0] + 1
            order *= 2
        raise ValueError("Order exceeds max_order without achieving tolerance.")


def _mps_polynomial_expansion(
    expansion: PolynomialExpansion,
    initial: Interval | MPS,
    clenshaw: bool = True,
    strategy: Strategy = DEFAULT_STRATEGY,
    rescale: bool = True,
) -> MPS:
    if isinstance(initial, Interval):
        initial_mps = mps_interval(initial)
    elif isinstance(initial, MPS):
        initial_mps = initial
    else:
        raise ValueError("Either an Interval or an initial MPS must be provided.")

    if rescale:
        orig = expansion.domain
        dest = expansion.canonical_domain
        initial_mps = mps_affine(initial_mps, orig, dest)

    I = MPS([np.ones((1, s, 1)) for s in initial_mps.physical_dimensions()])
    I_norm = I.norm()
    normalized_I = CanonicalMPS(I, center=0, normalize=True, strategy=strategy)

    x_norm = initial_mps.norm()
    normalized_x = CanonicalMPS(
        initial_mps, center=0, normalize=True, strategy=strategy
    )
    kappa = expansion.p1_factor

    c = expansion.coeffs
    steps = len(c)
    logger = make_logger(2)
    recurrences = [expansion.get_recurrence(l) for l in range(steps + 1)]

    if clenshaw:
        # Recurrence rules:
        # y_k = c_k + (α_k x + β_k I) y_{k+1} - γ_{k+1} y_{k+2}
        # f = y_0 + [(1 - α_0) x + β_0)] y_1
        logger("MPS Clenshaw evaluation started")
        y_k = y_k_plus_1 = normalized_I.zero_state()

        # Main loop
        for k, c_k in enumerate(reversed(c)):
            y_k_plus_1, y_k_plus_2 = y_k, y_k_plus_1

            # Since we reversed(c), loop index k=0 corresponds to c_{d-1} (highest degree).
            # Thus, l = (d - 1) - k gives the true degree index.
            l = (steps - 1) - k
            α_k, β_k, _ = recurrences[l]
            _, _, γ_k_plus_1 = recurrences[l + 1]

            # Avoid the zero branch when β_k == 0
            weights = [c_k * I_norm, α_k * x_norm, -γ_k_plus_1]
            states = [normalized_I, normalized_x * y_k_plus_1, y_k_plus_2]
            if β_k != 0:
                weights.append(β_k * I_norm)
                states.append(normalized_I * y_k_plus_1)
            y_k = simplify(MPSSum(weights, states, check_args=False), strategy=strategy)
            logger(
                f"MPS Clenshaw step {k + 1}/{steps}, maxbond={y_k.max_bond_dimension()}, error={y_k.error():6e}"
            )

        α_0, β_0, _ = recurrences[0]
        weights = [1.0, (1 - α_0 / kappa) * x_norm]
        states = [y_k, normalized_x * y_k_plus_1]
        if β_0 != 0:
            weights.append(-β_0 * I_norm)
            states.append(normalized_I * y_k_plus_1)
        f_mps = simplify(MPSSum(weights, states, check_args=False), strategy=strategy)

    else:
        # Recurrence rules:
        # f_2 = c_0 + c_1 x
        # f_{k+1} = f_k + c_{k+1} T_{k+1}
        # T_{k+1} = (α_{k} x + β_{k}) T_k - γ_k T_{k-1}
        logger("MPS expansion (direct) started")
        f_mps = simplify(
            MPSSum(
                weights=[c[0] * I_norm, c[1] * x_norm * kappa],
                states=[normalized_I, normalized_x],
                check_args=False,
            ),
            strategy=strategy,
        )
        T_k_minus_1, T_k = I_norm * normalized_I, x_norm * kappa * normalized_x
        for k, c_k in enumerate(c[2:], start=2):
            α_k, β_k, γ_k = recurrences[k - 1]
            weights = [α_k * x_norm, -γ_k]
            states = [normalized_x * T_k, T_k_minus_1]
            if β_k != 0:
                weights.append(β_k * I_norm)
                states.append(normalized_I * T_k)

            T_k_plus_1 = simplify(
                MPSSum(weights, states, check_args=False), strategy=strategy
            )
            f_mps = simplify(
                MPSSum(
                    weights=[1.0, c_k], states=[f_mps, T_k_plus_1], check_args=False
                ),
                strategy=strategy,
            )
            logger(
                f"MPS expansion step {k + 1}/{steps}, maxbond={f_mps.max_bond_dimension()}, error={f_mps.error():6e}"
            )
            T_k_minus_1, T_k = T_k, T_k_plus_1

    logger.close()
    return f_mps


def _mpo_polynomial_expansion(
    expansion: PolynomialExpansion,
    initial: MPO,
    clenshaw: bool = True,
    strategy: Strategy = DEFAULT_STRATEGY,
    rescale: bool = True,
) -> MPO:
    if rescale:
        orig = expansion.domain
        dest = expansion.canonical_domain
        initial_mpo = mpo_affine(initial, orig, dest)
    else:
        initial_mpo = initial

    c = expansion.coeffs
    kappa = expansion.p1_factor
    steps = len(c)
    I = MPO([np.eye(2).reshape(1, 2, 2, 1)] * len(initial_mpo))
    logger = make_logger(1)
    recurrences = [expansion.get_recurrence(l) for l in range(steps + 1)]

    if clenshaw:
        logger("MPO Clenshaw evaluation started")
        y_k = y_k_plus_1 = MPO([np.zeros((1, 2, 2, 1))] * len(initial_mpo))

        # Main loop
        for k, c_k in enumerate(reversed(c)):
            y_k_plus_1, y_k_plus_2 = y_k, y_k_plus_1

            l = (steps - 1) - k
            α_k, β_k, _ = recurrences[l]
            _, _, γ_k_plus_1 = recurrences[l + 1]

            weights = [c_k, α_k, -γ_k_plus_1]
            mpos: list[MPO | MPOList] = [
                I,
                MPOList([initial_mpo, y_k_plus_1]),
                y_k_plus_2,
            ]
            if β_k != 0:
                weights.append(β_k)
                mpos.append(MPOList([I, y_k_plus_1]))
            y_k = simplify_mpo(MPOSum(mpos, weights), strategy=strategy)
            logger(
                f"MPO Clenshaw step {k + 1}/{steps}, maxbond={y_k.max_bond_dimension()}"
            )

        α_0, β_0, _ = recurrences[0]
        weights = [1.0, (1 - α_0 / kappa)]
        mpos = [y_k, MPOList([initial_mpo, y_k_plus_1])]
        if β_0 != 0:
            weights.append(-β_0)
            mpos.append(MPOList([I, y_k_plus_1]))
        f_mpo = simplify_mpo(MPOSum(mpos, weights), strategy=strategy)

    else:
        logger("MPO expansion (direct) started")
        f_mpo = simplify_mpo(
            MPOSum(mpos=[I, kappa * initial_mpo], weights=[c[0], c[1]]),
            strategy=strategy,
        )
        T_k_minus_1, T_k = I, kappa * initial_mpo
        for k, c_k in enumerate(c[2:], start=2):
            α_k, β_k, γ_k = recurrences[k - 1]
            weights = [α_k, -γ_k]
            mpos = [MPOList([initial_mpo, T_k]), T_k_minus_1]
            if β_k != 0:
                weights.append(β_k)
                mpos.append(MPOList([I, T_k]))

            T_k_plus_1 = simplify_mpo(MPOSum(mpos, weights), strategy=strategy)
            f_mpo = simplify_mpo(
                MPOSum(mpos=[f_mpo, T_k_plus_1], weights=[1.0, c_k]), strategy=strategy
            )
            logger(
                f"MPO expansion step {k + 1}/{steps}, maxbond={f_mpo.max_bond_dimension()}"
            )
            T_k_minus_1, T_k = T_k, T_k_plus_1

    logger.close()
    return f_mpo
