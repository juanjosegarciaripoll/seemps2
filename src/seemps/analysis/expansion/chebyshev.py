from __future__ import annotations
import numpy as np
from scipy.fft import dct  # type: ignore
from typing import Literal

from ...typing import Vector
from ..mesh import array_affine, ChebyshevInterval
from .expansion import OrthogonalExpansion, ScalarFunction


class ChebyshevExpansion(OrthogonalExpansion):
    r"""
    Expansion in the Chebyshev basis.

    The Chebyshev polynomials :math:`T_k(x)` are orthogonal on the interval
    :math:`[−1, 1]` with weight :math:`1/\sqrt{1−x^2}`. They are widely used
    in approximation theory since truncated Chebyshev series minimize the
    maximum error (near-best polynomial approximation).

    See https://en.wikipedia.org/wiki/Chebyshev_polynomials for more information.
    """

    canonical_domain = (-1, 1)

    def __init__(self, coeffs: Vector, domain: tuple[float, float]):
        super().__init__(coeffs, domain)

    def get_recurrence(self, k: int) -> tuple[float, float, float]:
        """Chebyshev recurrence.

        Returns the three elements of the Chebyshev iteration

        .. math::
           T_{k+1}(x) = 2x T_k(x) - T_{k-1}(x)

        used by Clenshaw's evaluation formula.
        """
        _ = k  # Ignore k
        α_k = 2.0
        β_k = 0.0
        γ_k = 1.0
        return (α_k, β_k, γ_k)

    @property
    def p1_factor(self) -> float:
        return 1.0

    @classmethod
    def project(
        cls,
        func: ScalarFunction,
        start: float = -1.0,
        stop: float = 1.0,
        order: int | None = None,
    ) -> ChebyshevExpansion:
        if order is None:
            order = cls.estimate_order(func, start, stop)
        nodes = np.cos(np.pi * np.arange(1, 2 * order, 2) / (2.0 * order))
        nodes_affine = array_affine(
            nodes, orig=ChebyshevExpansion.canonical_domain, dest=(start, stop)
        )
        weights = np.ones(order) * (np.pi / order)
        T_matrix = np.cos(np.outer(np.arange(order), np.arccos(nodes)))
        coeffs = (2 / np.pi) * (T_matrix * func(nodes_affine)) @ weights
        coeffs[0] /= 2
        return cls(coeffs, domain=(start, stop))

    @classmethod
    def interpolate(
        cls,
        func: ScalarFunction,
        start: float,
        stop: float,
        order: int | None = None,
        nodes: Literal["zeros", "extrema"] = "zeros",
    ) -> ChebyshevExpansion:
        if order is None:
            order = cls.estimate_order(func, start, stop)
        if nodes == "zeros":
            x = ChebyshevInterval(start, stop, order).to_vector()
            coeffs = (1 / order) * dct(np.flip(func(x)), type=2)
        elif nodes == "extrema":
            x = ChebyshevInterval(start, stop, order, endpoints=True).to_vector()
            coeffs = 2 * dct(np.flip(func(x)), type=1, norm="forward")
        coeffs[0] /= 2
        return cls(coeffs, domain=(start, stop))

    def deriv(self, m: int = 1) -> ChebyshevExpansion:
        """Return the m-th derivative as a new ChebyshevExpansion."""
        T = np.polynomial.Chebyshev(self.coeffs, domain=self.domain).deriv(m)
        a, b = map(float, T.domain)  # Keep type checker happy
        return ChebyshevExpansion(T.coef, domain=(a, b))

    def integ(self, m: int = 1, lbnd: float = 0.0) -> ChebyshevExpansion:
        """Return the m-th integral as a new ChebyshevExpansion."""
        T = np.polynomial.Chebyshev(self.coeffs, domain=self.domain).integ(m, lbnd=lbnd)
        a, b = map(float, T.domain)
        return ChebyshevExpansion(T.coef, domain=(a, b))
