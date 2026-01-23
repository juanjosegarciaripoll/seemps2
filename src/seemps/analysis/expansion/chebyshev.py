from __future__ import annotations
import numpy as np
from scipy.fft import dct  # type: ignore
from typing import Literal

from ...state import MPS
from ...operators import MPO
from ...typing import Vector
from ..mesh import array_affine, ChebyshevInterval
from ..factories import mps_affine
from ..operators import mpo_affine
from .expansion import PolynomialExpansion, ScalarFunction


class ChebyshevExpansion(PolynomialExpansion):
    r"""
    Expansion in the Chebyshev basis.

    The Chebyshev polynomials :math:`T_k(x)` are orthogonal on the interval
    :math:`[−1, 1]` with weight :math:`1/\sqrt{1−x^2}`. They are widely used
    in approximation theory since truncated Chebyshev series minimize the
    maximum error (near-best polynomial approximation).

    See https://en.wikipedia.org/wiki/Chebyshev_polynomials for more information.
    """

    orthogonality_domain = (-1.0, 1.0)
    affine_fix = (1.0, 0.0)

    def __init__(self, coefficients: Vector, approximation_domain: tuple[float, float]):
        self.approximation_domain = approximation_domain
        super().__init__(coefficients)

    def recurrence_coefficients(self, k: int) -> tuple[float, float, float]:
        """
        Returns the three-term coefficients of the Chebyshev recursion:

        .. math::
           T_{k+1}(x) = 2x T_k(x) - T_{k-1}(x)
        """
        return (2.0, 0.0, 1.0)

    def rescale_mps(self, mps: MPS) -> MPS:
        orig = self.approximation_domain
        dest: tuple[float, float] = self.orthogonality_domain  # pyright: ignore
        return mps_affine(mps, orig, dest)

    def rescale_mpo(self, mpo: MPO) -> MPO:
        orig = self.approximation_domain
        dest: tuple[float, float] = self.orthogonality_domain  # pyright: ignore
        return mpo_affine(mpo, orig, dest)

    @classmethod
    def estimate_order(
        cls,
        func: ScalarFunction,
        approximation_domain: tuple[float, float] = (-1.0, 1.0),
        tol: float = 100 * float(np.finfo(np.float64).eps),
        min_order: int = 2,
        max_order: int = 2**12,  # 4096
    ) -> int:
        order = min_order
        while order <= max_order:
            expansion = cls.project(func, approximation_domain, order)
            c = expansion.coefficients
            pairs = np.maximum(np.abs(c[0::2]), np.abs(c[1::2]))
            idx = np.where(pairs < tol)[0]
            if idx.size > 0 and idx[0] != 0:
                return 2 * idx[0] + 1
            order *= 2
        raise ValueError("Order exceeds max_order without achieving tolerance.")

    @classmethod
    def project(
        cls,
        func: ScalarFunction,
        approximation_domain: tuple[float, float] = (-1.0, 1.0),
        order: int | None = None,
    ) -> ChebyshevExpansion:
        """
        Project a scalar function onto the Chebyshev basis on the given approximation domain.

        The approximation domain must contain the full range of arguments on which the expansion
        will be evaluated; otherwise, rescaling maps the argument outside the orthogonality domain
        where the basis is not defined, leading to large errors.
        """
        if order is None:
            order = cls.estimate_order(func, approximation_domain)
        x = np.cos(np.pi * np.arange(1, 2 * order, 2) / (2.0 * order))
        x_affine = array_affine(
            x,
            orig=cls.orthogonality_domain,  # pyright: ignore
            dest=approximation_domain,
        )
        w = np.ones(order) * (np.pi / order)
        T = np.cos(np.outer(np.arange(order), np.arccos(x), out=None))
        coefficients = (2 / np.pi) * (T * func(x_affine)) @ w
        coefficients[0] /= 2
        return cls(coefficients, approximation_domain=approximation_domain)

    @classmethod
    def interpolate(
        cls,
        func: ScalarFunction,
        approximation_domain: tuple[float, float] = (-1.0, 1.0),
        order: int | None = None,
        nodes: Literal["zeros", "extrema"] = "zeros",
    ) -> ChebyshevExpansion:
        """
        Project a scalar function onto the Chebyshev basis on the given approximation domain.

        The approximation domain must contain the full range of arguments on which the expansion
        will be evaluated; otherwise, rescaling maps the argument outside the orthogonality domain
        where the basis is not defined, leading to large errors.
        """
        if order is None:
            order = cls.estimate_order(func, approximation_domain)
        start, stop = approximation_domain
        if nodes == "zeros":
            x = ChebyshevInterval(start, stop, order).to_vector()
            coefficients = (1 / order) * dct(np.flip(func(x)), type=2)
        elif nodes == "extrema":
            x = ChebyshevInterval(start, stop, order, endpoints=True).to_vector()
            coefficients = 2 * dct(np.flip(func(x)), type=1, norm="forward")
        coefficients[0] /= 2
        return cls(coefficients, approximation_domain=approximation_domain)

    def deriv(self, m: int = 1) -> ChebyshevExpansion:
        """Return the m-th derivative as a new ChebyshevExpansion."""
        T = np.polynomial.Chebyshev(
            self.coefficients, domain=self.approximation_domain
        ).deriv(m)
        a, b = map(float, T.domain)  # Keep type checker happy
        return ChebyshevExpansion(T.coef, approximation_domain=(a, b))

    def integ(self, m: int = 1, lbnd: float = 0.0) -> ChebyshevExpansion:
        """Return the m-th integral as a new ChebyshevExpansion."""
        T = np.polynomial.Chebyshev(
            self.coefficients, domain=self.approximation_domain
        ).integ(m, lbnd=lbnd)
        a, b = map(float, T.domain)
        return ChebyshevExpansion(T.coef, approximation_domain=(a, b))
