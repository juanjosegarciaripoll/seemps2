from __future__ import annotations
import numpy as np

from ...state import MPS
from ...operators import MPO
from ...typing import Vector
from ..mesh import array_affine
from ..factories import mps_affine
from ..operators import mpo_affine
from .expansion import PolynomialExpansion, ScalarFunction


class LegendreExpansion(PolynomialExpansion):
    r"""
    Expansion in the Legendre basis.

    The Legendre polynomials :math:`P_k(x)` are orthogonal on the interval
    :math:`[−1, 1]` with respect to the uniform weight :math:`w(x)=1`.
    They are widely used in approximation theory since truncated Legendre series
    minimize the error in the :math:`L^2([-1,1])` norm.

    See https://en.wikipedia.org/wiki/Legendre_polynomials for more information.
    """

    orthogonality_domain = (-1.0, 1.0)
    affine_fix = (1.0, 0.0)

    def __init__(self, coefficients: Vector, approximation_domain: tuple[float, float]):
        self.approximation_domain = approximation_domain
        super().__init__(coefficients)

    def recurrence_coefficients(self, k: int) -> tuple[float, float, float]:
        """
        Returns the three-term coefficients of the Legendre recursion:

        .. math::
           (k+1) P_{k+1}(x) = (2k+1) x P_k(x) - k P_{k-1}(x)
        """
        return ((2 * k + 1) / (k + 1), 0.0, k / (k + 1))

    def rescale_mps(self, mps: MPS) -> MPS:
        orig = self.approximation_domain
        dest: tuple[float, float] = self.orthogonality_domain  # type: ignore
        return mps_affine(mps, orig, dest)

    def rescale_mpo(self, mpo: MPO) -> MPO:
        orig = self.approximation_domain
        dest: tuple[float, float] = self.orthogonality_domain  # type: ignore
        return mpo_affine(mpo, orig, dest)

    @classmethod
    def project(
        cls, func: ScalarFunction, approximation_domain: tuple[float, float], order: int
    ) -> LegendreExpansion:
        """
        Project a scalar function onto the Legendre basis on the given approximation domain.

        The approximation domain must contain the full range of arguments on which the expansion
        will be evaluated; otherwise, rescaling maps the argument outside the orthogonality domain
        where the basis is not defined, leading to large errors.
        """
        x, w = np.polynomial.legendre.leggauss(order)
        x_affine = array_affine(
            x,
            orig=cls.orthogonality_domain,  # type: ignore
            dest=approximation_domain,
        )
        P = np.vstack(
            [np.polynomial.legendre.legval(x, [0] * k + [1]) for k in range(order)]
        )
        coefficients = 0.5 * (2 * np.arange(order) + 1) * (P * func(x_affine)).dot(w)
        return cls(coefficients, approximation_domain=approximation_domain)
