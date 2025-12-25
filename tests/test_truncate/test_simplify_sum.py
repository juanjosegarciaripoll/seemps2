import numpy as np
from seemps.state import (
    MPSSum,
    DEFAULT_STRATEGY,
    Simplification,
    Truncation,
    random_uniform_mps,
)
from seemps.state.simplification import simplify_mps_sum
from .. import tools


class TestSimplifyMPSSum(tools.TestCase):
    def test_no_truncation(self):
        d = 2
        strategy = DEFAULT_STRATEGY.replace(
            method=Truncation.DO_NOT_TRUNCATE, simplify=Simplification.VARIATIONAL
        )
        for n in range(3, 9):
            ψ1 = random_uniform_mps(d, n, D=int(2 ** (n / 2)))
            ψ1 = ψ1 * (1 / ψ1.norm())
            ψ2 = random_uniform_mps(d, n, D=int(2 ** (n / 2)))
            ψ2 = ψ2 * (1 / ψ2.norm())
            a1 = np.random.randn()
            a2 = np.random.randn()
            ψ = a1 * ψ1.to_vector() + a2 * ψ2.to_vector()
            φ = simplify_mps_sum(
                MPSSum([a1, a2], [ψ1, ψ2]),
                strategy,
            )
            self.assertSimilar(ψ, φ.to_vector())

    def test_tolerance(self):
        d = 2
        tolerance = 1e-10
        strategy = DEFAULT_STRATEGY.replace(
            simplify=Simplification.VARIATIONAL, simplification_tolerance=tolerance
        )
        for n in range(3, 15):
            ψ1 = random_uniform_mps(d, n, D=int(2 ** (n / 2)))
            ψ1 = ψ1 * (1 / ψ1.norm())
            ψ2 = random_uniform_mps(d, n, D=int(2 ** (n / 2)))
            ψ2 = ψ2 * (1 / ψ2.norm())
            a1 = np.random.randn()
            a2 = np.random.randn()
            ψ = a1 * ψ1.to_vector() + a2 * ψ2.to_vector()
            φ = simplify_mps_sum(MPSSum([a1, a2], [ψ1, ψ2]), strategy=strategy)
            err = 2 * abs(
                1.0 - np.vdot(ψ, φ.to_vector()).real / (np.linalg.norm(ψ) * φ.norm())
            )
            self.assertTrue(err < tolerance)

    def test_max_bond_dimensions(self):
        d = 2
        n = 14
        for D in range(2, 15):
            strategy = DEFAULT_STRATEGY.replace(
                simplify=Simplification.VARIATIONAL, max_bond_dimension=D
            )
            ψ1 = random_uniform_mps(d, n, D=int(2 ** (n / 2)))
            ψ1 = ψ1 * (1 / ψ1.norm())
            ψ2 = random_uniform_mps(d, n, D=int(2 ** (n / 2)))
            ψ2 = ψ2 * (1 / ψ2.norm())
            a1 = np.random.randn()
            a2 = np.random.randn()
            φ = simplify_mps_sum(MPSSum([a1, a2], [ψ1, ψ2]), strategy=strategy)
            max_D_φ = max([max(t.shape) for t in φ])
            self.assertTrue(max_D_φ <= D)

    def test_simplification_method(self):
        d = 2
        strategy_0 = DEFAULT_STRATEGY.replace(simplify=Simplification.CANONICAL_FORM)
        strategy_1 = DEFAULT_STRATEGY.replace(simplify=Simplification.VARIATIONAL)
        for n in range(3, 9):
            ψ1 = random_uniform_mps(d, n, D=int(2 ** (n / 2)))
            ψ1 = ψ1 * (1 / ψ1.norm())
            ψ2 = random_uniform_mps(d, n, D=int(2 ** (n / 2)))
            ψ2 = ψ2 * (1 / ψ2.norm())
            a1 = np.random.randn()
            a2 = np.random.randn()
            φ0 = simplify_mps_sum(MPSSum([a1, a2], [ψ1, ψ2]), strategy=strategy_0)
            φ1 = simplify_mps_sum(MPSSum([a1, a2], [ψ1, ψ2]), strategy=strategy_1)
            ψ = a1 * ψ1.to_vector() + a2 * ψ2.to_vector()
            self.assertSimilar(ψ, φ0)
            self.assertSimilar(ψ, φ1)
            self.assertSimilar(φ0, φ1)
