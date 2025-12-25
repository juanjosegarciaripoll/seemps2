from seemps.state import (
    DEFAULT_STRATEGY,
    Simplification,
    Truncation,
    random_uniform_mps,
    scprod,
    simplify,
)
from .. import tools


class TestSimplify(tools.TestCase):
    def test_no_truncation(self):
        d = 2
        strategy = DEFAULT_STRATEGY.replace(
            method=Truncation.DO_NOT_TRUNCATE, simplify=Simplification.VARIATIONAL
        )
        for n in range(3, 9):
            ψ = random_uniform_mps(d, n, D=int(2 ** (n / 2)))
            ψ = ψ * (1 / ψ.norm())
            φ = simplify(ψ, strategy=strategy)
            self.assertSimilar(ψ.to_vector(), φ.to_vector())

    def test_tolerance(self):
        d = 2
        tolerance = 1e-10
        strategy = DEFAULT_STRATEGY.replace(
            simplify=Simplification.VARIATIONAL, simplification_tolerance=tolerance
        )
        for n in range(3, 15):
            ψ = random_uniform_mps(d, n, D=int(2 ** (n / 2)))
            ψ = ψ * (1 / ψ.norm())
            φ = simplify(ψ, strategy=strategy)
            err = 2 * abs(1.0 - scprod(ψ, φ).real / (ψ.norm() * φ.norm()))
            self.assertTrue(err < tolerance)

    def test_max_bond_dimensions(self):
        d = 2
        n = 14
        for D in range(2, 15):
            strategy = DEFAULT_STRATEGY.replace(
                simplify=Simplification.VARIATIONAL, max_bond_dimension=D
            )
            ψ = random_uniform_mps(d, n, D=int(2 ** (n / 2)))
            ψ = ψ * (1 / ψ.norm())
            φ = simplify(ψ, strategy=strategy)
            max_D_φ = max([max(t.shape) for t in φ])
            self.assertTrue(max_D_φ <= D)

    def test_simplification_method(self):
        d = 2
        strategy_0 = DEFAULT_STRATEGY.replace(simplify=Simplification.CANONICAL_FORM)
        strategy_1 = DEFAULT_STRATEGY.replace(simplify=Simplification.VARIATIONAL)
        for n in range(3, 9):
            ψ = random_uniform_mps(d, n, D=int(2 ** (n / 2)))
            ψ = ψ * (1 / ψ.norm())
            φ0 = simplify(ψ, strategy=strategy_0)
            φ1 = simplify(ψ, strategy=strategy_1)
            self.assertSimilar(ψ, φ0)
            self.assertSimilar(ψ, φ1)
            self.assertSimilar(φ0, φ1)
