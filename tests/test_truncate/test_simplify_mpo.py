from seemps.operators import MPO
from seemps.state import (
    DEFAULT_STRATEGY,
    Simplification,
    Truncation,
    random_uniform_mps,
    scprod,
)
from seemps.operators import mpo_as_mps, simplify_mpo

from .. import tools


class TestSimplify(tools.TestCase):
    def test_no_truncation(self):
        d = 2
        strategy = DEFAULT_STRATEGY.replace(
            method=Truncation.DO_NOT_TRUNCATE, simplify=Simplification.VARIATIONAL
        )
        for n in range(3, 9):
            ψ = random_uniform_mps(2 * d, n, D=int(2 ** (n / 2)))
            ψ = ψ * (1 / ψ.norm())
            mpo = MPO([t.reshape(t.shape[0], d, d, t.shape[-1]) for t in ψ._data])
            φ = simplify_mpo(mpo, strategy=strategy)
            self.assertSimilar(mpo.to_matrix(), φ.to_matrix())

    def test_tolerance(self):
        d = 2
        tolerance = 1e-10
        strategy = DEFAULT_STRATEGY.replace(
            simplify=Simplification.VARIATIONAL, simplification_tolerance=tolerance
        )
        for n in range(3, 15):
            ψ = random_uniform_mps(2 * d, n, D=int(2 ** (n / 2)))
            ψ = ψ * (1 / ψ.norm())
            mpo = MPO([t.reshape(t.shape[0], d, d, t.shape[-1]) for t in ψ._data])
            φ = simplify_mpo(mpo, strategy=strategy)
            φ = mpo_as_mps(φ)
            err = 2 * abs(1.0 - scprod(ψ, φ).real / (ψ.norm() * φ.norm()))
            self.assertTrue(err < tolerance)

    def test_max_bond_dimensions(self):
        d = 2
        n = 14
        for D in range(2, 15):
            strategy = DEFAULT_STRATEGY.replace(
                simplify=Simplification.VARIATIONAL, max_bond_dimension=D
            )
            ψ = random_uniform_mps(2 * d, n, D=int(2 ** (n / 2)))
            ψ = ψ * (1 / ψ.norm())
            mpo = MPO([t.reshape(t.shape[0], d, d, t.shape[-1]) for t in ψ._data])
            φ = simplify_mpo(mpo, strategy=strategy)
            max_D_φ = max([max(t.shape) for t in φ])
            self.assertTrue(max_D_φ <= D)

    def test_simplification_method(self):
        d = 2
        strategy_0 = DEFAULT_STRATEGY.replace(simplify=Simplification.CANONICAL_FORM)
        strategy_1 = DEFAULT_STRATEGY.replace(simplify=Simplification.VARIATIONAL)
        for n in range(3, 9):
            ψ = random_uniform_mps(2 * d, n, D=int(2 ** (n / 2)))
            ψ = ψ * (1 / ψ.norm())
            mpo = MPO([t.reshape(t.shape[0], d, d, t.shape[-1]) for t in ψ._data])
            φ0 = simplify_mpo(mpo, strategy=strategy_0)
            φ1 = simplify_mpo(mpo, strategy=strategy_1)
            self.assertSimilar(mpo.to_matrix(), φ0.to_matrix())
            self.assertSimilar(mpo.to_matrix(), φ1.to_matrix())
            self.assertSimilar(φ0.to_matrix(), φ1.to_matrix())
