from unittest.mock import patch
from seemps.state import (
    DEFAULT_STRATEGY,
    CanonicalMPS,
    MPSSum,
    Simplification,
    Truncation,
    scprod,
    simplify,
)
from seemps.state import simplification as simplification_module
from .. import tools


class TestSimplify(tools.SeeMPSTestCase):
    def test_no_truncation(self):
        d = 2
        strategy = DEFAULT_STRATEGY.replace(
            method=Truncation.DO_NOT_TRUNCATE, simplify=Simplification.VARIATIONAL
        )
        for n in range(3, 9):
            ψ = self.random_uniform_mps(d, n, D=int(2 ** (n / 2)))
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
            ψ = self.random_uniform_mps(d, n, D=int(2 ** (n / 2)))
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
            ψ = self.random_uniform_mps(d, n, D=int(2 ** (n / 2)))
            ψ = ψ * (1 / ψ.norm())
            φ = simplify(ψ, strategy=strategy)
            max_D_φ = max([max(t.shape) for t in φ])
            self.assertTrue(max_D_φ <= D)

    def test_simplification_method(self):
        d = 2
        strategy_0 = DEFAULT_STRATEGY.replace(simplify=Simplification.CANONICAL_FORM)
        strategy_1 = DEFAULT_STRATEGY.replace(simplify=Simplification.VARIATIONAL)
        for n in range(3, 9):
            ψ = self.random_uniform_mps(d, n, D=int(2 ** (n / 2)))
            ψ = ψ * (1 / ψ.norm())
            φ0 = simplify(ψ, strategy=strategy_0)
            φ1 = simplify(ψ, strategy=strategy_1)
            self.assertSimilar(ψ, φ0)
            self.assertSimilar(ψ, φ1)
            self.assertSimilar(φ0, φ1)

    def test_zero_state_is_preserved(self):
        strategy = DEFAULT_STRATEGY.replace(simplify=Simplification.VARIATIONAL)
        ψ = self.random_uniform_mps(2, 4, D=3).zero_state()
        φ = simplify(ψ, strategy=strategy)
        self.assertIsInstance(φ, CanonicalMPS)
        self.assertSimilar(φ.to_vector(), ψ.to_vector())

    def test_negative_direction_matches_positive_direction(self):
        strategy = DEFAULT_STRATEGY.replace(
            method=Truncation.DO_NOT_TRUNCATE, simplify=Simplification.VARIATIONAL
        )
        ψ = self.random_uniform_mps(2, 5, D=4)
        ψ = ψ * (1 / ψ.norm())
        φ_plus = simplify(ψ, strategy=strategy, direction=+1)
        φ_minus = simplify(ψ, strategy=strategy, direction=-1)
        self.assertSimilar(φ_plus, φ_minus)

    def test_do_not_simplify_returns_equivalent_state(self):
        strategy = DEFAULT_STRATEGY.replace(
            method=Truncation.DO_NOT_TRUNCATE, simplify=Simplification.DO_NOT_SIMPLIFY
        )
        ψ = self.random_uniform_mps(2, 4, D=4)
        ψ = ψ * (1 / ψ.norm())
        φ = simplify(ψ, strategy=strategy)
        self.assertSimilar(ψ, φ)

    def test_simplify_dispatches_mpssum_to_sum_helper(self):
        state = MPSSum(
            [1.0, -0.5],
            [self.random_uniform_mps(2, 3, D=2), self.random_uniform_mps(2, 3, D=2)],
        )
        strategy = DEFAULT_STRATEGY.replace(simplify=Simplification.VARIATIONAL)
        expected = self.random_uniform_canonical_mps(2, 3, D=2)
        with patch.object(
            simplification_module, "simplify_mps_sum", return_value=expected
        ) as mocked:
            actual = simplification_module.simplify(
                state, strategy=strategy, direction=-1, guess=state.states[0]
            )
        self.assertIs(actual, expected)
        mocked.assert_called_once_with(state, strategy, -1, state.states[0])
