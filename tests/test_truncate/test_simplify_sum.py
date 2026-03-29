import numpy as np
from unittest.mock import patch
from seemps.state import MPSSum, DEFAULT_STRATEGY, Simplification, Truncation
from seemps.state.simplification import combine, simplify_mps_sum
from seemps.state import simplification as simplification_module
from ..tools import SeeMPSTestCase
from seemps.tools import log


class TestSimplifyMPSSum(SeeMPSTestCase):
    def test_no_truncation(self):
        d = 2
        strategy = DEFAULT_STRATEGY.replace(
            method=Truncation.DO_NOT_TRUNCATE, simplify=Simplification.VARIATIONAL
        )
        log(
            f"rng={self.rng.bit_generator.state}, {self.rng.integers(0, 0xFFFFFFF, size=1)}"
        )
        for n in range(3, 4):  # range(3, 9):
            log(f"n={n}")
            ψ1 = self.random_uniform_mps(d, n, D=int(2 ** (n / 2)))
            ψ1 = ψ1 * (1 / ψ1.norm())
            ψ2 = self.random_uniform_mps(d, n, D=int(2 ** (n / 2)))
            ψ2 = ψ2 * (1 / ψ2.norm())
            a1 = self.rng.normal()
            a2 = self.rng.normal()
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
            ψ1 = self.random_uniform_mps(d, n, D=int(2 ** (n / 2)))
            ψ1 = ψ1 * (1 / ψ1.norm())
            ψ2 = self.random_uniform_mps(d, n, D=int(2 ** (n / 2)))
            ψ2 = ψ2 * (1 / ψ2.norm())
            a1 = self.rng.normal()
            a2 = self.rng.normal()
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
            ψ1 = self.random_uniform_mps(d, n, D=int(2 ** (n / 2)))
            ψ1 = ψ1 * (1 / ψ1.norm())
            ψ2 = self.random_uniform_mps(d, n, D=int(2 ** (n / 2)))
            ψ2 = ψ2 * (1 / ψ2.norm())
            a1 = self.rng.normal()
            a2 = self.rng.normal()
            φ = simplify_mps_sum(MPSSum([a1, a2], [ψ1, ψ2]), strategy=strategy)
            max_D_φ = max([max(t.shape) for t in φ])
            self.assertTrue(max_D_φ <= D)

    def test_simplification_method(self):
        d = 2
        strategy_0 = DEFAULT_STRATEGY.replace(simplify=Simplification.CANONICAL_FORM)
        strategy_1 = DEFAULT_STRATEGY.replace(simplify=Simplification.VARIATIONAL)
        for n in range(3, 9):
            ψ1 = self.random_uniform_mps(d, n, D=int(2 ** (n / 2)))
            ψ1 = ψ1 * (1 / ψ1.norm())
            ψ2 = self.random_uniform_mps(d, n, D=int(2 ** (n / 2)))
            ψ2 = ψ2 * (1 / ψ2.norm())
            a1 = self.rng.normal()
            a2 = self.rng.normal()
            φ0 = simplify_mps_sum(MPSSum([a1, a2], [ψ1, ψ2]), strategy=strategy_0)
            φ1 = simplify_mps_sum(MPSSum([a1, a2], [ψ1, ψ2]), strategy=strategy_1)
            ψ = a1 * ψ1.to_vector() + a2 * ψ2.to_vector()
            self.assertSimilar(ψ, φ0)
            self.assertSimilar(ψ, φ1)
            self.assertSimilar(φ0, φ1)

    def test_zero_weight_sum_returns_zero_state(self):
        ψ1 = self.random_uniform_mps(2, 4, D=3)
        ψ2 = self.random_uniform_mps(2, 4, D=3)
        φ = simplify_mps_sum(
            MPSSum([0.0, 0.0], [ψ1, ψ2]),
            strategy=DEFAULT_STRATEGY.replace(simplify=Simplification.VARIATIONAL),
        )
        self.assertAlmostEqual(φ.norm_squared(), 0.0)
        self.assertSimilar(φ.to_vector(), ψ1.zero_state().to_vector())

    def test_negative_direction_matches_positive_direction(self):
        strategy = DEFAULT_STRATEGY.replace(
            method=Truncation.DO_NOT_TRUNCATE, simplify=Simplification.VARIATIONAL
        )
        ψ1 = self.random_uniform_mps(2, 4, D=3)
        ψ2 = self.random_uniform_mps(2, 4, D=3)
        ψ1 = ψ1 * (1 / ψ1.norm())
        ψ2 = ψ2 * (1 / ψ2.norm())
        state = MPSSum([1.0, -0.25j], [ψ1, ψ2])
        φ_plus = simplify_mps_sum(state.copy(), strategy=strategy, direction=+1)
        φ_minus = simplify_mps_sum(state.copy(), strategy=strategy, direction=-1)
        self.assertSimilar(φ_plus, φ_minus)

    def test_do_not_simplify_returns_equivalent_sum(self):
        strategy = DEFAULT_STRATEGY.replace(
            method=Truncation.DO_NOT_TRUNCATE, simplify=Simplification.DO_NOT_SIMPLIFY
        )
        ψ1 = self.random_uniform_mps(2, 3, D=2)
        ψ2 = self.random_uniform_mps(2, 3, D=2)
        state = MPSSum([1.0, -0.5], [ψ1, ψ2])
        φ = simplify_mps_sum(state, strategy=strategy)
        self.assertSimilar(state.to_vector(), φ.to_vector())

    def test_combine_forwards_arguments_to_simplify_mps_sum(self):
        ψ1 = self.random_uniform_mps(2, 3, D=2)
        ψ2 = self.random_uniform_mps(2, 3, D=2)
        strategy = DEFAULT_STRATEGY.replace(
            method=Truncation.DO_NOT_TRUNCATE, simplify=Simplification.VARIATIONAL
        )
        expected = self.random_uniform_canonical_mps(2, 3, D=2)
        with patch.object(
            simplification_module, "simplify_mps_sum", return_value=expected
        ) as mocked:
            actual = combine([1.0, -1.0], [ψ1, ψ2], strategy=strategy, direction=-1, guess=ψ1)
        self.assertIs(actual, expected)
        args = mocked.call_args.args
        self.assertEqual(args[0].weights, [1.0, -1.0])
        self.assertEqual(args[0].states, [ψ1, ψ2])
        self.assertIs(args[1], strategy)
        self.assertEqual(args[2], -1)
        self.assertIs(args[3], ψ1)
