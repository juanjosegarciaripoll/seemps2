from .common import CoreComparisonTestCase
import seemps.cython.core
import seemps.cython.pybind


class TestCoreStrategy(CoreComparisonTestCase):
    def test_strategies_are_same(self):
        """Test that the Cython and Pybind strategies are the same."""

        for st1, st2 in self.stategies:
            with self.subTest(strategy=(st1, st2)):
                self.assertEqual(st1.get_tolerance(), st2.get_tolerance())
                self.assertEqual(
                    st1.get_simplification_tolerance(),
                    st2.get_simplification_tolerance(),
                )
                self.assertEqual(
                    st1.get_simplification_method(), st2.get_simplification_method()
                )
                self.assertEqual(
                    st1.get_max_bond_dimension(), st2.get_max_bond_dimension()
                )
                self.assertEqual(st1.get_max_sweeps(), st2.get_max_sweeps())
                self.assertEqual(st1.get_method(), st2.get_method())
                self.assertEqual(st1.get_normalize_flag(), st2.get_normalize_flag())
                self.assertEqual(st1.get_simplify_flag(), st2.get_simplify_flag())

    def test_same_constants(self):
        """Test that the Cython and Pybind constants are the same."""

        self.assertEqual(
            seemps.cython.core.DEFAULT_TOLERANCE,
            seemps.cython.pybind.DEFAULT_TOLERANCE,
        )
        self.assertEqual(
            seemps.cython.core.Simplification.DO_NOT_SIMPLIFY,
            seemps.cython.pybind.Simplification.DO_NOT_SIMPLIFY,
        )
        self.assertEqual(
            seemps.cython.core.Simplification.CANONICAL_FORM,
            seemps.cython.pybind.Simplification.CANONICAL_FORM,
        )
        self.assertEqual(
            seemps.cython.core.Simplification.VARIATIONAL,
            seemps.cython.pybind.Simplification.VARIATIONAL,
        )
        self.assertEqual(
            seemps.cython.core.Simplification.VARIATIONAL_EXACT_GUESS,
            seemps.cython.pybind.Simplification.VARIATIONAL_EXACT_GUESS,
        )
