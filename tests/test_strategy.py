import numpy as np
from .tools import *
from seemps.state.core import (
    Strategy,
    Truncation,
    DEFAULT_STRATEGY,
    NO_TRUNCATION,
    truncate_vector,
)


class TestStrategy(TestCase):
    def logarithmic_values(self):
        return np.array([1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7])


class TestStrategyNoTruncation(TestStrategy):
    def test_strategy_no_truncation(self):
        values = self.logarithmic_values()
        old_values = values.copy()
        s, err = truncate_vector(values, NO_TRUNCATION)
        self.assertTrue(s is values)
        self.assertEqual(s.size, old_values.size)
        self.assertEqual(err, 0.0)

    def test_strategy_no_truncation_ignores_tolerance(self):
        values = self.logarithmic_values()
        old_values = values.copy()
        s, err = truncate_vector(values, NO_TRUNCATION.replace(tolerance=1e-3))
        self.assertTrue(s is values)
        self.assertEqual(s.size, old_values.size)
        self.assertEqual(err, 0.0)

    def test_strategy_no_truncation_ignores_max_bond_dimension(self):
        values = self.logarithmic_values()
        old_values = values.copy()
        s, err = truncate_vector(values, NO_TRUNCATION.replace(tolerance=1e-3))
        self.assertTrue(s is values)
        self.assertEqual(s.size, old_values.size)
        self.assertEqual(err, 0.0)


class TestStrategyAbsoluteSingularValue(TestStrategy):
    def test_strategy_absolute_singular_value(self):
        values = np.array([1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7])
        strategy = Strategy(
            method=Truncation.ABSOLUTE_SINGULAR_VALUE,
            tolerance=0.5e-4,
        )
        values = self.logarithmic_values()
        s, err = truncate_vector(values, strategy)
        self.assertEqual(s.size, 4)
        self.assertTrue(s is not values)
        self.assertEqual(err, np.sum(values[4:] ** 2))

    def test_strategy_absolute_singular_value_respects_max_bond_dimension(self):
        values = np.array([1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7])
        strategy = Strategy(
            method=Truncation.ABSOLUTE_SINGULAR_VALUE,
            tolerance=0.5e-4,
            max_bond_dimension=2,
        )
        values = self.logarithmic_values()
        s, err = truncate_vector(values, strategy)
        self.assertEqual(s.size, 2)
        self.assertTrue(s is not values)
        self.assertEqual(err, np.sum(values[2:] ** 2))

    def test_strategy_absolute_singular_value_zero_tolerance_behavior(self):
        values = np.array([1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7])
        strategy = Strategy(method=Truncation.ABSOLUTE_SINGULAR_VALUE, tolerance=0.0)
        values = self.logarithmic_values()
        s, err = truncate_vector(values, strategy)
        self.assertEqual(s.size, values.size)
        self.assertTrue(s is not values)
        self.assertEqual(err, 0.0)

    def test_strategy_zero_tolerance_changes_method(self):
        strategy = Strategy(method=Truncation.RELATIVE_SINGULAR_VALUE, tolerance=0.0)
        self.assertEqual(strategy.get_method(), Truncation.ABSOLUTE_SINGULAR_VALUE)
        strategy = Strategy(
            method=Truncation.RELATIVE_NORM_SQUARED_ERROR, tolerance=0.0
        )
        self.assertEqual(strategy.get_method(), Truncation.ABSOLUTE_SINGULAR_VALUE)
        strategy = Strategy(method=Truncation.DO_NOT_TRUNCATE, tolerance=0.0)
        self.assertEqual(strategy.get_method(), Truncation.DO_NOT_TRUNCATE)


class TestStrategyRelativeSingularValue(TestStrategy):
    def test_strategy_relative_singular_value(self):
        values = np.array([1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7])
        strategy = Strategy(
            method=Truncation.RELATIVE_SINGULAR_VALUE,
            tolerance=0.5e-4,
        )
        values = self.logarithmic_values()
        s, err = truncate_vector(values, strategy)
        self.assertEqual(s.size, 5)
        self.assertTrue(s is not values)
        self.assertEqual(err, np.sum(values[5:] ** 2))

    def test_strategy_relative_singular_value_respects_max_bond_dimension(self):
        values = np.array([1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7])
        strategy = Strategy(
            method=Truncation.RELATIVE_SINGULAR_VALUE,
            tolerance=0.5e-4,
            max_bond_dimension=2,
        )
        values = self.logarithmic_values()
        s, err = truncate_vector(values, strategy)
        self.assertEqual(s.size, 2)
        self.assertTrue(s is not values)
        self.assertEqual(err, np.sum(values[2:] ** 2))
