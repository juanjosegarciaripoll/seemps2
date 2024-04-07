import numpy as np
from seemps.state.core import Truncation, Strategy, destructively_truncate_vector
from .. import tools


class TestStrategy(tools.TestCase):
    def test_strategy_no_truncation(self):
        s = np.array([1.0, 0.2, 0.01, 0.005, 0.0005])
        strategy = Strategy(method=Truncation.DO_NOT_TRUNCATE)
        err = destructively_truncate_vector(s, strategy)
        self.assertEqual(err, 0.0)

    def test_strategy_relative_singular_value(self):
        s = np.array([1.0, 0.1, 0.01, 0.001, 0.0001])

        strategy = Strategy(
            method=Truncation.RELATIVE_SINGULAR_VALUE,
            tolerance=0.5,
            normalize=False,
        )
        err = destructively_truncate_vector(news := s.copy(), strategy)
        self.assertSimilar(news, np.array([1.0]))
        self.assertAlmostEqual(err, np.sum(s[1:] ** 2))

        strategy = Strategy(
            method=Truncation.RELATIVE_SINGULAR_VALUE,
            tolerance=0.05,
            normalize=False,
        )
        err = destructively_truncate_vector(news := s.copy(), strategy)
        self.assertSimilar(news, np.array([1.0, 0.1]))
        self.assertAlmostEqual(err, np.sum(s[2:] ** 2))

        strategy = Strategy(
            method=Truncation.RELATIVE_SINGULAR_VALUE,
            tolerance=0.005,
            normalize=False,
        )
        err = destructively_truncate_vector(news := s.copy(), strategy)
        self.assertSimilar(news, np.array([1.0, 0.1, 0.01]))
        self.assertAlmostEqual(err, np.sum(s[3:] ** 2))

        strategy = Strategy(
            method=Truncation.RELATIVE_SINGULAR_VALUE,
            tolerance=0.0005,
            normalize=False,
        )
        err = destructively_truncate_vector(news := s.copy(), strategy)
        self.assertSimilar(news, np.array([1.0, 0.1, 0.01, 0.001]))
        self.assertAlmostEqual(err, np.sum(s[4:] ** 2))

        strategy = Strategy(
            method=Truncation.RELATIVE_SINGULAR_VALUE,
            tolerance=0.00005,
            normalize=False,
        )
        err = destructively_truncate_vector(news := s.copy(), strategy)
        self.assertSimilar(news, s)
        self.assertAlmostEqual(err, 0.0)

    def test_strategy_relative_norm(self):
        s = np.array([1.0, 0.1, 0.01, 0.001, 0.0001])
        norm_errors = [
            1.01010100e-02,
            1.01010000e-04,
            1.01000000e-06,
            9.99999994e-09,
            0.00000000e00,
        ]

        strategy = Strategy(
            method=Truncation.RELATIVE_NORM_SQUARED_ERROR,
            tolerance=0.1,
            normalize=False,
        )
        err = destructively_truncate_vector(news := s.copy(), strategy)
        self.assertSimilar(news, np.array([1.0]))
        self.assertAlmostEqual(err, norm_errors[0])

        strategy = Strategy(
            method=Truncation.RELATIVE_NORM_SQUARED_ERROR,
            tolerance=1e-3,
            normalize=False,
        )
        err = destructively_truncate_vector(news := s.copy(), strategy)
        self.assertSimilar(news, np.array([1.0, 0.1]))
        self.assertAlmostEqual(err, norm_errors[1])

        strategy = Strategy(
            method=Truncation.RELATIVE_NORM_SQUARED_ERROR,
            tolerance=1e-5,
            normalize=False,
        )
        err = destructively_truncate_vector(news := s.copy(), strategy)
        self.assertSimilar(news, np.array([1.0, 0.1, 0.01]))
        self.assertAlmostEqual(err, norm_errors[2])

        strategy = Strategy(
            method=Truncation.RELATIVE_NORM_SQUARED_ERROR,
            tolerance=1e-7,
            normalize=False,
        )
        err = destructively_truncate_vector(news := s.copy(), strategy)
        self.assertSimilar(news, np.array([1.0, 0.1, 0.01, 0.001]))
        self.assertAlmostEqual(err, norm_errors[3])

        strategy = Strategy(
            method=Truncation.RELATIVE_NORM_SQUARED_ERROR,
            tolerance=1e-9,
            normalize=False,
        )
        err = destructively_truncate_vector(news := s.copy(), strategy)
        self.assertSimilar(news, s)
        self.assertAlmostEqual(err, norm_errors[4])
