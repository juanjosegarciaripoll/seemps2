import numpy as np
from seemps.analysis.integration.mps_quadratures import (
    mps_clenshaw_curtis,
    mps_fejer,
    mps_fifth_order,
    mps_simpson38,
    mps_trapezoidal,
)
from seemps.analysis.integration.vector_quadratures import (
    vector_best_newton_cotes,
    vector_boole,
    vector_clenshaw_curtis,
    vector_fejer,
    vector_fifth_order,
    vector_simpson13,
    vector_simpson38,
    vector_trapezoidal,
)
from ..tools import SeeMPSTestCase


class TestVectorQuadratures(SeeMPSTestCase):
    def test_trapezoidal_matches_known_weights(self):
        self.assertSimilar(vector_trapezoidal(-1.0, 1.0, 3), [0.5, 1.0, 0.5])

    def test_simpson13_matches_known_weights(self):
        self.assertSimilar(
            vector_simpson13(-1.0, 1.0, 5), np.array([1, 4, 2, 4, 1]) / 6.0
        )

    def test_simpson38_matches_mps_quadrature(self):
        self.assertSimilar(
            vector_simpson38(-2.0, 2.0, 16), mps_simpson38(-2.0, 2.0, 4).to_vector()
        )

    def test_boole_matches_known_weights(self):
        self.assertSimilar(
            vector_boole(-1.0, 1.0, 5), np.array([7, 32, 12, 32, 7]) / 45.0
        )

    def test_fifth_order_matches_mps_quadrature(self):
        self.assertSimilar(
            vector_fifth_order(-1.0, 1.0, 16), mps_fifth_order(-1.0, 1.0, 4).to_vector()
        )

    def test_best_newton_cotes_selects_highest_valid_order(self):
        self.assertSimilar(
            vector_best_newton_cotes(-1.0, 1.0, 6), vector_fifth_order(-1.0, 1.0, 6)
        )
        self.assertSimilar(
            vector_best_newton_cotes(-1.0, 1.0, 5), vector_boole(-1.0, 1.0, 5)
        )
        self.assertSimilar(
            vector_best_newton_cotes(-1.0, 1.0, 4), vector_simpson38(-1.0, 1.0, 4)
        )
        self.assertSimilar(
            vector_best_newton_cotes(-1.0, 1.0, 3), vector_simpson13(-1.0, 1.0, 3)
        )
        self.assertSimilar(
            vector_best_newton_cotes(-1.0, 1.0, 2), vector_trapezoidal(-1.0, 1.0, 2)
        )

    def test_newton_cotes_methods_raise_for_incompatible_sizes(self):
        with self.assertRaises(ValueError):
            vector_simpson13(-1.0, 1.0, 4)
        with self.assertRaises(ValueError):
            vector_simpson38(-1.0, 1.0, 5)
        with self.assertRaises(ValueError):
            vector_boole(-1.0, 1.0, 6)
        with self.assertRaises(ValueError):
            vector_fifth_order(-1.0, 1.0, 7)

    def test_fejer_matches_mps_quadrature(self):
        self.assertSimilar(vector_fejer(-2.0, 2.0, 8), mps_fejer(-2.0, 2.0, 3).to_vector())

    def test_clenshaw_curtis_matches_mps_quadrature(self):
        self.assertSimilar(
            vector_clenshaw_curtis(-2.0, 2.0, 7),
            mps_clenshaw_curtis(-2.0, 2.0, 3).to_vector(),
        )
