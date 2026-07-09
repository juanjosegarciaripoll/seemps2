import numpy as np

from seemps.analysis.integration.vector_quadratures import (
    _get_newton_cotes,
    vector_trapezoidal,
    vector_simpson13,
    vector_simpson38,
    vector_boole,
    vector_fifth_order,
    vector_best_newton_cotes,
    vector_fejer,
    vector_clenshaw_curtis,
)

from .. import tools


class TestNewtonCotesVectors(tools.SeeMPSTestCase):
    # (rule, valid number of nodes, polynomial degree integrated exactly)
    RULES = [
        (vector_trapezoidal, 9, 1),
        (vector_simpson13, 9, 3),
        (vector_simpson38, 10, 3),
        (vector_boole, 9, 5),
        (vector_fifth_order, 11, 5),
    ]

    def test_weights_integrate_constant(self):
        # Every Newton-Cotes rule reproduces the length of the interval,
        # i.e. the weights integrate f(x) = 1 exactly.
        a, b = -1.0, 2.0
        for rule, nodes, _ in self.RULES:
            q = rule(a, b, nodes)
            self.assertAlmostEqual(float(np.sum(q)), b - a, places=12)

    def test_weights_integrate_polynomials_exactly(self):
        a, b = -1.0, 2.0
        for rule, nodes, degree in self.RULES:
            q = rule(a, b, nodes)
            x = np.linspace(a, b, nodes)
            p = np.poly1d(np.arange(1, degree + 2))
            approx = q @ p(x)
            exact = np.polyint(p)(b) - np.polyint(p)(a)
            self.assertAlmostEqual(approx, exact, places=10)

    def test_get_newton_cotes_rejects_incompatible_size(self):
        # A three-point cell (Simpson 1/3) does not tile six nodes.
        with self.assertRaises(ValueError):
            _get_newton_cotes(6, np.array([1, 4, 1]))

    def test_best_newton_cotes_selects_highest_order(self):
        a, b = -1.0, 2.0
        # nodes = 6 fits the fifth-order rule exactly.
        self.assertSimilar(
            vector_best_newton_cotes(a, b, 6), vector_fifth_order(a, b, 6)
        )

    def test_best_newton_cotes_falls_back(self):
        a, b = -1.0, 2.0
        # nodes = 8 only fits the trapezoidal rule.
        self.assertSimilar(
            vector_best_newton_cotes(a, b, 8), vector_trapezoidal(a, b, 8)
        )
        # nodes = 3 falls back to Simpson 1/3.
        self.assertSimilar(
            vector_best_newton_cotes(a, b, 3), vector_simpson13(a, b, 3)
        )


class TestSpectralQuadratureVectors(tools.SeeMPSTestCase):
    def test_fejer_integrates_polynomials(self):
        a, b, nodes = -1.0, 2.0, 16
        q = vector_fejer(a, b, nodes)
        theta = (np.arange(nodes) + 0.5) * np.pi / nodes
        x = 0.5 * (b - a) * np.cos(theta) + 0.5 * (a + b)
        for degree in range(4):
            p = np.poly1d([1.0] + [0.0] * degree)
            approx = float(np.real(q @ p(x)))
            exact = np.polyint(p)(b) - np.polyint(p)(a)
            self.assertAlmostEqual(approx, exact, places=10)

    def test_clenshaw_curtis_shape_is_finite_and_real(self):
        a, b, nodes = -1.0, 2.0, 16
        q = vector_clenshaw_curtis(a, b, nodes)
        # The rule appends the first endpoint, so it returns nodes + 1 weights.
        self.assertEqual(len(q), nodes + 1)
        self.assertTrue(np.all(np.isfinite(q)))
        self.assertTrue(np.isrealobj(q))
