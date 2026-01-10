import numpy as np
from seemps.analysis.derivatives import (
    fourier_derivative,
    fourier_derivative_mpo,
)
from seemps.analysis.factories import mps_exponential, mps_sin, mps_cos
from seemps.analysis.mesh import QuantizedInterval
from .. import tools


class TestFourierDerivative(tools.TestCase):
    def setUp(self) -> None:
        self.interval = interval = QuantizedInterval(0, 2 * np.pi, qubits=2)
        self.exp1j = mps_exponential(interval, 1j)
        self.sin = mps_sin(interval)
        self.cos = mps_cos(interval)

    def test_first_derivative_periodic_functions(self):
        L = self.interval.length()
        self.assertSimilar(fourier_derivative(self.exp1j, 1, L), 1j * self.exp1j)
        self.assertSimilar(fourier_derivative(self.sin, 1, L), self.cos)
        self.assertSimilar(fourier_derivative(self.cos, 1, L), -1.0 * self.sin)

    def test_second_derivative_periodic_functions(self):
        L = self.interval.length()
        self.assertSimilar(fourier_derivative(self.exp1j, 2, L), -1.0 * self.exp1j)
        self.assertSimilar(fourier_derivative(self.sin, 2, L), -1.0 * self.sin)
        self.assertSimilar(fourier_derivative(self.cos, 2, L), -1.0 * self.cos)

    def test_first_derivative_mpo(self):
        A = fourier_derivative_mpo(2, 1, self.interval.step)
        self.assertSimilar(
            A.to_matrix(),
            [
                [-0.5j, 0.5 + 0.5j, -0.5j, -0.5 + 0.5j],
                [-0.5 + 0.5j, -0.5j, 0.5 + 0.5j, -0.5j],
                [-0.5j, -0.5 + 0.5j, -0.5j, 0.5 + 0.5j],
                [0.5 + 0.5j, -0.5j, -0.5 + 0.5j, -0.5j],
            ],
        )

    def test_second_derivative_mpo(self):
        A = fourier_derivative_mpo(2, 2, self.interval.step)
        self.assertSimilar(
            A.to_matrix(),
            [
                [-1.5, 1, -0.5, 1],
                [1, -1.5, 1, -0.5],
                [-0.5, 1, -1.5, 1],
                [1, -0.5, 1, -1.5],
            ],
        )
