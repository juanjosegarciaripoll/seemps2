import numpy as np
from seemps.analysis.interpolation import *
from seemps.analysis.space import Space
from seemps.state import MPS, DEFAULT_STRATEGY

from ..tools import *
from .tools_interpolation import *


class TestInterpolation(TestCase):
    strategy = DEFAULT_STRATEGY.replace(simplify=2)

    def test_fourier_interpolation_1D(self):
        for n in range(3, 8):
            qubits_per_dimension = [n]
            dims = [2**n for n in qubits_per_dimension]
            L = 10
            space = Space(qubits_per_dimension, L=[[-L / 2, L / 2]])
            r_N = get_position_regular_grid(dims, space.a, space.dx)
            sol_N = gaussian(r_N)
            sol_N_mps = MPS.from_vector(
                sol_N, [2] * sum(qubits_per_dimension), normalize=False
            )
            m = n + 2
            M = 2**m
            sol_int = fourier_interpolation_vector_1D(sol_N, M)
            sol_int /= np.linalg.norm(sol_int)
            sol_int_mps, _ = fourier_interpolation_1D(
                sol_N_mps, space, n, m, dim=0, strategy=self.strategy
            )
            self.assertSimilar(sol_int, sol_int_mps)

    def test_fourier_interpolation(self):
        for n in range(3, 8):
            qubits_per_dimension = [n, n]
            dims = [2**n for n in qubits_per_dimension]
            L = 10
            space = Space(qubits_per_dimension, L=[[-L / 2, L / 2], [-L / 2, L / 2]])
            r_N = get_position_regular_grid(dims, space.a, space.dx)
            sol_N = gaussian(r_N)
            sol_N_mps = MPS.from_vector(
                sol_N, [2] * sum(qubits_per_dimension), normalize=False
            )
            m = n + 2
            qubits_per_dimension_M = [m, m]
            dims_M = [2**m for m in qubits_per_dimension_M]
            sol_int = fourier_interpolation_vector(sol_N, dims_M)
            sol_int /= np.linalg.norm(sol_int)
            sol_int_mps = fourier_interpolation(
                sol_N_mps,
                space,
                qubits_per_dimension,
                qubits_per_dimension_M,
                strategy=self.strategy,
            )
            self.assertSimilar(sol_int.flatten(), sol_int_mps)

    def test_finite_differences_interpolation_1D(self):
        for n in range(3, 8):
            qubits_per_dimension = [n]
            dims = [2**n for n in qubits_per_dimension]
            L = 10
            space = Space(qubits_per_dimension, L=[[-L / 2, L / 2]])
            r_N = get_position_regular_grid(dims, space.a, space.dx)
            sol_N = gaussian(r_N)
            sol_N_mps = MPS.from_vector(
                sol_N, [2] * sum(qubits_per_dimension), normalize=False
            )
            sol_int = interpolate_first_axis(sol_N.reshape(sol_N.shape[0], 1))
            sol_int /= np.linalg.norm(sol_int)
            sol_int_mps, _ = finite_differences_interpolation_1D(
                sol_N_mps, space, strategy=self.strategy
            )
            sol_int_mps = sol_int_mps.to_vector()
            sol_int_mps /= np.linalg.norm(sol_int_mps)
            self.assertSimilar(sol_int.flatten(), sol_int_mps)

    def test_finite_differences_interpolation(self):
        for n in range(3, 8):
            qubits_per_dimension = [n, n]
            dims = [2**n for n in qubits_per_dimension]
            L = 10
            space = Space(qubits_per_dimension, L=[[-L / 2, L / 2], [-L / 2, L / 2]])
            r_N = get_position_regular_grid(dims, space.a, space.dx)
            sol_N = gaussian(r_N)
            sol_N_mps = MPS.from_vector(
                sol_N, [2] * sum(qubits_per_dimension), normalize=False
            )
            sol_int = finite_differences_interpolation_vector_2D(sol_N)
            sol_int /= np.linalg.norm(sol_int)
            sol_int_mps = finite_differences_interpolation(
                sol_N_mps, space, strategy=self.strategy
            )
            sol_int_mps = sol_int_mps.to_vector()
            sol_int_mps /= np.linalg.norm(sol_int_mps)
            self.assertSimilar(sol_int.flatten(), sol_int_mps)
