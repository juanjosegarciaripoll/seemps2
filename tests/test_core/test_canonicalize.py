import numpy as np
from .common import CoreComparisonTestCase
import seemps.cython.core
import seemps.cython.pybind


class TestCanonicalize(CoreComparisonTestCase):
    max_length: int = 10
    max_dim: int = 5
    max_bond_dim: int = 5

    def make_many_mps(self, complex: bool = False):
        return [
            (center, state)
            for L in range(2, self.max_length + 1)
            for d in range(2, self.max_dim + 1)
            for D in range(1, self.max_bond_dim + 1)
            for state in [self.random_uniform_mps(d=d, size=L, D=D, complex=complex)]
            for center in range(L)
        ]

    def test_canonicalize_real_are_same(self):
        """Test that the Cython and Pybind canonicalize give the same results for real MPS."""

        for st1, st2 in self.stategies:
            for center, state in self.make_many_mps(complex=False):
                with self.subTest(
                    length=len(state),
                    dim=state.physical_dimensions()[0],
                    center=center,
                    strategy=(st1, st2),
                ):
                    state_cython = [A.copy() for A in state]
                    state_pybind = [A.copy() for A in state]
                    seemps.cython.core._canonicalize(state_cython, center, st1)
                    seemps.cython.pybind._canonicalize(state_pybind, center, st2)

                    for A1, A2 in zip(state_cython, state_pybind):
                        self.assertEqual(A1.dtype, np.float64)
                        self.assertEqual(A2.dtype, np.float64)
                        np.testing.assert_array_equal(A1, A2)

    def test_canonicalize_complex_are_same(self):
        """Test that the Cython and Pybind canonicalize give the same results for complex MPS."""

        for st1, st2 in self.stategies:
            for center, state in self.make_many_mps(complex=True):
                with self.subTest(
                    length=len(state),
                    dim=state.physical_dimensions()[0],
                    center=center,
                    strategy=(st1, st2),
                ):
                    state_cython = [A.copy() for A in state]
                    state_pybind = [A.copy() for A in state]
                    seemps.cython.core._canonicalize(state_cython, center, st1)
                    seemps.cython.pybind._canonicalize(state_pybind, center, st2)

                    for A1, A2 in zip(state_cython, state_pybind):
                        self.assertEqual(A1.dtype, np.complex128)
                        self.assertEqual(A2.dtype, np.complex128)
                        np.testing.assert_array_equal(A1, A2)
