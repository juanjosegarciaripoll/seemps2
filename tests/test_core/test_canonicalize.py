import numpy as np
from .common import CoreComparisonTestCase
import seemps.cython.core
import seemps.cython.pybind


class TestCanonicalize(CoreComparisonTestCase):
    max_length: int = 10
    max_dim: int = 5
    max_bond_dim: int = 5

    def make_many_mps(self):
        return [
            (st1, st2, complex, center, state)
            for complex in (True, False)
            for L in range(2, self.max_length + 1)
            for d in range(2, self.max_dim + 1)
            for D in range(1, self.max_bond_dim + 1)
            for state in [self.random_uniform_mps(d=d, size=L, D=D, complex=complex)]
            for center in range(L)
            for st1, st2 in self.stategies
        ]

    def test_canonicalize_are_same(self):
        """Test that the Cython and Pybind _canonicalize give the same results for real MPS."""

        for st1, st2, complex, center, state in self.make_many_mps():
            with self.subTest(
                complex=complex,
                length=len(state),
                dim=state.physical_dimensions()[0],
                center=center,
                strategy=(st1, st2),
            ):
                state_cython = [A.copy() for A in state]
                state_pybind = [A.copy() for A in state]
                err1 = seemps.cython.core._canonicalize(state_cython, center, st1)
                err2 = seemps.cython.pybind._canonicalize(state_pybind, center, st2)

                self.assertEqual(err1, err2)
                for A1, A2 in zip(state_cython, state_pybind):
                    self.assertEqual(A1.dtype, np.complex128 if complex else np.float64)
                    self.assertEqual(A2.dtype, np.complex128 if complex else np.float64)
                    np.testing.assert_array_equal(A1, A2)

    def test_recanonicalize_are_same(self):
        """Test that the Cython and Pybind _recanonicalize give the same results for real MPS."""

        for st1, st2, complex, oldcenter, state in self.make_many_mps():
            d = state.physical_dimensions()[0]
            L = len(state)
            state = [A.copy() for A in state]
            seemps.cython.core._canonicalize(state, oldcenter, st1)
            for new_center in range(L):
                with self.subTest(
                    complex=complex,
                    length=L,
                    dim=d,
                    old_center=oldcenter,
                    center=new_center,
                    strategy=(st1, st2),
                ):
                    state_cython = [A.copy() for A in state]
                    state_pybind = [A.copy() for A in state]
                    err1 = seemps.cython.core._recanonicalize(
                        state_cython, oldcenter, new_center, st1
                    )
                    err2 = seemps.cython.pybind._recanonicalize(
                        state_pybind, oldcenter, new_center, st2
                    )

                    self.assertEqual(err1, err2)
                    for A1, A2 in zip(state_cython, state_pybind):
                        self.assertEqual(
                            A1.dtype, np.complex128 if complex else np.float64
                        )
                        self.assertEqual(
                            A2.dtype, np.complex128 if complex else np.float64
                        )
                        np.testing.assert_array_equal(A1, A2)
