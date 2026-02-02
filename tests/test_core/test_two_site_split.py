import numpy as np
from .common import CoreComparisonTestCase
import seemps.cython.core
import seemps.cython.pybind


class TesTwoSiteSplitLeft(CoreComparisonTestCase):
    max_bond_dimension: int = 6
    max_dim: int = 10

    def test_compare_two_site_split_left(self):
        """Compare _left_orth_2site implementations."""
        L = self.max_dim
        D = self.max_bond_dimension
        for (a, i, j, b), A in self.make_real_tensors((D, L, L, D)):
            for st1, st2 in self.stategies:
                with self.subTest(shape=(a, i, j, b), strategy=(st1, st2)):
                    Acython = A.copy()
                    Apybind = A.copy()
                    U1, V1, err1 = seemps.cython.core._left_orth_2site(Acython, st1)
                    U2, V2, err2 = seemps.cython.pybind._left_orth_2site(Apybind, st2)

                    self.assertEqual(U1.dtype, np.float64)
                    self.assertEqual(V1.dtype, np.float64)
                    self.assertEqual(U2.dtype, np.float64)
                    self.assertEqual(V2.dtype, np.float64)
                    np.testing.assert_array_equal(U1, U2)
                    np.testing.assert_array_equal(V1, V2)
                    np.testing.assert_array_equal(Acython, Apybind)
                    self.assertEqual(err1, err2)
                    self.assertEqual(U1.shape, (a, i, U1.shape[-1]))
                    self.assertEqual(V1.shape, (U1.shape[-1], j, b))

    def test_compare_two_site_split_right(self):
        """Compare _right_orth_2site implementations."""
        L = self.max_dim
        D = self.max_bond_dimension
        for (a, i, j, b), A in self.make_complex_tensors((D, L, L, D)):
            for st1, st2 in self.stategies:
                with self.subTest(shape=(a, i, j, b)):
                    Acython = A.copy()
                    Apybind = A.copy()
                    U1, V1, err1 = seemps.cython.core._left_orth_2site(Acython, st1)
                    U2, V2, err2 = seemps.cython.pybind._left_orth_2site(Apybind, st2)

                    self.assertEqual(U1.dtype, np.complex128)
                    self.assertEqual(V1.dtype, np.complex128)
                    self.assertEqual(U2.dtype, np.complex128)
                    self.assertEqual(V2.dtype, np.complex128)
                    np.testing.assert_array_equal(U1, U2)
                    np.testing.assert_array_equal(V1, V2)
                    np.testing.assert_array_equal(Acython, Apybind)
                    self.assertEqual(err1, err2)
                    self.assertEqual(U1.shape, (a, i, U1.shape[-1]))
                    self.assertEqual(V1.shape, (U1.shape[-1], j, b))
