import numpy as np
from .common import CoreComparisonTestCase
import seemps.cython.core
import seemps.cython.pybind


class TestGESDD(CoreComparisonTestCase):
    max_rows: int = 30
    max_cols: int = 30

    @classmethod
    def setupClass(cls):
        super().setUpClass()
        assert seemps.cython.core._get_svd_driver(), "gesdd"
        assert seemps.cython.pybind._get_svd_driver(), "gesdd"

    def test_destructive_svd_double_are_same(self):
        """Test that the Cython and Pybind destructive SVD give the same results."""

        for rows, cols, A in self.make_double_arrays(self.max_rows, self.max_cols):
            with self.subTest(rows=rows, cols=cols):
                U1, S1, VT1 = seemps.cython.core._destructive_svd(A.copy())
                U2, S2, VT2 = seemps.cython.pybind._destructive_svd(A.copy())

                self.assertEqual(U1.dtype, np.float64)
                self.assertEqual(S1.dtype, np.float64)
                self.assertEqual(VT1.dtype, np.float64)
                self.assertEqual(U2.dtype, np.float64)
                self.assertEqual(S2.dtype, np.float64)
                self.assertEqual(VT2.dtype, np.float64)
                np.testing.assert_array_equal(U1, U2)
                np.testing.assert_array_equal(S1, S2)
                np.testing.assert_array_equal(VT1, VT2)

    def test_destructive_svd_complex_are_same(self):
        """Test that the Cython and Pybind destructive SVD give the same results."""

        for rows, cols, A in self.make_complex_arrays(self.max_rows, self.max_cols):
            with self.subTest(rows=rows, cols=cols):
                U1, S1, VT1 = seemps.cython.core._destructive_svd(A.copy())
                U2, S2, VT2 = seemps.cython.pybind._destructive_svd(A.copy())

                self.assertEqual(U1.dtype, np.complex128)
                self.assertEqual(S1.dtype, np.float64)
                self.assertEqual(VT1.dtype, np.complex128)
                self.assertEqual(U2.dtype, np.complex128)
                self.assertEqual(S2.dtype, np.float64)
                self.assertEqual(VT2.dtype, np.complex128)
                np.testing.assert_array_equal(U1, U2)
                np.testing.assert_array_equal(S1, S2)
                np.testing.assert_array_equal(VT1, VT2)

    def test_destructive_svd_other_real_types_are_same(self):
        """Test that the Cython and Pybind destructive SVD give the same results."""

        for dtype in [np.float32, np.int32, np.int64]:
            with self.subTest(dtype=dtype):
                for rows, cols, A in self.make_double_arrays(
                    self.max_rows, self.max_cols, dtype=dtype
                ):
                    with self.subTest(rows=rows, cols=cols):
                        U1, S1, VT1 = seemps.cython.core._destructive_svd(A.copy())
                        U2, S2, VT2 = seemps.cython.pybind._destructive_svd(A.copy())
                        U3, S3, VT3 = seemps.cython.core._destructive_svd(
                            A.astype(np.float64).copy()
                        )

                        self.assertEqual(U1.dtype, np.float64)
                        self.assertEqual(S1.dtype, np.float64)
                        self.assertEqual(VT1.dtype, np.float64)
                        self.assertEqual(U2.dtype, np.float64)
                        self.assertEqual(S2.dtype, np.float64)
                        self.assertEqual(VT2.dtype, np.float64)
                        np.testing.assert_array_equal(U1, U2)
                        np.testing.assert_array_equal(S1, S2)
                        np.testing.assert_array_equal(VT1, VT2)
                        np.testing.assert_array_equal(U1, U3)
                        np.testing.assert_array_equal(S1, S3)
                        np.testing.assert_array_equal(VT1, VT3)

    def test_destructive_svd_other_complex_types_are_same(self):
        """Test that the Cython and Pybind destructive SVD give the same results."""

        for dtype in [np.complex64]:
            with self.subTest(dtype=dtype):
                for rows, cols, A in self.make_complex_arrays(
                    self.max_rows, self.max_cols, dtype=dtype
                ):
                    with self.subTest(rows=rows, cols=cols):
                        U1, S1, VT1 = seemps.cython.core._destructive_svd(A.copy())
                        U2, S2, VT2 = seemps.cython.pybind._destructive_svd(A.copy())
                        U3, S3, VT3 = seemps.cython.core._destructive_svd(
                            A.astype(np.complex128).copy()
                        )

                        self.assertEqual(U1.dtype, np.complex128)
                        self.assertEqual(S1.dtype, np.float64)
                        self.assertEqual(VT1.dtype, np.complex128)
                        self.assertEqual(U2.dtype, np.complex128)
                        self.assertEqual(S2.dtype, np.float64)
                        self.assertEqual(VT2.dtype, np.complex128)
                        np.testing.assert_array_equal(U1, U2)
                        np.testing.assert_array_equal(S1, S2)
                        np.testing.assert_array_equal(VT1, VT2)
                        np.testing.assert_array_equal(U1, U3)
                        np.testing.assert_array_equal(S1, S3)
                        np.testing.assert_array_equal(VT1, VT3)


class TestGESVD(TestGESDD):
    @classmethod
    def setupClass(cls):
        super().setUpClass()
        assert seemps.cython.core._get_svd_driver(), "gesdd"
        assert seemps.cython.pybind._get_svd_driver(), "gesdd"

    def setUp(self):
        super().setUp()
        seemps.cython.core._select_svd_driver("gesvd")
        seemps.cython.pybind._select_svd_driver("gesvd")

    def tearDown(self):
        seemps.cython.core._select_svd_driver("gesdd")
        seemps.cython.pybind._select_svd_driver("gesdd")
        return super().tearDown()

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        assert seemps.cython.core._get_svd_driver(), "gesdd"
        assert seemps.cython.pybind._get_svd_driver(), "gesdd"
