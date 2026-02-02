from typing import Any
import numpy as np
from .common import CoreComparisonTestCase
import seemps.cython.core
import seemps.cython.pybind


class TestSVD(CoreComparisonTestCase):
    max_rows: int = 30
    max_cols: int = 30

    def make_double_arrays(
        self, dtype: Any = np.float64
    ) -> list[tuple[int, int, np.ndarray]]:
        """Generate a list of random double arrays for testing."""
        if self.test_args is None:
            self.test_args = [
                (rows, cols, self.rng.normal(size=(rows, cols)).astype(dtype))
                for rows in range(1, self.max_rows + 1)
                for cols in range(1, self.max_cols + 1)
                for copies in range(10)
            ]
        return self.test_args

    def make_complex_arrays(
        self, dtype: Any = np.complex128
    ) -> list[tuple[int, int, np.ndarray]]:
        """Generate a list of random double arrays for testing."""
        if self.test_args is None:
            self.test_args = [
                (
                    rows,
                    cols,
                    (
                        self.rng.normal(size=(rows, cols))
                        + 1j * self.rng.normal(size=(rows, cols))
                    ).astype(dtype),
                )
                for rows in range(1, self.max_rows + 1)
                for cols in range(1, self.max_cols + 1)
                for copies in range(10)
            ]
        return self.test_args

    def test_destructive_svd_double_are_same(self):
        """Test that the Cython and Pybind destructive SVD give the same results."""

        for rows, cols, A in self.make_double_arrays():
            with self.subTest(rows=rows, cols=cols):
                U1, S1, VT1 = seemps.cython.core._destructive_svd(A.copy())
                U2, S2, VT2 = seemps.cython.pybind._destructive_svd(A.copy())

                np.testing.assert_array_equal(U1, U2)
                np.testing.assert_array_equal(S1, S2)
                np.testing.assert_array_equal(VT1, VT2)

    def test_destructive_svd_complex_are_same(self):
        """Test that the Cython and Pybind destructive SVD give the same results."""

        for rows, cols, A in self.make_complex_arrays():
            with self.subTest(rows=rows, cols=cols):
                U1, S1, VT1 = seemps.cython.core._destructive_svd(A.copy())
                U2, S2, VT2 = seemps.cython.pybind._destructive_svd(A.copy())

                np.testing.assert_array_equal(U1, U2)
                np.testing.assert_array_equal(S1, S2)
                np.testing.assert_array_equal(VT1, VT2)

    def test_destructive_svd_other_real_types_are_same(self):
        """Test that the Cython and Pybind destructive SVD give the same results."""

        for dtype in [np.float32, np.int32, np.int64]:
            with self.subTest(dtype=dtype):
                for rows, cols, A in self.make_double_arrays(dtype=dtype):
                    with self.subTest(rows=rows, cols=cols):
                        U1, S1, VT1 = seemps.cython.core._destructive_svd(A.copy())
                        U2, S2, VT2 = seemps.cython.pybind._destructive_svd(A.copy())
                        U3, S3, VT3 = seemps.cython.core._destructive_svd(
                            A.astype(np.float64).copy()
                        )

                        self.assertTrue(U1.dtype == np.float64)
                        self.assertTrue(S1.dtype == np.float64)
                        self.assertTrue(VT1.dtype == np.float64)
                        self.assertTrue(U2.dtype == np.float64)
                        self.assertTrue(S2.dtype == np.float64)
                        self.assertTrue(VT2.dtype == np.float64)
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
                for rows, cols, A in self.make_complex_arrays(dtype=dtype):
                    with self.subTest(rows=rows, cols=cols):
                        U1, S1, VT1 = seemps.cython.core._destructive_svd(A.copy())
                        U2, S2, VT2 = seemps.cython.pybind._destructive_svd(A.copy())
                        U3, S3, VT3 = seemps.cython.core._destructive_svd(
                            A.astype(np.complex128).copy()
                        )

                        self.assertTrue(U1.dtype == np.complex128)
                        self.assertTrue(S1.dtype == np.float64)
                        self.assertTrue(VT1.dtype == np.complex128)
                        self.assertTrue(U2.dtype == np.complex128)
                        self.assertTrue(S2.dtype == np.float64)
                        self.assertTrue(VT2.dtype == np.complex128)
                        np.testing.assert_array_equal(U1, U2)
                        np.testing.assert_array_equal(S1, S2)
                        np.testing.assert_array_equal(VT1, VT2)
                        np.testing.assert_array_equal(U1, U3)
                        np.testing.assert_array_equal(S1, S3)
                        np.testing.assert_array_equal(VT1, VT3)
