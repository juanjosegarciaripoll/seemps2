import numpy as np
from typing import Any
import seemps.cython.core as core
import seemps.cython.pybind as pybind
from .common import CoreComparisonTestCase


class TestCoreGemm(CoreComparisonTestCase):
    max_rows: int = 30
    max_cols: int = 30

    def make_pairs_of_arrays(
        self,
    ) -> list[tuple[int, int, int, Any, np.ndarray, np.ndarray]]:
        """Generate a list of random double array pairs for testing."""
        if self.test_args is None:
            self.test_args = [
                (
                    rows,
                    n,
                    cols,
                    dtype,
                    self.random_tensor((rows, n), dtype),
                    self.random_tensor((n, cols), dtype),
                )
                for rows in range(1, self.max_rows + 1)
                for n in range(1, self.max_cols + 1)
                for cols in range(1, self.max_cols + 1)
                for dtype in [np.float64, np.complex128]
            ]
        return self.test_args

    def test_gemm_order_enum_are_same(self):
        """Test that the Cython and Pybind GemmOrder enums are the same."""
        self.assertEqual(core.GemmOrder.NORMAL, pybind.GemmOrder.NORMAL)
        self.assertEqual(core.GemmOrder.TRANSPOSE, pybind.GemmOrder.TRANSPOSE)
        self.assertEqual(core.GemmOrder.ADJOINT, pybind.GemmOrder.ADJOINT)

    def test_gemm_normal_are_same(self):
        """Test that the Cython and Pybind gemm give the same results."""
        for rows, n, cols, dtype, A, B in self.make_pairs_of_arrays():
            with self.subTest(rows=rows, n=n, cols=cols, dtype=dtype):
                Acython = A.copy()
                Bcython = B.copy()
                Ccython = core._gemm(
                    Acython,
                    core.GemmOrder.NORMAL,
                    Bcython,
                    core.GemmOrder.NORMAL,
                )
                Apybind = A.copy()
                Bpybind = B.copy()
                Cpybind = pybind._gemm(
                    Apybind,
                    pybind.GemmOrder.NORMAL,
                    Bpybind,
                    pybind.GemmOrder.NORMAL,
                )

                self.assertEqual(Ccython.dtype, A.dtype)
                self.assertEqual(Cpybind.dtype, A.dtype)
                np.testing.assert_array_equal(Acython, A)
                np.testing.assert_array_equal(Apybind, A)
                np.testing.assert_array_equal(Bcython, B)
                np.testing.assert_array_equal(Bpybind, B)
                np.testing.assert_array_equal(Ccython, Cpybind)
                np.testing.assert_almost_equal(Ccython, A @ B)

    def test_gemm_transpose_normal_are_same(self):
        """Test that the Cython and Pybind gemm give the same results."""
        for rows, n, cols, dtype, A, B in self.make_pairs_of_arrays():
            AT = np.ascontiguousarray(A.T)
            with self.subTest(rows=rows, n=n, cols=cols, dtype=dtype):
                ATcython = AT.copy()
                Bcython = B.copy()
                Ccython = core._gemm(
                    ATcython,
                    core.GemmOrder.TRANSPOSE,
                    Bcython,
                    core.GemmOrder.NORMAL,
                )
                ATpybind = AT.copy()
                Bpybind = B.copy()
                Cpybind = pybind._gemm(
                    ATpybind,
                    pybind.GemmOrder.TRANSPOSE,
                    Bpybind,
                    pybind.GemmOrder.NORMAL,
                )

                self.assertEqual(Ccython.dtype, A.dtype)
                self.assertEqual(Cpybind.dtype, A.dtype)
                np.testing.assert_array_equal(ATcython, AT)
                np.testing.assert_array_equal(ATpybind, AT)
                np.testing.assert_array_equal(Bcython, B)
                np.testing.assert_array_equal(Bpybind, B)
                np.testing.assert_array_equal(Ccython, Cpybind)
                np.testing.assert_almost_equal(Ccython, A @ B)

    def test_gemm_normal_transpose_are_same(self):
        """Test that the Cython and Pybind gemm give the same results."""
        for rows, n, cols, dtype, A, B in self.make_pairs_of_arrays():
            BT = np.ascontiguousarray(B.T)
            with self.subTest(rows=rows, n=n, cols=cols, dtype=dtype):
                Acython = A.copy()
                BTcython = BT.copy()
                Ccython = core._gemm(
                    Acython,
                    core.GemmOrder.NORMAL,
                    BTcython,
                    core.GemmOrder.TRANSPOSE,
                )
                Apybind = A.copy()
                BTpybind = BT.copy()
                Cpybind = pybind._gemm(
                    Apybind,
                    pybind.GemmOrder.NORMAL,
                    BTpybind,
                    pybind.GemmOrder.TRANSPOSE,
                )

                self.assertEqual(Ccython.dtype, A.dtype)
                self.assertEqual(Cpybind.dtype, A.dtype)
                np.testing.assert_array_equal(Acython, A)
                np.testing.assert_array_equal(Apybind, A)
                np.testing.assert_array_equal(BTcython, BT)
                np.testing.assert_array_equal(BTpybind, BT)
                np.testing.assert_array_equal(Ccython, Cpybind)
                np.testing.assert_almost_equal(Ccython, A @ B)

    def test_gemm_adjoint_normal_are_same(self):
        """Test that the Cython and Pybind gemm give the same results."""
        for rows, n, cols, dtype, A, B in self.make_pairs_of_arrays():
            AT = np.ascontiguousarray(A.T.conjugate())
            with self.subTest(rows=rows, n=n, cols=cols, dtype=dtype):
                ATcython = AT.copy()
                Bcython = B.copy()
                Ccython = core._gemm(
                    ATcython,
                    core.GemmOrder.ADJOINT,
                    Bcython,
                    core.GemmOrder.NORMAL,
                )
                ATpybind = AT.copy()
                Bpybind = B.copy()
                Cpybind = pybind._gemm(
                    ATpybind,
                    pybind.GemmOrder.ADJOINT,
                    Bpybind,
                    pybind.GemmOrder.NORMAL,
                )

                self.assertEqual(Ccython.dtype, A.dtype)
                self.assertEqual(Cpybind.dtype, A.dtype)
                np.testing.assert_array_equal(ATcython, AT)
                np.testing.assert_array_equal(ATpybind, AT)
                np.testing.assert_array_equal(Bcython, B)
                np.testing.assert_array_equal(Bpybind, B)
                np.testing.assert_array_equal(Ccython, Cpybind)
                np.testing.assert_almost_equal(Ccython, A @ B)

    def test_gemm_normal_adjoint_are_same(self):
        """Test that the Cython and Pybind gemm give the same results."""
        for rows, n, cols, dtype, A, B in self.make_pairs_of_arrays():
            BT = np.ascontiguousarray(B.T.conjugate())
            with self.subTest(rows=rows, n=n, cols=cols, dtype=dtype):
                Acython = A.copy()
                BTcython = BT.copy()
                Ccython = core._gemm(
                    Acython,
                    core.GemmOrder.NORMAL,
                    BTcython,
                    core.GemmOrder.ADJOINT,
                )
                Apybind = A.copy()
                BTpybind = BT.copy()
                Cpybind = pybind._gemm(
                    Apybind,
                    pybind.GemmOrder.NORMAL,
                    BTpybind,
                    pybind.GemmOrder.ADJOINT,
                )

                self.assertEqual(Ccython.dtype, A.dtype)
                self.assertEqual(Cpybind.dtype, A.dtype)
                np.testing.assert_array_equal(Acython, A)
                np.testing.assert_array_equal(Apybind, A)
                np.testing.assert_array_equal(BTcython, BT)
                np.testing.assert_array_equal(BTpybind, BT)
                np.testing.assert_array_equal(Ccython, Cpybind)
                np.testing.assert_almost_equal(Ccython, A @ B)

    def test_gemm_works_on_transposed_arrays(self):
        """Test that gemm works on transposed arrays."""
        for rows, n, cols, dtype, A, B in self.make_pairs_of_arrays():
            A = self.random_tensor((4, 3), dtype=dtype)
            B = self.random_tensor((3, 5), dtype=dtype)
            C3 = A @ B  # type: ignore

            C1 = core._gemm(A, core.GemmOrder.NORMAL, B.T, core.GemmOrder.TRANSPOSE)
            C2 = pybind._gemm(
                A, pybind.GemmOrder.NORMAL, B.T, pybind.GemmOrder.TRANSPOSE
            )
            np.testing.assert_array_almost_equal(C1, C2)
            np.testing.assert_array_almost_equal(C1, C3)

            C1 = core._gemm(A.T, core.GemmOrder.TRANSPOSE, B, core.GemmOrder.NORMAL)
            C2 = pybind._gemm(
                A.T, pybind.GemmOrder.TRANSPOSE, B, pybind.GemmOrder.NORMAL
            )
            np.testing.assert_array_almost_equal(C1, C2)
            np.testing.assert_array_almost_equal(C1, C3)

            C1 = core._gemm(
                A.T, core.GemmOrder.TRANSPOSE, B.T, core.GemmOrder.TRANSPOSE
            )
            C2 = pybind._gemm(
                A.T, pybind.GemmOrder.TRANSPOSE, B.T, pybind.GemmOrder.TRANSPOSE
            )
            np.testing.assert_array_almost_equal(C1, C2)
            np.testing.assert_array_almost_equal(C1, C3)
