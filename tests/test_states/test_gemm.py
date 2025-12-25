import numpy as np
from seemps.cython.core import _gemm, GemmOrder
from .. import tools


class TestGEMM(tools.TestCase):
    def size_range(self, max_size: int = 6):
        for m in range(1, max_size):
            for k in range(1, max_size):
                for n in range(1, max_size):
                    yield (m, n, k)

    def test_real_product(self):
        for m, k, n in self.size_range():
            A = self.rng.normal(size=(m, k))
            B = self.rng.normal(size=(k, n))
            self.assertSimilar(
                np.matmul(A, B),
                _gemm(A, GemmOrder.NORMAL, B, GemmOrder.NORMAL),
                rtol=1e-10,
                atol=1e-15,
            )

    def test_real_product_transpose_first(self):
        for m, k, n in self.size_range():
            A = self.rng.normal(size=(k, m))
            B = self.rng.normal(size=(k, n))
            self.assertSimilar(
                np.matmul(A.T, B),
                _gemm(A, GemmOrder.TRANSPOSE, B, GemmOrder.NORMAL),
                rtol=1e-10,
                atol=1e-15,
            )

    def test_real_product_transpose_second(self):
        for m, k, n in self.size_range():
            A = self.rng.normal(size=(m, k))
            B = self.rng.normal(size=(n, k))
            self.assertSimilar(
                np.matmul(A, B.T),
                _gemm(A, GemmOrder.NORMAL, B, GemmOrder.TRANSPOSE),
                rtol=1e-10,
                atol=1e-15,
            )

    def _test_complex_product(self):
        for m, k, n in self.size_range():
            A = self.rng.normal(size=(m, k)) + 1j * self.rng.normal(size=(m, k))
            B = self.rng.normal(size=(k, n)) + 1j * self.rng.normal(size=(k, n))
            self.assertSimilar(
                np.matmul(A, B),
                _gemm(A, GemmOrder.NORMAL, B, GemmOrder.NORMAL),
                rtol=1e-10,
                atol=1e-15,
            )

    def test_complex_product_transpose_first(self):
        for m, k, n in self.size_range():
            A = self.rng.normal(size=(k, m)) + 1j * self.rng.normal(size=(k, m))
            B = self.rng.normal(size=(k, n)) + 1j * self.rng.normal(size=(k, n))
            self.assertSimilar(
                np.matmul(A.T, B),
                _gemm(A, GemmOrder.TRANSPOSE, B, GemmOrder.NORMAL),
                rtol=1e-10,
                atol=1e-15,
            )

    def test_complex_product_transpose_second(self):
        for m, k, n in self.size_range():
            A = self.rng.normal(size=(m, k)) + 1j * self.rng.normal(size=(m, k))
            B = self.rng.normal(size=(n, k)) + 1j * self.rng.normal(size=(n, k))
            self.assertSimilar(
                np.matmul(A, B.T),
                _gemm(A, GemmOrder.NORMAL, B, GemmOrder.TRANSPOSE),
                rtol=1e-10,
                atol=1e-15,
            )

    def test_complex_product_adjoint_first(self):
        for m, k, n in self.size_range():
            A = self.rng.normal(size=(k, m)) + 1j * self.rng.normal(size=(k, m))
            B = self.rng.normal(size=(k, n)) + 1j * self.rng.normal(size=(k, n))
            self.assertSimilar(
                np.matmul(A.T.conj(), B),
                _gemm(A, GemmOrder.ADJOINT, B, GemmOrder.NORMAL),
                rtol=1e-10,
                atol=1e-15,
            )

    def test_complex_product_adjoint_second(self):
        for m, k, n in self.size_range():
            A = self.rng.normal(size=(m, k)) + 1j * self.rng.normal(size=(m, k))
            B = self.rng.normal(size=(n, k)) + 1j * self.rng.normal(size=(n, k))
            self.assertSimilar(
                np.matmul(A, B.T.conj()),
                _gemm(A, GemmOrder.NORMAL, B, GemmOrder.ADJOINT),
                rtol=1e-10,
                atol=1e-15,
            )

    def test_gemm_works_with_views(self):
        A = self.rng.normal(size=(4, 4))
        B = self.rng.normal(size=(4, 4))
        self.assertSimilar(
            np.matmul(A, B.T), _gemm(A, GemmOrder.NORMAL, B.T, GemmOrder.NORMAL)
        )
        self.assertSimilar(
            np.matmul(A.T, B), _gemm(A.T, GemmOrder.NORMAL, B, GemmOrder.NORMAL)
        )
        self.assertSimilar(
            np.matmul(A.T, B.T), _gemm(A.T, GemmOrder.NORMAL, B.T, GemmOrder.NORMAL)
        )
