import numpy as np
from abc import abstractmethod
import unittest
from numpy.typing import NDArray
from seemps.state import MPS
from seemps.typing import Matrix
from seemps.analysis.mesh import (
    Mesh,
    RegularInterval,
    mps_to_mesh_matrix,
    interleaving_permutation,
)
from seemps.analysis.cross import (
    BlackBoxLoadMPS,
    BlackBoxLoadMPO,
    BlackBoxComposeMPS,
    cross_maxvol,
    cross_dmrg,
    cross_greedy,
    cross_prrlu,
    CrossStrategyGreedy,
)
from seemps.analysis.cross.cross import maxvol_square, CrossResults
from seemps.analysis.cross.cross_maxvol import maxvol_rectangular
from seemps.analysis.cross.cross_greedy import CrossInterpolationGreedy
from seemps.analysis.cross.cross_prrlu import _prrlu
from seemps.operators import mps_as_mpo
from .tools_analysis import reorder_tensor
from ..tools import SeeMPSTestCase


def gaussian_setup_mps(dims, n=5, a=-1, b=1):
    def func(tensor):
        return np.exp(-(np.sum(tensor, axis=0) ** 2))

    intervals = [RegularInterval(a, b, 2**n) for _ in range(dims)]
    mesh = Mesh(intervals)  # type: ignore
    mesh_tensor = mesh.to_tensor()
    func_vector = func(mesh_tensor.T).reshape(-1)
    mps = MPS.from_vector(func_vector, [2] * (n * dims), normalize=False)
    return func, mesh, mps, func_vector


def _EPR_function(x: NDArray, y: NDArray) -> NDArray:
    return np.exp(-(x * x)) * ((x - y) == 0).astype(np.float64)


def _Gaussian_2d(x: NDArray, y: NDArray) -> NDArray:
    return np.exp(-(x * x + y * y))


def gaussian_setup_1d_mpo(is_diagonal, n=5, a=-1, b=1):
    interval = RegularInterval(a, b, 2**n)
    vec_x = interval.to_vector()
    mps_identity = MPS([np.ones((1, 2, 1))] * n)
    mesh = Mesh([interval, interval])
    if is_diagonal:
        func = _EPR_function
    else:
        func = _Gaussian_2d
    return func, vec_x, mesh, mps_identity


class CrossTests(SeeMPSTestCase):
    @classmethod
    def setUpClass(cls):
        if cls is CrossTests:
            raise unittest.SkipTest(f"Skip {cls} tests, it's a base class")
        super().setUpClass()

    @abstractmethod
    def cross_method(self, function, *args, **kwdargs):
        raise Exception("cross_method not implemented in " + str(type(self)))

    def test_load_1d_mps(self, n=5):
        func, mesh, _, y = gaussian_setup_mps(1, n=n)
        map_matrix = mps_to_mesh_matrix([n])
        physical_dimensions = [2] * n
        black_box = BlackBoxLoadMPS(func, mesh, map_matrix, physical_dimensions)
        cross_results: CrossResults = self.cross_method(black_box)
        self.assertSimilar(y, cross_results.mps.to_vector())

    def test_load_2d_mps(self, n=5):
        func, mesh, _, y = gaussian_setup_mps(2, n=n)
        map_matrix = mps_to_mesh_matrix([n, n])
        physical_dimensions = [2] * (2 * n)
        black_box = BlackBoxLoadMPS(func, mesh, map_matrix, physical_dimensions)
        cross_results: CrossResults = self.cross_method(black_box)
        self.assertSimilar(y, cross_results.mps.to_vector())

    def test_load_2d_mps_with_order_B(self, n=5):
        func, mesh, _, y = gaussian_setup_mps(2, n=n)
        permutation = interleaving_permutation([n, n])
        map_matrix = mps_to_mesh_matrix([n, n], permutation)
        physical_dimensions = [2] * (2 * n)
        black_box = BlackBoxLoadMPS(func, mesh, map_matrix, physical_dimensions)
        cross_results: CrossResults = self.cross_method(black_box)
        qubits = [int(np.log2(s)) for s in mesh.dimensions]
        tensor = reorder_tensor(cross_results.mps.to_vector(), qubits)
        self.assertSimilar(y, tensor)

    def test_load_2d_tt(self, n=5):
        func, mesh, _, y = gaussian_setup_mps(2, n=n)
        black_box = BlackBoxLoadMPS(func, mesh)
        cross_results: CrossResults = self.cross_method(black_box)
        vector = cross_results.mps.to_vector()
        self.assertSimilar(y, vector)

    def test_load_1d_mpo_diagonal(self, n=5):
        func, x, mesh, mps_I = gaussian_setup_1d_mpo(is_diagonal=True, n=n)
        black_box = BlackBoxLoadMPO(func, mesh, is_diagonal=True)
        cross_results: CrossResults = self.cross_method(black_box)
        mps_diagonal = mps_as_mpo(cross_results.mps).apply(mps_I)
        self.assertSimilar(func(x, x), mps_diagonal.to_vector())

    def test_load_1d_mpo_nondiagonal(self, n=5):
        func, x, mesh, _ = gaussian_setup_1d_mpo(is_diagonal=False, n=n)
        black_box = BlackBoxLoadMPO(func, mesh)
        cross_results: CrossResults = self.cross_method(black_box)
        y_mps = mps_as_mpo(cross_results.mps).to_matrix()
        xx, yy = np.meshgrid(x, x)
        self.assertSimilar(func(xx, yy), y_mps)

    def test_compose_1d_mps_list(self, n=5):
        _, _, mps_0, y_0 = gaussian_setup_mps(1, n=n)

        def func(v):
            return v[0] + np.sin(v[1]) + np.cos(v[2])

        black_box = BlackBoxComposeMPS(func, [mps_0, mps_0, mps_0])
        cross_results: CrossResults = self.cross_method(black_box)
        self.assertSimilar(func([y_0, y_0, y_0]), cross_results.mps.to_vector())

    def test_compose_2d_mps_list(self, n=5):
        _, _, mps_0, y_0 = gaussian_setup_mps(2, n=n)

        def func(v):
            return v[0] + np.sin(v[1]) + np.cos(v[2])

        black_box = BlackBoxComposeMPS(func, [mps_0, mps_0, mps_0])
        cross_results: CrossResults = self.cross_method(black_box)
        self.assertSimilar(func([y_0, y_0, y_0]), cross_results.mps.to_vector())


class TestCrossMaxvol(CrossTests):
    def cross_method(self, function, *args, **kwdargs):
        return cross_maxvol(function, *args, **kwdargs)


class TestCrossDMRG(CrossTests):
    def cross_method(self, function, *args, **kwdargs):
        return cross_dmrg(function, *args, **kwdargs)


class TestCrossGreedy(CrossTests):
    def cross_method(self, function, *args, **kwdargs):
        kwdargs["cross_strategy"] = CrossStrategyGreedy()
        return cross_greedy(function, *args, **kwdargs)


class TestCrossPRRLU(CrossTests):
    def cross_method(self, function, *args, **kwdargs):
        return cross_prrlu(function, *args, **kwdargs)


class TestSkeleton(SeeMPSTestCase):
    def random_matrix(self, m=1000, n=1000, r=5):
        """Computes a m x n random matrix of rank r"""
        return np.dot(self.rng.normal(size=(m, r)), self.rng.normal(size=(r, n)))

    def test_maxvol_square(self):
        A = self.random_matrix()
        J = self.rng.choice(A.shape[1], 5, replace=False)
        I: int | Matrix = 0
        for _ in range(1):
            C = A[:, J]
            I, _ = maxvol_square(C)
            R = A[I, :]
            J, _ = maxvol_square(R.T)
        A_new = A[:, J] @ np.linalg.inv(A[I, :][:, J]) @ A[I, :]
        self.assertSimilar(A, A_new)

    def test_maxvol_rectangular(self):
        A = self.random_matrix()
        J = self.rng.choice(A.shape[1], 1, replace=False)
        I: int | Matrix = 0
        for _ in range(2):
            C = A[:, J]
            I, _ = maxvol_rectangular(C, rank_kick=(0, 1))
            R = A[I, :]
            J, _ = maxvol_rectangular(R.T, rank_kick=(0, 1))
        I, _ = maxvol_square(A[:, J])
        A_new = A[:, J] @ np.linalg.inv(A[I, :][:, J]) @ A[I, :]
        self.assertSimilar(A, A_new)


class TestPRRLU(SeeMPSTestCase):
    def test_rank_revelation(self):
        for complex_values in (False, True):
            with self.subTest(complex_values=complex_values):
                left = self.rng.normal(size=(23, 4))
                right = self.rng.normal(size=(4, 17))
                if complex_values:
                    left = left + 1j * self.rng.normal(size=left.shape)
                    right = right + 1j * self.rng.normal(size=right.shape)
                matrix = left @ right
                original = matrix.copy()

                rows, columns, lower, upper = _prrlu(matrix, 1e-11, 17)

                self.assertEqual(len(rows), 4)
                self.assertEqual(len(columns), 4)
                self.assertSimilar(lower @ upper, matrix, atol=1e-12)
                self.assertEqualTensors(matrix, original)
                self.assertSimilar(lower[rows], np.tril(lower[rows]))
                self.assertSimilar(np.diag(lower[rows]), np.ones(4))
                self.assertSimilar(upper[:, columns], np.triu(upper[:, columns]))

    def test_truncation_and_rank_cap(self):
        matrix = np.diag([1e-14, 1.0, 1e-3, 1e-7])
        cases = ((1e-6, 4, 2), (0, 2, 2), (0, 4, 4))
        for tolerance, max_rank, expected_rank in cases:
            with self.subTest(tolerance=tolerance, max_rank=max_rank):
                rows, columns, lower, upper = _prrlu(matrix, tolerance, max_rank)
                self.assertEqual(len(rows), expected_rank)
                self.assertEqual(len(columns), expected_rank)
                self.assertLessEqual(
                    np.max(np.abs(matrix - lower @ upper)),
                    max(tolerance, 1e-7),
                )

    def test_zero_and_nonfinite_matrices(self):
        for dtype in (int, float, complex):
            with self.subTest(dtype=dtype):
                rows, columns, lower, upper = _prrlu(
                    np.zeros((3, 5), dtype=dtype), 0, 3
                )
                self.assertEqual(len(rows), 1)
                self.assertEqual(len(columns), 1)
                self.assertSimilar(lower @ upper, np.zeros((3, 5)))

        for value in (np.inf, np.nan):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    _prrlu(np.array([[value]]), 1e-10, 1)


class TestGreedyStability(SeeMPSTestCase):
    def assert_indices_are_consistent(self, cross):
        for k in range(cross.sites - 1):
            left = cross.combine_indices(cross.I_l[k], cross.I_s[k])
            right = cross.combine_indices(cross.I_s[k + 1], cross.I_g[k + 1])
            self.assertEqualTensors(left[cross.J_l[k + 1]], cross.I_l[k + 1])
            self.assertEqualTensors(right[cross.J_g[k]], cross.I_g[k])

    def test_stable_core_construction(self):
        """Selected rows remain an identity despite large row-scale differences."""
        fiber = np.array([[1e-20, 1e-20], [3e-20, 6e-20], [1, 3], [1, -1], [0, 0]])
        selected_rows = np.array([0, 1])
        core = CrossInterpolationGreedy.fiber_to_core(
            fiber.reshape(1, 5, 2), selected_rows
        ).reshape(5, 2)
        self.assertEqualTensors(core[selected_rows], np.eye(2))
        self.assertSimilar(core @ fiber[selected_rows], fiber, atol=1e-14, rtol=1e-14)

    def test_rook_search_keeps_matching_column(self):
        """A bounded rook search stores the column matching its final pivot."""
        tensor = np.array([[1.0, 1.0, 1.0], [1.0, 2.0, 3.0], [1.0, 4.0, 5.0]])
        mps = MPS.from_vector(tensor.reshape(-1), [3, 3], normalize=False)
        black_box = BlackBoxComposeMPS(lambda values: values[0], [mps])
        cross = CrossInterpolationGreedy(
            CrossStrategyGreedy(
                maxiter=1, initial_points=1, rng=np.random.default_rng(2)
            ),
            black_box,
            None,
        )
        cross.update(0, True)
        self.assertEqual(cross.I_g[0][-1, 0], 2)
        self.assert_indices_are_consistent(cross)

    def test_nonzero_initial_indices(self):
        """Right flattened indices keep the correct stride as bonds grow."""
        tensor = self.rng.uniform(0.5, 1.5, (3, 4, 5, 3))
        mps = MPS.from_vector(tensor.reshape(-1), list(tensor.shape), normalize=False)
        black_box = BlackBoxComposeMPS(lambda values: values[0], [mps])
        cross = CrossInterpolationGreedy(
            CrossStrategyGreedy(rng=np.random.default_rng(7)),
            black_box,
            np.array([1, 2, 3, 1]),
        )
        self.assert_indices_are_consistent(cross)
        for sweep in range(8):
            sites = range(3) if sweep % 2 == 0 else reversed(range(3))
            for k in sites:
                cross.update(k, sweep % 2 == 0)
                self.assert_indices_are_consistent(cross)

    def test_roundoff_does_not_inflate_rank(self):
        """Cancellation noise does not add pivots beyond the matrix rank."""
        left = self.rng.normal(size=(256, 8))
        right = self.rng.normal(size=(8, 256))
        matrix = 1e8 * (left @ right)
        mps = MPS.from_vector(matrix.reshape(-1), [256, 256], normalize=False)
        black_box = BlackBoxComposeMPS(lambda values: values[0], [mps])
        cross = CrossInterpolationGreedy(
            CrossStrategyGreedy(rng=np.random.default_rng(12345)), black_box, None
        )
        for _ in range(40):
            cross.update(0, True)
        self.assertEqual(len(cross.I_g[0]), 8)
        approximation = cross.mps.to_vector().reshape(matrix.shape)
        relative_error = np.max(np.abs(approximation - matrix)) / np.max(np.abs(matrix))
        self.assertLess(relative_error, 1e-13)
