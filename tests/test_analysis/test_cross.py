import numpy as np
from abc import abstractmethod
import unittest
from numpy.typing import NDArray
from seemps.state import MPS
from seemps.analysis.mesh import (
    ArrayInterval,
    ChebyshevInterval,
    Mesh,
    QuantizedInterval,
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
    CrossStrategyPRRLU,
)
from seemps.analysis.cross.cross import CrossResults
from seemps.analysis.cross.kernels import (
    combine_indices,
    indices_to_mesh_coordinates,
    maxvol_square,
    maxvol_rectangular,
    prrlu,
)
from seemps.analysis.cross.cross_greedy import CrossInterpolationGreedy
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

    def test_rook_search(self, n=5):
        """Rook search interpolates smooth functions (it may miss sparse ones)."""
        func, mesh, _, y = gaussian_setup_mps(2, n=n)
        map_matrix = mps_to_mesh_matrix([n, n])
        black_box = BlackBoxLoadMPS(func, mesh, map_matrix, [2] * (2 * n))
        strategy = CrossStrategyPRRLU(pivot_search="rook")
        cross_results = cross_prrlu(black_box, cross_strategy=strategy)
        self.assertSimilar(y, cross_results.mps.to_vector())


class TestSampling(SeeMPSTestCase):
    def test_combine_indices(self):
        left = np.array([[1, 2], [3, 4], [5, 6]])
        right = np.array([[0], [1]])
        expected = np.array([np.concatenate((x, y)) for x in left for y in right])
        row_major = np.array([np.concatenate((x, y)) for y in right for x in left])
        self.assertEqualTensors(combine_indices(left, right), expected)
        self.assertEqualTensors(combine_indices(left, right, row_major=True), row_major)

    def test_mesh_coordinates(self):
        """The native kernel matches indexing the mesh, and skips other intervals."""
        mesh = Mesh([QuantizedInterval(-2, 1, 3), ChebyshevInterval(-1, 1, 16)])
        map_matrix = mps_to_mesh_matrix([3, 4])
        blocks = (self.rng.integers(0, 2, (5, 3)), self.rng.integers(0, 2, (4, 4)))
        points = combine_indices(*blocks) @ map_matrix
        coordinates = indices_to_mesh_coordinates(mesh, blocks, map_matrix)
        self.assertSimilar(coordinates, mesh[points])
        array_mesh = Mesh([ArrayInterval([1, 2, 4])])
        self.assertIsNone(indices_to_mesh_coordinates(array_mesh, np.array([[0]])))

    def test_cache(self):
        """Cached sampling evaluates every distinct point only once."""
        mesh = Mesh([RegularInterval(0, 2, 3), RegularInterval(0, 3, 4)])
        box = BlackBoxLoadMPS(lambda x: x[0] + x[1], mesh)
        points = np.array([[1, 2], [0, 1], [1, 2], [2, 3]])
        cross = CrossStrategyPRRLU(cache_evals=True).make_interpolator(box)
        self.assertEqualTensors(cross.sample(points), box[points])
        self.assertEqualTensors(cross.sample(points[::-1]), box[points[::-1]])
        self.assertEqual(cross.evals, 3)


class TestSkeleton(SeeMPSTestCase):
    def random_matrix(self, m=1000, n=1000, r=5):
        """Computes a m x n random matrix of rank r"""
        return np.dot(self.rng.normal(size=(m, r)), self.rng.normal(size=(r, n)))

    def test_maxvol_square(self):
        A = self.random_matrix()
        J = self.rng.choice(A.shape[1], 5, replace=False)
        I, _ = maxvol_square(A[:, J])
        J, _ = maxvol_square(A[I, :].T)
        A_new = A[:, J] @ np.linalg.inv(A[I, :][:, J]) @ A[I, :]
        self.assertSimilar(A, A_new)

    def test_maxvol_rectangular(self):
        """Rectangular maxvol adds rows while preserving interpolation."""
        A = self.random_matrix(m=100, n=5, r=5)
        I, B = maxvol_rectangular(A, rank_kick=(1, 1))
        self.assertEqual(len(np.unique(I)), 6)
        self.assertSimilar(B @ A[I], A)
        self.assertSimilar(B[I], np.eye(6))

    def test_prrlu(self):
        """PRRLU reveals the rank and gives a cross interpolation of the matrix."""
        A = self.random_matrix(m=100, n=80, r=5)
        I, J, L, U = prrlu(A, 1e-10 * np.max(np.abs(A)), 80)
        self.assertEqual(len(I), 5)
        self.assertSimilar(L @ U, A)
        self.assertSimilar(A[:, J] @ np.linalg.inv(A[I, :][:, J]) @ A[I, :], A)
        self.assertEqual(len(prrlu(A, 0, 3)[0]), 3)


class TestGreedyStability(SeeMPSTestCase):
    def make_black_box(self, tensor):
        mps = MPS.from_vector(tensor.reshape(-1), list(tensor.shape), normalize=False)
        return BlackBoxComposeMPS(lambda values: values[0], [mps])

    def test_stable_core_construction(self):
        """Selected rows remain an identity despite large row-scale differences."""
        fiber = np.array([[1e-20, 1e-20], [3e-20, 6e-20], [1, 3], [1, -1], [0, 0]])
        selected_rows = np.array([0, 1])
        core = CrossInterpolationGreedy.fiber_to_core(
            fiber.reshape(1, 5, 2), selected_rows
        ).reshape(5, 2)
        self.assertEqualTensors(core[selected_rows], np.eye(2))
        self.assertSimilar(core @ fiber[selected_rows], fiber, atol=1e-14, rtol=1e-14)

    def test_nonzero_initial_points(self):
        """Starting away from the origin, greedy TCI recovers a full-rank tensor."""
        tensor = self.rng.uniform(0.5, 1.5, (3, 4, 5, 3))
        black_box = self.make_black_box(tensor)
        strategy = CrossStrategyGreedy(rng=np.random.default_rng(7))
        results = cross_greedy(black_box, np.array([1, 2, 3, 1]), strategy)
        self.assertSimilar(results.mps.to_vector(), tensor.reshape(-1))

    def test_roundoff_does_not_inflate_rank(self):
        """Cancellation noise does not add pivots beyond the matrix rank."""
        matrix = 1e8 * self.rng.normal(size=(256, 8)) @ self.rng.normal(size=(8, 256))
        strategy = CrossStrategyGreedy(rng=np.random.default_rng(12345))
        cross = CrossInterpolationGreedy(strategy, self.make_black_box(matrix), None)
        for _ in range(40):
            cross.update(0, True)
        self.assertEqual(len(cross.I_g[0]), 8)
        approximation = cross.mps.to_vector().reshape(matrix.shape)
        relative_error = np.max(np.abs(approximation - matrix)) / np.max(np.abs(matrix))
        self.assertLess(relative_error, 1e-13)
