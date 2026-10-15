import numpy as np
from abc import abstractmethod
import unittest
from numpy.typing import NDArray
from seemps.state import MPS
from seemps.typing import Matrix
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
    CrossStrategyDMRG,
    CrossStrategyGreedy,
    CrossStrategyMaxvol,
    CrossStrategyPRRLU,
)
from seemps.analysis.cross.cross import (
    CrossError,
    CrossInterpolation,
    CrossResults,
)
from seemps.analysis.cross.cross_dmrg import (
    DEFAULT_CROSS_DMRG_STRATEGY,
    _compressed_svd,
)
from seemps.analysis.cross.cross_prrlu import _prrlu_core_left, _prrlu_core_right
from seemps.analysis.cross.kernels import (
    _combine_indices,
    _indices_to_mesh_coordinates,
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

    def test_default_rank_kick(self):
        self.assertEqual(CrossStrategyMaxvol().rank_kick, (1, 4))


class TestCrossDMRG(CrossTests):
    def cross_method(self, function, *args, **kwdargs):
        return cross_dmrg(function, *args, **kwdargs)

    def test_compressed_svd_preserves_error_budget(self):
        for complex_values in (False, True):
            left = self.rng.normal(size=(160, 12))
            right = self.rng.normal(size=(12, 192))
            if complex_values:
                left = left + 1j * self.rng.normal(size=left.shape)
                right = right + 1j * self.rng.normal(size=right.shape)
            matrix = left @ right
            matrix += 1e-9 * self.rng.normal(size=matrix.shape)
            original = matrix.copy()
            strategy = DEFAULT_CROSS_DMRG_STRATEGY.replace(tolerance=1e-10)
            cutoff = (
                0.01
                * np.sqrt(strategy.get_tolerance() / matrix.size)
                * np.linalg.norm(matrix)
            )
            U, S, V = _compressed_svd(matrix, cutoff, 40)
            self.assertEqual(len(S), 12)
            self.assertLess(
                np.linalg.norm(matrix - (U * S) @ V) / np.linalg.norm(matrix),
                1e-7,
            )
            self.assertSimilar(U.conj().T @ U, np.eye(len(S)), atol=1e-12)
            self.assertSimilar(V @ V.conj().T, np.eye(len(S)), atol=1e-12)
            self.assertEqualTensors(matrix, original)

    def test_compressed_svd_rank_limit(self):
        matrix = self.rng.normal(size=(128, 144))
        U, S, V = _compressed_svd(matrix, 0, 32)
        self.assertEqual(len(S), 32)
        self.assertSimilar(U.conj().T @ U, np.eye(32), atol=1e-12)
        self.assertSimilar(V @ V.conj().T, np.eye(32), atol=1e-12)
        U, S, V = _compressed_svd(np.zeros((128, 160)), 0, 32)
        self.assertEqual(len(S), 1)
        self.assertEqualTensors((U * S) @ V, np.zeros((128, 160)))


class TestCrossGreedy(CrossTests):
    def cross_method(self, function, *args, **kwdargs):
        kwdargs["cross_strategy"] = CrossStrategyGreedy()
        return cross_greedy(function, *args, **kwdargs)


class TestCrossPRRLU(CrossTests):
    def cross_method(self, function, *args, **kwdargs):
        return cross_prrlu(function, *args, **kwdargs)

    def test_binary_auto_search_uses_full_superblock(self):
        """Auto search factors a binary superblock once when rook covers it."""
        tensor = self.rng.normal(size=(2,) * 6)
        mesh = Mesh([RegularInterval(0, 2, 2)] * 6)
        points = np.indices((2, 2, 2)).reshape(3, -1).T
        rows = columns = points

        for forward in (False, True):
            box = BlackBoxLoadMPS(lambda x: tensor[tuple(x.astype(int))], mesh)
            cross = CrossStrategyPRRLU(
                full_pivot_threshold=0,
                rng=np.random.default_rng(15),
            ).make_interpolator(box)
            if forward:
                cross.I_g[2] = points[::2]
            else:
                cross.I_l[3] = points[::2]

            I, J, L, U, carry = cross._factor(rows, columns, 2, forward)
            matrix = tensor.reshape(8, 8)
            self.assertSimilar(L @ U, matrix)
            self.assertEqualTensors(carry, matrix[I] if forward else matrix[:, J])
            self.assertEqual(cross.evals, tensor.size)


class TestSampleCache(SeeMPSTestCase):
    def test_repeated_and_overlapping_samples(self):
        """Repeated tensor indices are evaluated once and returned in order."""
        mesh = Mesh([RegularInterval(0, 2, 3), RegularInterval(0, 3, 4)])
        batches = []

        def function(coordinates):
            batches.append(coordinates.shape[1])
            return np.asarray(coordinates[0] + 1j * coordinates[1], dtype=np.complex64)

        box = BlackBoxLoadMPS(function, mesh)
        cross = CrossInterpolation(box, cache_samples=True)
        points = np.array([[1, 2], [0, 1], [1, 2], [2, 3]])
        expected = function(mesh[points].T)
        batches.clear()

        values = cross.sample(points)
        self.assertEqual(values.dtype, np.dtype(np.complex64))
        self.assertEqualTensors(values, expected)
        self.assertEqual(cross.evals, 3)
        self.assertEqual(len(cross.cache), 3)

        values[:] = 99
        self.assertEqualTensors(cross.sample(points[::-1]), expected[::-1])
        self.assertEqual(cross.evals, 3)
        cross.sample(np.array([[1, 2], [1, 1]]))
        self.assertEqual(cross.evals, 4)
        self.assertEqual(batches, [3, 1])

        uncached = CrossInterpolation(box)
        uncached.sample(points)
        uncached.sample(points)
        self.assertEqual(uncached.evals, 2 * len(points))
        self.assertEqual(cross.evals, 4)
        self.assertFalse(uncached.cache)

        other = CrossInterpolation(box, cache_samples=True)
        self.assertEqualTensors(other.sample(points), expected)
        self.assertEqual(other.evals, 3)
        self.assertEqual(cross.evals, 4)
        self.assertEqual(batches, [3, 1, 4, 4, 3])

    def test_cache_is_used_by_all_cross_variants(self):
        """Every TCI variant forwards its cache setting to the interpolator."""
        tensor = np.arange(1, 7, dtype=float).reshape(2, 3)
        mps = MPS.from_vector(tensor.reshape(-1), [2, 3], normalize=False)
        points = np.array([[0, 0], [1, 2]])
        strategies = (
            CrossStrategyDMRG(cache_samples=True),
            CrossStrategyGreedy(cache_samples=True),
            CrossStrategyMaxvol(cache_samples=True),
            CrossStrategyPRRLU(cache_samples=True),
        )
        for strategy in strategies:
            with self.subTest(strategy=type(strategy).__name__):
                box = BlackBoxComposeMPS(lambda values: values[0], [mps])
                cross = strategy.make_interpolator(box)
                self.assertTrue(cross.cache_samples)
                cross.update(0, True)
                self.assertTrue(cross.cache)
                cross.sample(points)
                evaluations = cross.evals
                cross.sample(points)
                self.assertEqual(cross.evals, evaluations)

        self.assertFalse(CrossStrategyDMRG().cache_samples)
        self.assertFalse(CrossStrategyGreedy().cache_samples)
        self.assertFalse(CrossStrategyMaxvol().cache_samples)
        self.assertFalse(CrossStrategyPRRLU().cache_samples)

    def test_error_samples_share_the_cross_cache(self):
        """Global error samples enter the same cache as sweep samples."""
        tensor = np.arange(6, dtype=float).reshape(2, 3)
        mps = MPS.from_vector(tensor.reshape(-1), [2, 3], normalize=False)
        box = BlackBoxComposeMPS(lambda values: values[0], [mps])
        cross = CrossInterpolation(box, cache_samples=True)
        error = CrossError(
            CrossStrategyDMRG(num_samples=8, rng=np.random.default_rng(91))
        )
        error.sample_error(cross)
        indices = error.mps_indices
        self.assertIsNotNone(indices)
        if indices is not None:
            for point in indices.tolist():
                self.assertIn(tuple(point), cross.cache)


class TestMeshCoordinates(SeeMPSTestCase):
    def test_cartesian_repeated_coordinate_patterns(self):
        # Cover short patterns copied many times, long constant runs, negative
        # indices, and zero-only blocks alongside nonzero contributions.
        for interval in (
            RegularInterval(-3, 2, 8),
            ChebyshevInterval(-3, 2, 8, endpoints=True),
            ChebyshevInterval(-3, 2, 8, endpoints=False),
        ):
            mesh = Mesh([interval] * 3)
            blocks = (
                np.array([[0], [-1], [3]], dtype=np.intp),
                np.zeros((5, 1), dtype=np.intp),
                np.array([[2], [0], [-8], [7]], dtype=np.intp),
            )
            expected = mesh[_combine_indices(*blocks)]
            self.assertSimilar(
                _indices_to_mesh_coordinates(mesh, blocks),
                expected,
                atol=3e-15,
                rtol=3e-15,
            )
            mapping = np.eye(3, dtype=np.intp)[::-1].copy()
            self.assertSimilar(
                _indices_to_mesh_coordinates(mesh, blocks, mapping),
                mesh[_combine_indices(*blocks) @ mapping],
                atol=3e-15,
                rtol=3e-15,
            )
            with self.assertRaises(IndexError):
                _indices_to_mesh_coordinates(
                    mesh, (blocks[0], blocks[1], np.array([[-9]], dtype=np.intp))
                )

    def test_cartesian_coordinates_match_materialized_indices(self):
        mesh = Mesh([RegularInterval(-2, 3, 16), ChebyshevInterval(-1, 2, 8)])
        mapping = mps_to_mesh_matrix([4, 3])
        blocks = (
            self.rng.integers(0, 2, (7, 2)),
            self.rng.integers(0, 2, (5, 3)),
            self.rng.integers(0, 2, (9, 2)),
        )
        for permutation in (np.arange(7), np.array([0, 4, 1, 5, 2, 6, 3])):
            mixed_mapping = mapping[permutation]
            actual = _indices_to_mesh_coordinates(mesh, blocks, mixed_mapping)
            expected = mesh[_combine_indices(*blocks) @ mixed_mapping]
            self.assertSimilar(actual, expected, atol=3e-15, rtol=3e-15)
        blocks = (np.arange(16).reshape(-1, 1)[::-2], np.arange(8).reshape(-1, 1))
        self.assertSimilar(
            _indices_to_mesh_coordinates(mesh, blocks),
            mesh[_combine_indices(*blocks)],
            atol=3e-15,
        )
        empty = (np.empty((0, 1), dtype=int), blocks[1])
        self.assertEqual(_indices_to_mesh_coordinates(mesh, empty).shape, (0, 2))
        with self.assertRaises(IndexError):
            _indices_to_mesh_coordinates(mesh, (np.array([[16]]), blocks[1]))
        # Verify that dispatch and accounting work without changing callback batches.
        blocks = (
            np.arange(16).reshape(-1, 1),
            np.tile(np.arange(8), 80).reshape(-1, 1),
        )
        box = BlackBoxLoadMPS(lambda x: np.exp(x[0]) + x[1], mesh)
        cross = CrossInterpolation(box)
        self.assertSimilar(
            cross.sample(*blocks), box[_combine_indices(*blocks)], atol=1e-13
        )
        self.assertEqual(cross.evals, 16 * 8 * 80)

    def test_regular_and_chebyshev_intervals(self):
        """Native coordinates match every endpoint convention and direction."""
        indices = np.arange(-17, 17, dtype=np.intp).reshape(-1, 1)
        for endpoint_left in (False, True):
            for endpoint_right in (False, True):
                for start, stop in ((-2.0, 3.0), (3.0, -2.0), (1.0, 1.0)):
                    interval = RegularInterval(
                        start, stop, 17, endpoint_left, endpoint_right
                    )
                    mesh = Mesh([interval])  # type: ignore
                    actual = _indices_to_mesh_coordinates(mesh, indices)
                    self.assertIsNotNone(actual)
                    if actual is not None:
                        self.assertSimilar(
                            actual, mesh[indices], atol=3e-15, rtol=3e-15
                        )

        for endpoints in (False, True):
            for start, stop in ((-1.0, 1.0), (-2.0, 5.0), (1.0, -1.0)):
                mesh = Mesh([ChebyshevInterval(start, stop, 17, endpoints)])  # type: ignore
                actual = _indices_to_mesh_coordinates(mesh, indices)
                self.assertIsNotNone(actual)
                if actual is not None:
                    self.assertSimilar(actual, mesh[indices], atol=3e-15, rtol=3e-15)

    def test_mixed_quantized_mapping_and_black_box(self):
        """Mapped mixed meshes use the native path and preserve callback layout."""
        mesh = Mesh(
            [
                QuantizedInterval(-2, 1, 3),
                ChebyshevInterval(-1, 1, 16, endpoints=True),
                RegularInterval(1, 3, 8, False, True),
            ]
        )  # type: ignore
        mapping = mps_to_mesh_matrix([3, 4, 3])
        points = self.rng.integers(0, 2, size=(113, 10), dtype=np.intp)
        strided_points = points[::-2, ::-1]
        strided_mapping = mapping[::-1]
        strided_points.flags.writeable = False
        strided_mapping.flags.writeable = False
        actual = _indices_to_mesh_coordinates(mesh, strided_points, strided_mapping)
        self.assertIsNotNone(actual)
        if actual is not None:
            self.assertSimilar(
                actual,
                mesh[strided_points @ strided_mapping],
                atol=3e-15,
                rtol=3e-15,
            )
        seen = []

        def function(coordinates):
            seen.append(coordinates.copy())
            return np.sum(coordinates, axis=0)

        box = BlackBoxLoadMPS(function, mesh, mapping, [2] * 10)
        expected = mesh[points @ mapping]
        self.assertSimilar(box[points], np.sum(expected, axis=1))
        self.assertSimilar(seen[-1], expected.T, atol=3e-15, rtol=3e-15)

    def test_fallbacks_and_bounds(self):
        """Unsupported inputs fall back, while native bounds remain checked."""
        regular_mesh = Mesh([RegularInterval(-1, 1, 8)])  # type: ignore
        for dtype in (np.int32, np.uint64):
            indices = np.array([[0], [1]], dtype=dtype)
            self.assertSimilar(
                _indices_to_mesh_coordinates(regular_mesh, indices),
                regular_mesh[indices],
            )
        indices = np.array([[0], [1]], dtype=np.float64)
        self.assertIsNone(_indices_to_mesh_coordinates(regular_mesh, indices))

        array_mesh = Mesh([ArrayInterval([1, 2, 4])])  # type: ignore
        indices = np.array([[0], [2]], dtype=np.intp)
        self.assertIsNone(_indices_to_mesh_coordinates(array_mesh, indices))
        box = BlackBoxLoadMPS(lambda coordinates: coordinates, array_mesh)
        self.assertEqualTensors(box[indices], array_mesh[indices].T)

        for index in (-9, 8):
            with self.assertRaises(IndexError):
                _indices_to_mesh_coordinates(
                    regular_mesh, np.array([[index]], dtype=np.intp)
                )
        degenerate = Mesh([ChebyshevInterval(-1, 1, 1, endpoints=True)])  # type: ignore
        self.assertIsNone(
            _indices_to_mesh_coordinates(
                degenerate, np.array([[0]], dtype=np.intp)
            )
        )


class TestCombineIndices(SeeMPSTestCase):
    def test_product_order_and_dtype(self):
        """The product preserves block order, dtype, and input arrays."""
        left = np.arange(24, dtype=np.int32).reshape(4, 6)[::-1, ::2]
        right = np.arange(4, dtype=np.int16).reshape(2, 2)
        left.flags.writeable = False
        expected = np.array([np.concatenate((x, y)) for x in left for y in right])
        expected_row_major = np.array(
            [np.concatenate((x, y)) for y in right for x in left]
        )

        result = _combine_indices(left, right)
        self.assertEqual(result.dtype, np.result_type(left, right))
        self.assertEqualTensors(result, expected)
        self.assertEqualTensors(
            _combine_indices(left, right, row_major=True), expected_row_major
        )
        self.assertIs(_combine_indices(left), left)
        self.assertFalse(hasattr(CrossInterpolation, "combine_indices"))

    def test_empty_and_invalid_blocks(self):
        """Empty products are valid while malformed index blocks are rejected."""
        empty_product = _combine_indices()
        self.assertEqual(empty_product.shape, (1, 0))
        self.assertEqual(empty_product.dtype, np.dtype(np.intp))
        self.assertEqual(
            _combine_indices(np.empty((0, 1), dtype=int)).shape,
            (0, 1),
        )
        self.assertEqual(
            _combine_indices(
                np.empty((0, 1), dtype=int), np.ones((2, 2), dtype=int)
            ).shape,
            (0, 3),
        )

        for block in (np.zeros(3, dtype=int), np.zeros((2, 2), dtype=float)):
            with self.subTest(shape=block.shape, dtype=block.dtype):
                with self.assertRaises(ValueError):
                    _combine_indices(block)


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
        """Rectangular maxvol adds rows while preserving interpolation."""
        A = self.random_matrix(m=100, n=5, r=5)
        I, B = maxvol_rectangular(A, rank_kick=(1, 1))
        self.assertEqual(len(I), 6)
        self.assertEqual(len(np.unique(I)), 6)
        self.assertSimilar(B @ A[I], A)
        self.assertSimilar(B[I], np.eye(6))


class TestPRRLU(SeeMPSTestCase):
    def test_tied_pivots_and_residual_cutoff(self):
        # Equal maxima must still select the first row/column, and the
        # elimination stops at an exact cutoff.
        for width in (7, 8, 9, 17):
            matrix = np.zeros((9, width))
            matrix[np.arange(7), np.arange(7)] = [8, 4, 4, 2, 2, 1, 1]
            rows, columns, lower, upper = prrlu(matrix, 1, 9)
            self.assertEqualTensors(rows, np.arange(5))
            self.assertEqualTensors(columns, np.arange(5))
            self.assertLessEqual(np.max(np.abs(matrix - lower @ upper)), 1)

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
                rows, columns, lower, upper = prrlu(matrix, 1e-11, 17)
                self.assertEqual(len(rows), 4)
                self.assertEqual(len(columns), 4)
                self.assertSimilar(lower @ upper, matrix, atol=1e-12)
                self.assertEqualTensors(matrix, original)
                self.assertSimilar(lower[rows], np.tril(lower[rows]))
                self.assertSimilar(np.diag(lower[rows]), np.ones(4))
                self.assertSimilar(upper[:, columns], np.triu(upper[:, columns]))

    def test_interpolation_gauges(self):
        """Gauges reproduce a matrix from its pivot rows or columns."""
        for shape in ((17, 13), (131, 129)):
            for complex_values in (False, True):
                with self.subTest(shape=shape, complex_values=complex_values):
                    matrix = self.rng.normal(size=shape)
                    if complex_values:
                        matrix = matrix + 1j * self.rng.normal(size=shape)
                    rows, columns, lower, upper = prrlu(matrix, 0, min(shape))
                    original_lower, original_upper = lower.copy(), upper.copy()
                    left = _prrlu_core_left(lower, rows)
                    right = _prrlu_core_right(upper, columns)

                    self.assertSimilar(left @ matrix[rows], matrix, atol=1e-11)
                    self.assertSimilar(matrix[:, columns] @ right, matrix, atol=1e-11)
                    self.assertSimilar(left[rows], np.eye(len(rows)), atol=1e-12)
                    self.assertSimilar(
                        right[:, columns], np.eye(len(columns)), atol=1e-12
                    )
                    self.assertEqualTensors(lower, original_lower)
                    self.assertEqualTensors(upper, original_upper)

        rows, columns, lower, upper = prrlu(np.zeros((3, 5)), 0, 3)
        self.assertEqualTensors(_prrlu_core_left(lower, rows), np.ones((3, 1)))
        self.assertEqualTensors(_prrlu_core_right(upper, columns), np.ones((1, 5)))

    def test_truncation_and_rank_cap(self):
        matrix = np.diag([1e-14, 1.0, 1e-3, 1e-7])
        cases = ((1e-6, 4, 2), (0, 2, 2), (0, 4, 4))
        for tolerance, max_rank, expected_rank in cases:
            with self.subTest(tolerance=tolerance, max_rank=max_rank):
                rows, columns, lower, upper = prrlu(matrix, tolerance, max_rank)
                self.assertEqual(len(rows), expected_rank)
                self.assertEqual(len(columns), expected_rank)
                self.assertLessEqual(
                    np.max(np.abs(matrix - lower @ upper)),
                    max(tolerance, 1e-7),
                )

    def test_zero_and_nonfinite_matrices(self):
        for dtype in (int, float, complex):
            with self.subTest(dtype=dtype):
                rows, columns, lower, upper = prrlu(np.zeros((3, 5), dtype=dtype), 0, 3)
                self.assertEqual(len(rows), 1)
                self.assertEqual(len(columns), 1)
                self.assertSimilar(lower @ upper, np.zeros((3, 5)))

        for value in (np.inf, np.nan):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    prrlu(np.array([[value]]), 1e-10, 1)


class TestGreedyStability(SeeMPSTestCase):
    def assert_indices_are_consistent(self, cross):
        for k in range(cross.sites - 1):
            left = _combine_indices(cross.I_l[k], cross.I_s[k])
            right = _combine_indices(cross.I_s[k + 1], cross.I_g[k + 1])
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
