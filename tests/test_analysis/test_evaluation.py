import numpy as np

from seemps.analysis.evaluation import evaluate_mps
from seemps.state import MPS
from ..tools import SeeMPSTestCase


class TestEvaluateMPS(SeeMPSTestCase):
    def test_evaluates_elements(self):
        mps = MPS(
            [
                np.array([[[1.0, 0.0], [0.0, 1.0]]]),
                np.array([[[2.0], [3.0]], [[5.0], [7.0]]]),
            ]
        )
        indices = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        self.assertEqualTensors(
            evaluate_mps(mps, indices), np.array([2.0, 3.0, 5.0, 7.0])
        )
        self.assertEqualTensors(evaluate_mps(mps, np.array([1, 0])), np.array([5.0]))

    def test_supported_dtypes_and_block_boundary(self):
        """Cython evaluation matches dense indexing for every supported dtype."""
        dimensions = (3, 4, 2, 5)
        ranks = (1, 16, 32, 17, 1)
        indices = np.column_stack(
            [self.rng.integers(-dimension, dimension, 65) for dimension in dimensions]
        )
        for dtype in (np.float32, np.float64, np.complex64, np.complex128):
            with self.subTest(dtype=dtype):
                cores = []
                for left, physical, right in zip(ranks, dimensions, ranks[1:]):
                    core = self.rng.normal(size=(left, physical, right))
                    if np.issubdtype(dtype, np.complexfloating):
                        core = core + 1j * self.rng.normal(size=core.shape)
                    cores.append((core / 8).astype(dtype))
                mps = MPS(cores)
                dense = mps.to_vector().reshape(dimensions)
                expected = dense[tuple(indices.T)]
                result = evaluate_mps(mps, indices)
                tolerance = 2e-5 if np.dtype(dtype).itemsize <= 8 else 2e-13
                self.assertEqual(result.dtype, np.dtype(dtype))
                self.assertSimilar(result, expected, atol=tolerance, rtol=tolerance)

    def test_mixed_strided_inputs_and_single_index(self):
        """Mixed core dtypes and noncontiguous index buffers retain NumPy semantics."""
        core0 = np.asfortranarray(self.rng.normal(size=(1, 2, 4)).astype(np.float32))
        core1 = self.rng.normal(size=(4, 3, 3))[:, ::-1, :]
        core2 = self.rng.normal(size=(3, 2, 1)).astype(np.complex128)
        core2 = core2 + 1j * self.rng.normal(size=core2.shape)
        mps = MPS([core0, core1, core2])
        dense = mps.to_vector().reshape(2, 3, 2)
        index_buffer = self.rng.integers(0, 2, size=(130, 3))
        index_buffer[:, 1] = self.rng.integers(0, 3, size=130)
        indices = index_buffer[::2]
        self.assertSimilar(
            evaluate_mps(mps, indices), dense[tuple(indices.T)], atol=2e-13, rtol=2e-13
        )
        self.assertSimilar(evaluate_mps(mps, np.array([1, -1, 0])), [dense[1, -1, 0]])

    def test_invalid_inputs(self):
        """Malformed indices and unsupported core dtypes fail with clear errors."""
        mps = MPS([np.ones((1, 2, 3)), np.ones((3, 2, 1))])
        with self.assertRaises(ValueError):
            evaluate_mps(mps, np.zeros((2, 3), dtype=int))
        with self.assertRaises(TypeError):
            evaluate_mps(mps, np.zeros((2, 2)))
        with self.assertRaises(IndexError):
            evaluate_mps(mps, np.array([[2, 0]]))
        with self.assertRaises(IndexError):
            evaluate_mps(mps, np.array([[-3, 0]]))
        integer_mps = MPS(
            [np.ones((1, 2, 3), dtype=int), np.ones((3, 2, 1), dtype=int)]
        )
        with self.assertRaises(TypeError):
            evaluate_mps(integer_mps, np.zeros((1, 2), dtype=int))

    def test_grouped_evaluation_and_tail_batch(self):
        dimensions = (2, 3, 2, 2)
        ranks = (1, 12, 20, 12, 1)
        for dtype in (np.float32, np.float64, np.complex64, np.complex128):
            cores = []
            for left, physical, right in zip(ranks, dimensions, ranks[1:]):
                core = self.rng.normal(size=(left, physical, right))
                if np.issubdtype(dtype, np.complexfloating):
                    core = core + 1j * self.rng.normal(size=core.shape)
                cores.append((core / 8).astype(dtype))
            mps = MPS(cores)
            indices = np.column_stack(
                [self.rng.integers(-d, d, 513) for d in dimensions]
            )
            # An empty physical group is legal, and invalid indices in a tail
            # batch must still be rejected after several grouped contractions.
            indices[:, 1] = -1
            dense = mps.to_vector().reshape(dimensions)
            atol = 1e-5 if dtype in (np.float32, np.complex64) else 1e-12
            self.assertSimilar(
                evaluate_mps(mps, indices),
                dense[tuple(indices.T)],
                atol=atol,
                rtol=atol,
            )
            indices[-1, 2] = dimensions[2]
            with self.assertRaises(IndexError):
                evaluate_mps(mps, indices)

    def test_grouped_evaluation_preserves_sample_order(self):
        dimensions = (2, 16, 3)
        ranks = (1, 12, 17, 1)
        cores = [
            self.rng.normal(size=(left, physical, right)) / 8
            for left, physical, right in zip(ranks, dimensions, ranks[1:])
        ]
        mps = MPS(cores)
        indices = np.column_stack([self.rng.integers(-d, d, 600) for d in dimensions])
        # All sixteen groups, repeated samples, and both index signs cross
        # multiple full batches and a grouped tail.
        indices[::7] = indices[0]
        expected = mps.to_vector().reshape(dimensions)[tuple(indices.T)]
        self.assertSimilar(evaluate_mps(mps, indices), expected, atol=1e-12)
