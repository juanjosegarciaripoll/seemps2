import numpy as np

from seemps.analysis.evaluation import evaluate_mps
from seemps.state import MPS
from ..tools import SeeMPSTestCase


class TestEvaluateMPS(SeeMPSTestCase):
    def test_evaluate_mps(self):
        dimensions = (2, 3, 16, 2)
        for dtype in (np.float64, np.complex128):
            tensor = self.rng.normal(size=dimensions).astype(dtype)
            mps = MPS.from_vector(tensor.reshape(-1), list(dimensions), normalize=False)
            indices = np.column_stack(
                [self.rng.integers(-d, d, 100) for d in dimensions]
            )
            expected = tensor[tuple(indices.T)]
            self.assertSimilar(evaluate_mps(mps, indices), expected)
            self.assertSimilar(evaluate_mps(mps, indices[0]), expected[:1])

    def test_invalid_indices(self):
        mps = MPS([np.ones((1, 2, 3)), np.ones((3, 2, 1))])
        with self.assertRaises(ValueError):
            evaluate_mps(mps, np.zeros((2, 3), dtype=int))
        with self.assertRaises(IndexError):
            evaluate_mps(mps, np.array([[2, 0]]))
