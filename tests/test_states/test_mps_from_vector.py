import numpy as np
from seemps.state import NO_TRUNCATION
from seemps.state.schmidt import _vector2mps
from .. import tools


class TestMPSFromVector(tools.TestCase):
    def join_tensors(self, state):
        w = np.ones((1, 1))
        for A in state:
            w = np.einsum("ia,ajb->ijb", w, A)
            w = w.reshape(-1, w.shape[-1])
        return w.reshape(-1)

    def test_mps_from_vector_on_one_site(self):
        v = self.rng.normal(size=5)
        state, err = _vector2mps(v, [5], strategy=NO_TRUNCATION, normalize=False)
        self.assertTrue(err >= 0)
        self.assertAlmostEqual(err, 0)
        self.assertEqual(len(state), 1)
        self.assertEqual(state[0].shape, (1, 5, 1))

    def test_mps_from_vector_on_different_sizes(self):
        v1 = self.rng.normal(size=2)
        v2 = self.rng.normal(size=3)
        v3 = self.rng.normal(size=4)
        w = v1[:, np.newaxis, np.newaxis] * v2[:, np.newaxis] * v3

        state, err = _vector2mps(
            w.reshape(-1), [2, 3, 4], strategy=NO_TRUNCATION, normalize=False
        )
        self.assertTrue(err >= 0)
        self.assertAlmostEqual(err, 0)
        self.assertEqual(len(state), 3)
        self.assertEqual(state[0].shape, (1, 2, 2))
        self.assertEqual(state[1].shape, (2, 3, 4))
        self.assertEqual(state[2].shape, (4, 4, 1))

        w = np.einsum("aib,bjc,ckd->ijk", state[0], state[1], state[2])
        self.assertSimilar(w.reshape(-1), w.reshape(-1))

    def test_mps_from_vector_on_random_qubit_states(self):
        for normalize in [False, True]:
            for N in range(1, 18):
                v = self.rng.normal(size=(2**N,))
                state, err = _vector2mps(
                    v, [2] * N, strategy=NO_TRUNCATION, normalize=normalize
                )

                self.assertTrue(err >= 0)
                self.assertAlmostEqual(err, 0)

                self.assertEqual(len(state), N)
                for i in range(N):
                    self.assertEqual(state[i].shape[1], 2)

                w = self.join_tensors(state)
                if normalize:
                    self.assertSimilar(w, v / np.linalg.norm(v))
                    self.assertAlmostEqual(np.linalg.norm(w), np.float64(1.0))
                else:
                    self.assertSimilar(w, v)

    def test_mps_from_vector_works_on_all_centers(self):
        for N in range(1, 10):
            v = self.rng.normal(size=(2**N,))
            for center in range(-N + 1, N):
                state, _ = _vector2mps(
                    v, [2] * N, center=center, strategy=NO_TRUNCATION, normalize=False
                )
                self.assertSimilar(self.join_tensors(state), v)

    def test_mps_from_vector_produces_isometries(self):
        for N in range(2, 10):
            v = self.rng.normal(size=(2**N,))
            for center in range(0, N):
                state, _ = _vector2mps(
                    v, [2] * N, center=center, strategy=NO_TRUNCATION
                )
                for i, A in enumerate(state):
                    if i < center:
                        self.assertApproximateIsometry(A, +1)
                    elif i > center:
                        self.assertApproximateIsometry(A, -1)

    def test_mps_from_vector_normalizes_central_tensor(self):
        for N in range(1, 10):
            v = self.rng.normal(size=(2**N,))
            for center in range(0, N):
                state, _ = _vector2mps(
                    v, [2] * N, center=center, normalize=True, strategy=NO_TRUNCATION
                )
                self.assertAlmostEqual(
                    np.linalg.norm(state[center].reshape(-1)), np.float64(1.0)
                )
