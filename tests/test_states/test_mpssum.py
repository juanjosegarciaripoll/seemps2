import numpy as np

from seemps.state import MPS, MPSSum

from ..fixture_mps_states import MPSStatesFixture


class TestMPSSum(MPSStatesFixture):
    def test_mpssum_requires_non_empty_list(self):
        with self.assertRaises(Exception):
            B = MPSSum([], [])  # type: ignore # noqa: F841

    def test_mpssum_size(self):
        A = MPS(self.product_state)
        B = MPSSum([1], [A])
        self.assertEqual(B.size, A.size)

    def make_simple_sum(self):
        A = MPS(self.product_state)
        B = MPS(self.product_state.copy())
        return MPSSum(weights=[1, 2], states=[A, B])

    def test_mpssum_init_does_not_copy_data(self):
        A = self.make_simple_sum()
        B = MPSSum(A.weights, A.states)
        self.assertTrue(B.weights is not A.weights)
        self.assertTrue(B.states is not A.states)

    def test_mpssum_copy_is_shallow(self):
        A = self.make_simple_sum()
        B = A.copy()
        self.assertTrue(B.weights is not A.weights)
        self.assertTrue(B.states is not A.states)

    def test_simple_sums(self):
        A = MPS(self.product_state)
        B = MPS(self.product_state.copy())
        C = MPSSum(weights=[1, 2], states=[A, B])
        self.assertTrue(C.weights == [1, 2])
        self.assertTrue(C.states == [A, B])

    def test_simple_subtractions(self):
        A = MPS(self.product_state)
        B = MPS(self.product_state.copy())
        C = A - B
        self.assertIsInstance(C, MPSSum)
        self.assertTrue(C.weights == [1, -1])
        self.assertTrue(C.states == [A, B])

    def test_scalar_multiplication_only_changes_weights(self):
        A = self.make_simple_sum()
        B = A * 0.5
        self.assertIsInstance(B, MPSSum)
        self.assertTrue(all(wb == 0.5 * wa for wa, wb in zip(A.weights, B.weights)))
        self.assertEqual(A.states, B.states)

        C = 0.5 * A
        self.assertIsInstance(C, MPSSum)
        self.assertEqual(B.weights, C.weights)
        self.assertEqual(A.states, C.states)

    def test_mpssum_accepts_mpssums_as_arguments(self):
        A = MPS(self.product_state)
        B = MPSSum([0.5, -3], [A.copy(), A.copy()])
        C = MPSSum([1, 0.5], [A, B])
        self.assertIsInstance(C, MPSSum)
        self.assertSimilar(C.weights, [1, 0.5 * 0.5, -0.5 * 3])
        self.assertTrue(C.states[0] is A)
        self.assertTrue(C.states[1] is B.states[0])
        self.assertTrue(C.states[2] is B.states[1])

    def test_addition_mpssum_and_mps(self):
        A = MPS(self.product_state)
        B = MPSSum(weights=[0.5], states=[A])
        C = MPS(self.inhomogeneous_state.copy())
        D = B + C
        self.assertEqual(D.weights, [0.5, 1])
        self.assertEqual(D.states, [A, C])

    def test_subtraction_mpssum_and_mps(self):
        A = MPS(self.product_state)
        B = MPSSum(weights=[0.5], states=[A])
        C = MPS(self.inhomogeneous_state.copy())
        D = B - C
        self.assertEqual(D.weights, [0.5, -1])
        self.assertEqual(D.states, [A, C])

    def test_subtraction_mps_and_mpssum(self):
        A = MPS(self.product_state)
        B = MPSSum(weights=[0.5], states=[A])
        C = MPS(self.inhomogeneous_state.copy())
        D = C - B
        self.assertEqual(D.weights, [1, -0.5])
        self.assertEqual(D.states, [C, A])

    def test_subtraction_mpssum_and_mpsum(self):
        A = self.make_simple_sum()
        D = A - A
        self.assertEqual(D.weights, [1, 2, -1, -2])
        self.assertEqual(D.states, A.states + A.states)

    def test_mpssum_to_vector(self):
        A = MPS(self.product_state)
        B = MPS(self.product_state.copy())
        C = MPSSum(weights=[0.5, -1.0], states=[A, B])
        self.assertTrue(np.all((0.5 * A.to_vector() - B.to_vector()) == C.to_vector()))

    def test_mpssum_join_produces_right_size_tensors(self):
        A = MPS(self.product_state)
        B = MPS(self.product_state.copy())
        C = MPSSum(weights=[0.5, -1.0], states=[A, B]).join()
        for i, A in enumerate(C):
            if i > 0:
                self.assertEqual(A.shape[0], 2)
            if i < C.size - 1:
                self.assertEqual(A.shape[2], 2)
            self.assertEqual(A.shape[1], 2)

    def test_mpssum_join_produces_sum(self):
        A = MPS(self.product_state)
        B = MPS(self.product_state.copy())
        C = MPSSum(weights=[0.5, -1.0], states=[A, B]).join()
        self.assertSimilar(0.5 * A.to_vector() - B.to_vector(), C.to_vector())

    def test_mpssum_norm_squared(self):
        A = self.random_uniform_mps(2, 3)
        B = self.random_uniform_mps(2, 3)
        C = MPSSum([1.0, 1j], [A, B])
        n2 = C.norm_squared()
        self.assertAlmostEqual(n2, C.join().norm_squared())
