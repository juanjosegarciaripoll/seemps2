import numpy as np
from math import sqrt
from seemps.state.factories import graph_state, GHZ, product_state, W, AKLT
from ..tools import TestCase


# TODO: Is this file redundant or better than test_sampling.py
class TestSampleStates(TestCase):
    def test_product_state(self):
        a = np.array([1.0, 7.0])
        b = np.array([0.0, 1.0, 3.0])
        c = np.array([3.0, 5.0])

        # Test a product state MPS of size 3
        state1 = product_state(a, length=3)
        tensor1 = np.reshape(a, (1, 2, 1))
        #
        # Test whether the MPS has the right size and dimension
        self.assertEqual(state1.size, 3)
        self.assertEqual(state1.dimension(), 8)
        #
        # Verify that it has the same data as input
        self.assertTrue(np.array_equal(state1[0], tensor1))
        self.assertTrue(np.array_equal(state1[1], tensor1))
        self.assertTrue(np.array_equal(state1[2], tensor1))
        #
        # Verify that they produce the same wavefunction as directly
        # creating the vector
        state1ψ = np.kron(a, np.kron(a, a))
        self.assertTrue(np.array_equal(state1.to_vector(), state1ψ))

        # Test a product state with different physical dimensions on
        # even and odd sites.
        state2 = product_state([a, b, c])
        tensor2a = np.reshape(a, (1, 2, 1))
        tensor2b = np.reshape(b, (1, 3, 1))
        tensor2c = np.reshape(c, (1, 2, 1))
        #
        # Test whether the MPS has the right size and dimension
        self.assertEqual(state2.size, 3)
        self.assertEqual(state2.dimension(), 2 * 3 * 2)
        #
        # Verify that it has the same data as input
        self.assertTrue(np.array_equal(state2[0], tensor2a))
        self.assertTrue(np.array_equal(state2[1], tensor2b))
        self.assertTrue(np.array_equal(state2[2], tensor2c))
        #
        # Verify that they produce the same wavefunction as directly
        # creating the vector
        state2ψ = np.kron(a, np.kron(b, c))
        self.assertTrue(np.array_equal(state2.to_vector(), state2ψ))

    def test_GHZ(self):
        ghz1 = np.array([1.0, 1.0]) / sqrt(2.0)
        mps1 = GHZ(1)
        self.assertTrue(np.array_equal(mps1.to_vector(), ghz1))

        ghz2 = np.array([1.0, 0.0, 0.0, 1.0]) / sqrt(2.0)
        mps2 = GHZ(2)
        self.assertTrue(np.array_equal(mps2.to_vector(), ghz2))

        ghz3 = np.array([1.0, 0, 0, 0, 0, 0, 0, 1.0]) / sqrt(2.0)
        mps3 = GHZ(3)
        self.assertTrue(np.array_equal(mps3.to_vector(), ghz3))

        for i in range(1, 2):
            Ψ = GHZ(i)
            self.assertEqual(Ψ.size, i)
            self.assertEqual(Ψ.dimension(), 2**i)

    def test_W(self):
        W1 = np.array([0, 1.0])
        mps1 = W(1)
        self.assertTrue(np.array_equal(mps1.to_vector(), W1))

        W2 = np.array([0, 1, 1, 0]) / sqrt(2.0)
        mps2 = W(2)
        self.assertTrue(np.array_equal(mps2.to_vector(), W2))

        W3 = np.array([0, 1, 1, 0, 1, 0, 0, 0]) / sqrt(3.0)
        mps3 = W(3)
        self.assertTrue(np.array_equal(mps3.to_vector(), W3))

        for i in range(1, 2):
            Ψ = W(i)
            self.assertEqual(Ψ.size, i)
            self.assertEqual(Ψ.dimension(), 2**i)

    def test_AKLT(self):
        AKLT2 = np.zeros(3**2)
        AKLT2[1] = 1
        AKLT2[3] = -1
        AKLT2 = AKLT2 / sqrt(2)
        self.assertTrue(np.array_equal(AKLT(2).to_vector(), AKLT2))

        AKLT3 = np.zeros(3**3)
        AKLT3[4] = 1
        AKLT3[6] = -1
        AKLT3[10] = -1
        AKLT3[12] = 1
        AKLT3 = AKLT3 / (sqrt(2) ** 2)
        self.assertTrue(np.array_equal(AKLT(3).to_vector(), AKLT3))

        for i in range(2, 5):
            Ψ = AKLT(i)
            self.assertEqual(Ψ.size, i)
            self.assertEqual(Ψ.dimension(), 3**i)

    def test_graph(self):
        GR = np.ones(2**2) / sqrt(2**2)
        GR[-1] = -GR[-1]
        self.assertTrue(np.array_equal(graph_state(2).to_vector(), GR))

        GR = np.ones(2**3) / sqrt(2**3)
        GR[3] = -GR[3]
        GR[-2] = -GR[-2]
        self.assertTrue(np.array_equal(graph_state(3).to_vector(), GR))

        for i in range(1, 2):
            Ψ = W(i)
            self.assertEqual(Ψ.size, i)
            self.assertEqual(Ψ.dimension(), 2**i)
