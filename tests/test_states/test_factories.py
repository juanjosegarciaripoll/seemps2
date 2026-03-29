import numpy as np
from math import sqrt
from seemps.state.factories import graph_state, GHZ, product_state, W, AKLT, mps_ones
from ..tools import SeeMPSTestCase


class TestFactories(SeeMPSTestCase):
    def test_product_state(self):
        a = np.array([1.0, 7.0])
        b = np.array([0.0, 1.0, 3.0])
        c = np.array([3.0, 5.0])

        state1 = product_state(a, length=3)
        tensor1 = np.reshape(a, (1, 2, 1))
        self.assertEqual(state1.size, 3)
        self.assertEqual(state1.dimension(), 8)
        self.assertTrue(np.array_equal(state1[0], tensor1))
        self.assertTrue(np.array_equal(state1[1], tensor1))
        self.assertTrue(np.array_equal(state1[2], tensor1))
        self.assertTrue(np.array_equal(state1.to_vector(), np.kron(a, np.kron(a, a))))

        state2 = product_state([a, b, c])
        tensor2a = np.reshape(a, (1, 2, 1))
        tensor2b = np.reshape(b, (1, 3, 1))
        tensor2c = np.reshape(c, (1, 2, 1))
        self.assertEqual(state2.size, 3)
        self.assertEqual(state2.dimension(), 2 * 3 * 2)
        self.assertTrue(np.array_equal(state2[0], tensor2a))
        self.assertTrue(np.array_equal(state2[1], tensor2b))
        self.assertTrue(np.array_equal(state2[2], tensor2c))
        self.assertTrue(np.array_equal(state2.to_vector(), np.kron(a, np.kron(b, c))))

    def test_GHZ(self):
        ghz1 = np.array([1.0, 1.0]) / sqrt(2.0)
        ghz2 = np.array([1.0, 0.0, 0.0, 1.0]) / sqrt(2.0)
        ghz3 = np.array([1.0, 0, 0, 0, 0, 0, 0, 1.0]) / sqrt(2.0)
        self.assertTrue(np.array_equal(GHZ(1).to_vector(), ghz1))
        self.assertTrue(np.array_equal(GHZ(2).to_vector(), ghz2))
        self.assertTrue(np.array_equal(GHZ(3).to_vector(), ghz3))

        for i in range(1, 2):
            state = GHZ(i)
            self.assertEqual(state.size, i)
            self.assertEqual(state.dimension(), 2**i)

    def test_W(self):
        W1 = np.array([0, 1.0])
        W2 = np.array([0, 1, 1, 0]) / sqrt(2.0)
        W3 = np.array([0, 1, 1, 0, 1, 0, 0, 0]) / sqrt(3.0)
        self.assertTrue(np.array_equal(W(1).to_vector(), W1))
        self.assertTrue(np.array_equal(W(2).to_vector(), W2))
        self.assertTrue(np.array_equal(W(3).to_vector(), W3))

        for i in range(1, 2):
            state = W(i)
            self.assertEqual(state.size, i)
            self.assertEqual(state.dimension(), 2**i)

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
            state = AKLT(i)
            self.assertEqual(state.size, i)
            self.assertEqual(state.dimension(), 3**i)

    def test_graph_state(self):
        graph2 = np.ones(2**2) / sqrt(2**2)
        graph2[-1] = -graph2[-1]
        self.assertTrue(np.array_equal(graph_state(2).to_vector(), graph2))

        graph3 = np.ones(2**3) / sqrt(2**3)
        graph3[3] = -graph3[3]
        graph3[-2] = -graph3[-2]
        self.assertTrue(np.array_equal(graph_state(3).to_vector(), graph3))

        for i in range(2, 4):
            state = graph_state(i)
            self.assertEqual(state.size, i)
            self.assertEqual(state.dimension(), 2**i)

    def test_mps_ones(self):
        A23 = mps_ones([2, 3, 4])
        self.assertEqual(A23.physical_dimensions(), [2, 3, 4])
        self.assertEqual(A23.bond_dimensions(), [1, 1, 1, 1])
        self.assertSimilar(A23.to_vector(), np.ones(2 * 3 * 4))

        A23_norm = mps_ones([2, 3, 4], normalize=True)
        self.assertEqual(A23_norm.physical_dimensions(), [2, 3, 4])
        self.assertEqual(A23_norm.bond_dimensions(), [1, 1, 1, 1])
        self.assertSimilar(
            A23_norm.to_vector(), np.ones(2 * 3 * 4) / np.sqrt(2 * 3 * 4)
        )
