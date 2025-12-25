from seemps.operators import MPO, MPOList, MPOSum
from seemps.state import (
    MPSSum,
    random_uniform_mps,
    DEFAULT_STRATEGY,
    NO_TRUNCATION,
    Simplification,
    Strategy,
    simplify,
)
from seemps.tools import σx, σy, σz

from ..tools import TestCase, contain_same_objects

TEST_STRATEGY = DEFAULT_STRATEGY.replace(simplify=Simplification.VARIATIONAL)


class TestMPOSum(TestCase):
    def setUp(self):
        self.mpoA = MPO([σx.reshape(1, 2, 2, 1)] * 10)
        self.mpoB = MPO([σz.reshape(1, 2, 2, 1)] * 10)
        self.mpoC = MPOList([self.mpoA, self.mpoB])
        self.mpoD = MPO([σy.reshape(1, 2, 2, 1)] * 10)

    def assertIdenticalLists(self, a: list, b: list):
        if not all(A is B for A, B in zip(a, b)):
            raise AssertionError("Different lists:\na = {a}\nb = {b}")

    def test_mposum_init_copies_data(self):
        mpos = [self.mpoA, self.mpoB]
        weights = [1.0, 2.0]
        mposum = MPOSum(mpos, weights)
        self.assertTrue(mposum.mpos is not mpos)
        self.assertTrue(mposum.weights is not weights)

    def test_mposum_copy_is_shallow(self):
        A = MPOSum([self.mpoA, self.mpoB], [1.0, 2.0], Strategy())
        B = A.copy()
        self.assertTrue(A.mpos is not B.mpos)
        self.assertTrue(contain_same_objects(A.mpos, B.mpos))
        self.assertTrue(A.weights is not B.weights)
        self.assertTrue(contain_same_objects(A.weights, B.weights))
        self.assertTrue(A.strategy is B.strategy)

    def test_mposum_simple(self):
        mposum = MPOSum([self.mpoA, self.mpoB])
        self.assertIdenticalLists(mposum.mpos, [self.mpoA, self.mpoB, self.mpoC])
        self.assertSimilar(mposum.weights, [1.0, 1.0])
        self.assertEqual(mposum.size, self.mpoA.size)

    def test_mposum_arises_from_summing_mpos(self):
        mposum = self.mpoA + self.mpoB
        self.assertIsInstance(mposum, MPOSum)
        self.assertIdenticalLists(mposum.mpos, [self.mpoA, self.mpoB])
        self.assertSimilar(mposum.weights, [1.0, 1.0])
        self.assertEqual(mposum.size, self.mpoA.size)

        mposum = self.mpoA + self.mpoC
        self.assertIsInstance(mposum, MPOSum)
        self.assertIdenticalLists(mposum.mpos, [self.mpoA, self.mpoC])
        self.assertSimilar(mposum.weights, [1.0, 1.0])
        self.assertEqual(mposum.size, self.mpoA.size)

        mposum = self.mpoC + self.mpoA
        self.assertIsInstance(mposum, MPOSum)
        self.assertIdenticalLists(mposum.mpos, [self.mpoC, self.mpoA])
        self.assertSimilar(mposum.weights, [1.0, 1.0])
        self.assertEqual(mposum.size, self.mpoA.size)

    def test_mposum_arises_from_subtracting_mpos(self):
        mposum = self.mpoA - self.mpoB
        self.assertIsInstance(mposum, MPOSum)
        self.assertIdenticalLists(mposum.mpos, [self.mpoA, self.mpoB])
        self.assertSimilar(mposum.weights, [1.0, -1.0])

        mposum = self.mpoA - self.mpoC
        self.assertIsInstance(mposum, MPOSum)
        self.assertIdenticalLists(mposum.mpos, [self.mpoA, self.mpoC])
        self.assertSimilar(mposum.weights, [1.0, -1.0])

        mposum = self.mpoC - self.mpoB
        self.assertIsInstance(mposum, MPOSum)
        self.assertIdenticalLists(mposum.mpos, [self.mpoC, self.mpoB])
        self.assertSimilar(mposum.weights, [1.0, -1.0])

    def test_mposum_application_creates_mpssum(self):
        state = random_uniform_mps(2, self.mpoA.size, D=10)

        mposum = self.mpoA + self.mpoB
        newstate = mposum.apply(state, strategy=NO_TRUNCATION)
        self.assertIsInstance(newstate, MPSSum)
        self.assertSimilar(
            (self.mpoA + self.mpoB).apply(state).to_vector(),
            self.mpoA.apply(state).to_vector() + self.mpoB.apply(state).to_vector(),
        )

    def test_mposum_apply_can_simplify(self):
        state = random_uniform_mps(2, self.mpoA.size, D=10)
        mposum = self.mpoA + self.mpoB
        self.assertSimilar(
            mposum.apply(state, strategy=TEST_STRATEGY).to_vector(),
            mposum.to_matrix() @ state.to_vector(),
        )

    def test_mpo_set_strategy(self):
        new_strategy = Strategy(tolerance=1e-10)
        mposum = (self.mpoA + self.mpoB).set_strategy(new_strategy)
        self.assertTrue(new_strategy, mposum.strategy)

    def test_mposum_application_works_on_mpssum(self):
        mposum = self.mpoA + self.mpoB
        state = random_uniform_mps(2, self.mpoA.size, D=10)
        combined_state = simplify(
            self.mpoA.apply(2 * state) + self.mpoB.apply(2 * state),
            strategy=DEFAULT_STRATEGY.replace(simplify=Simplification.VARIATIONAL),
        )
        self.assertSimilar(mposum.apply(state + state), combined_state)
        self.assertSimilar(mposum @ (state + state), combined_state)

    def test_mposum_join_real_mpos(self):
        state = random_uniform_mps(2, self.mpoA.size, D=10)
        mposum = self.mpoA + self.mpoB
        newstate = mposum @ state
        newstate_join = mposum.join() @ state
        self.assertSimilar(newstate.to_vector(), newstate_join.to_vector())

    def test_mposum_join_complex_mpos(self):
        state = random_uniform_mps(2, self.mpoA.size, D=10)
        mposum = self.mpoA + self.mpoD
        newstate = mposum @ state
        newstate_join = mposum.join() @ state
        self.assertSimilar(newstate.to_vector(), newstate_join.to_vector())

    def test_mposum_T_is_transpose(self):
        mposum = MPOSum([self.mpoA, self.mpoB], [1, 1j])
        self.assertSimilar(
            mposum.T.to_matrix(), self.mpoA.to_matrix().T + 1j * self.mpoB.to_matrix().T
        )
