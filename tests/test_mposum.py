import numpy as np
from seemps import MPO, MPOList, MPSSum, random_uniform_mps
from seemps.operators import MPOSum
from seemps.state import MPSSum
from seemps.state.core import DEFAULT_STRATEGY, Simplification, Strategy
from seemps.tools import σx, σy, σz

from .tools import TestCase, similar

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
        newstate = mposum.apply(state)
        self.assertIsInstance(newstate, MPSSum)
        self.assertSimilar(
            (self.mpoA + self.mpoB).apply(state).to_vector(),
            self.mpoA.apply(state).to_vector() + self.mpoB.apply(state).to_vector(),
        )

    def test_mposum_matmul_creates_mpssum(self):
        state = random_uniform_mps(2, self.mpoA.size, D=10)

        mposum = self.mpoA + self.mpoB
        newstate = mposum @ state
        self.assertIsInstance(newstate, MPSSum)
        self.assertSimilar(
            ((self.mpoA + self.mpoB) @ state).to_vector(),
            self.mpoA.apply(state).to_vector() + self.mpoB.apply(state).to_vector(),
        )

    def test_mposum_apply_can_simplify(self):
        state = random_uniform_mps(2, self.mpoA.size, D=10)

        mposum = self.mpoA + self.mpoB
        newstate = mposum.apply(state)
        self.assertIsInstance(newstate, MPSSum)
        self.assertSimilar(
            (self.mpoA + self.mpoB)
            .apply(state, simplify=True, strategy=TEST_STRATEGY)
            .to_vector(),
            (self.mpoA + self.mpoB).tomatrix() @ state.to_vector(),
        )

    def test_mpo_set_strategy(self):
        new_strategy = Strategy(tolerance=1e-10)
        mposum = (self.mpoA + self.mpoB).set_strategy(new_strategy)
        self.assertTrue(new_strategy, mposum.strategy)

    def test_mposum_application_works_on_mpssum(self):
        mposum = self.mpoA + self.mpoB
        state = random_uniform_mps(2, 3, rng=self.rng)
        self.assertSimilar(
            mposum.apply(state + state), self.mpoA.apply(self.mpoB.apply(2.0 * state))
        )
        self.assertSimilar(
            mposum @ (state + state), self.mpoA.apply(self.mpoB.apply(2.0 * state))
        )

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
