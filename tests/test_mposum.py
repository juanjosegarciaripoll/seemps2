import numpy as np
from .tools import TestCase, similar
from seemps import MPSSum, MPO, MPOList, random_uniform_mps
from seemps.operators import MPOSum
from seemps.tools import σx, σz


class TestMPOSum(TestCase):
    def setUp(self):
        self.mpoA = MPO([σx.reshape(1, 2, 2, 1)] * 10)
        self.mpoB = MPO([σz.reshape(1, 2, 2, 1)] * 10)
        self.mpoC = MPOList([self.mpoA, self.mpoB])

    def assertIdenticalLists(self, a: list, b: list):
        if not all(A is B for A, B in zip(a, b)):
            raise AssertionError("Different lists:\na = {a}\nb = {b}")

    def test_mposum_simple(self):
        mposum = MPOSum([self.mpoA, self.mpoB])
        self.assertIdenticalLists(mposum.mpos, [self.mpoA, self.mpoB, self.mpoC])
        self.assertSimilar(mposum.weights, [1.0, 1.0])

    def test_mposum_arises_from_summing_mpos(self):
        mposum = self.mpoA + self.mpoB
        self.assertIsInstance(mposum, MPOSum)
        self.assertIdenticalLists(mposum.mpos, [self.mpoA, self.mpoB])
        self.assertSimilar(mposum.weights, [1.0, 1.0])

        mposum = self.mpoA + self.mpoC
        self.assertIsInstance(mposum, MPOSum)
        self.assertIdenticalLists(mposum.mpos, [self.mpoA, self.mpoC])
        self.assertSimilar(mposum.weights, [1.0, 1.0])

        mposum = self.mpoC + self.mpoA
        self.assertIsInstance(mposum, MPOSum)
        self.assertIdenticalLists(mposum.mpos, [self.mpoC, self.mpoA])
        self.assertSimilar(mposum.weights, [1.0, 1.0])

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
