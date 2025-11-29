import numpy as np
from seemps.tools import σx
from seemps.state import (
    random_uniform_mps,
    MPSSum,
    DEFAULT_STRATEGY,
    Simplification,
    Strategy,
)
from seemps.operators import MPO, MPOList
from ..tools import TestCase, contain_different_objects, contain_same_objects

TEST_STRATEGY = DEFAULT_STRATEGY.replace(simplify=Simplification.VARIATIONAL)


class TestMPO(TestCase):
    def test_initial_data_is_copied(self):
        data = [np.zeros((1, 2, 2, 1))] * 10
        A = MPO(data)
        self.assertFalse(A._data is data)
        self.assertEqual(A._data, data)

    def test_copy_is_shallow(self):
        A = MPO([np.zeros((1, 2, 2, 1))] * 10, Strategy())
        B = A.copy()
        self.assertTrue(A._data is not B._data)
        self.assertTrue(contain_same_objects(A, B))
        self.assertTrue(A.strategy is B.strategy)
        A[::] = [np.ones((1, 2, 2, 1))] * 10
        self.assertTrue(contain_different_objects(A, B))

    def test_mpo_multiplies_by_number(self):
        mpo = MPO([σx.reshape(1, 2, 2, 1)] * 5)
        self.assertSimilar(mpo.to_matrix() * (-3.0), (mpo * (-3)).to_matrix())
        self.assertSimilar((-3.0) * mpo.to_matrix(), ((-3) * mpo).to_matrix())

    def test_mpo_rejects_multiplication_by_non_numbers(self):
        mpo = MPO([σx.reshape(1, 2, 2, 1)] * 5)
        state = random_uniform_mps(2, 3, rng=self.rng)
        with self.assertRaises(Exception):
            mpo * state  # type: ignore
        with self.assertRaises(Exception):
            [1.0] * mpo  # type: ignore

    def test_mpo_apply_is_matrix_multiplication(self):
        mpo = MPO([σx.reshape(1, 2, 2, 1)] * 5)
        mps = random_uniform_mps(2, mpo.size, D=2)
        self.assertSimilar((mpo @ mps).to_vector(), (mpo.to_matrix() @ mps.to_vector()))
        self.assertSimilar(
            mpo.apply(mps).to_vector(), (mpo.to_matrix() @ mps.to_vector())
        )

    def test_mpo_apply_can_simplify(self):
        mpo = MPO([σx.reshape(1, 2, 2, 1)] * 5)
        mps = random_uniform_mps(2, mpo.size, D=2)
        self.assertSimilar(
            mpo.apply(mps, simplify=True, strategy=TEST_STRATEGY).to_vector(),
            (mpo.to_matrix() @ mps.to_vector()),
        )

    def test_mpo_set_strategy(self):
        new_strategy = Strategy(tolerance=1e-10)
        mpo = MPO([σx.reshape(1, 2, 2, 1)] * 5).set_strategy(new_strategy)
        self.assertTrue(new_strategy, mpo.strategy)

    def test_mpo_apply_works_on_mpssum(self):
        mpo = MPO([σx.reshape(1, 2, 2, 1)] * 5)
        mps = random_uniform_mps(2, mpo.size, D=2)
        self.assertIsInstance(mpo.apply(mps + mps, simplify=False), MPSSum)
        self.assertSimilar(
            mpo.apply(mps + mps, simplify=True, strategy=TEST_STRATEGY).to_vector(),
            2 * (mpo.to_matrix() @ mps.to_vector()),
        )

    def test_mpo_apply_rejects_non_mps(self):
        mpo = MPO([σx.reshape(1, 2, 2, 1)] * 5)
        with self.assertRaises(TypeError):
            mpo.apply([np.zeros((1, 2, 1))] * 3)  # type: ignore
        with self.assertRaises(Exception):
            mpo @ [np.zeros((1, 2, 1))]  # type: ignore

    def test_mpo_extend(self):
        mpo = random_uniform_mps(2, 5, D=5, truncate=False)
        new_mpo = mpo.extend(7, sites=[0, 2, 4, 5, 6], dimensions=3)
        self.assertTrue(mpo[0] is new_mpo[0])
        self.assertEqual(new_mpo[1].shape, (5, 3, 5))
        self.assertTrue(mpo[1] is new_mpo[2])
        self.assertEqual(new_mpo[3].shape, (5, 3, 5))
        self.assertTrue(mpo[2] is new_mpo[4])
        self.assertTrue(mpo[3] is new_mpo[5])
        self.assertTrue(mpo[4] is new_mpo[6])

    def test_mpo_extend_accepts_dimensions_list_with_proper_size(self):
        mpo = MPO([σx.reshape(1, 2, 2, 1)] * 5)
        new_mpo = mpo.extend(7, sites=[0, 2, 4, 5, 6], dimensions=[5, 4])
        self.assertEqual(new_mpo.dimensions(), [2, 5, 2, 4, 2, 2, 2])
        with self.assertRaises(Exception):
            mpo.extend(7, sites=[0, 2, 4, 5, 6], dimensions=[5])
        with self.assertRaises(Exception):
            mpo.extend(7, sites=[0, 2, 4, 5, 6], dimensions=[5, 6, 8])

    def test_mpo_extend_cannot_shrink_mpo(self):
        mpo = MPO([σx.reshape(1, 2, 2, 1)] * 5)
        with self.assertRaises(Exception):
            mpo.extend(3)

    def test_mpo_pow_returns_mpolist(self):
        mpo = MPO([σx.reshape(1, 2, 2, 1)] * 5)
        mps = random_uniform_mps(2, mpo.size, D=2)
        self.assertIsInstance(mpo**2, MPOList)
        self.assertSimilar((mpo**2) @ mps, MPOList([mpo, mpo]) @ mps)

    def test_mpo_T_returns_transpose(self):
        mpo = MPO([self.rng.normal(size=(1, 2, 3, 1)) for _ in range(5)])
        mpoT = mpo.T
        self.assertSimilar(mpo.to_matrix().T, mpoT.to_matrix())
