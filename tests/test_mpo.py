import numpy as np
from .tools import *
from seemps import MPO, σx, random_uniform_mps


class TestMPO(TestCase):
    def test_mpo_multiplies_by_number(self):
        mpo = MPO([σx.reshape(1, 2, 2, 1)] * 5)
        self.assertSimilar(mpo.tomatrix() * (-3.0), (mpo * (-3)).tomatrix())
        self.assertSimilar((-3.0) * mpo.tomatrix(), ((-3) * mpo).tomatrix())

    def test_mpo_rejects_multiplication_by_non_numbers(self):
        mpo = MPO([σx.reshape(1, 2, 2, 1)] * 5)
        state = random_uniform_mps(2, 3, rng=self.rng)
        with self.assertRaises(Exception):
            mpo * state
        with self.assertRaises(Exception):
            [1.0] * mpo

    def test_mpo_apply_is_matrix_multiplication(self):
        mpo = MPO([σx.reshape(1, 2, 2, 1)] * 5)
        mps = random_uniform_mps(2, mpo.size, D=2)
        self.assertSimilar((mpo @ mps).to_vector(), (mpo.tomatrix() @ mps.to_vector()))
        self.assertSimilar(
            mpo.apply(mps).to_vector(), (mpo.tomatrix() @ mps.to_vector())
        )

    def test_mpo_apply_can_simplify(self):
        mpo = MPO([σx.reshape(1, 2, 2, 1)] * 5)
        mps = random_uniform_mps(2, mpo.size, D=2)
        self.assertSimilar(
            mpo.apply(mps, simplify=True).to_vector(),
            (mpo.tomatrix() @ mps.to_vector()),
        )

    def test_mpo_apply_works_on_mpssum(self):
        mpo = MPO([σx.reshape(1, 2, 2, 1)] * 5)
        mps = random_uniform_mps(2, mpo.size, D=2)
        self.assertSimilar(
            mpo.apply(mps + mps, simplify=True).to_vector(),
            2 * (mpo.tomatrix() @ mps.to_vector()),
        )

    def test_mpo_apply_rejects_non_mps(self):
        mpo = MPO([σx.reshape(1, 2, 2, 1)] * 5)
        with self.assertRaises(TypeError):
            mpo.apply([np.zeros((1, 2, 1))] * 3)
        with self.assertRaises(Exception):
            mpo @ [np.zeros((1, 2, 1))]

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
