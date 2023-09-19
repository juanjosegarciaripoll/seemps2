import numpy as np
from .tools import *
from seemps import MPO, σx, random_uniform_mps


class TestMPO(TestCase):
    def test_mpo_multiplies_by_number(self):
        mpo = MPO([σx.reshape(1, 2, 2, 1)] * 5)
        self.assertSimilar(mpo.tomatrix() * (-3.0), (mpo * (-3)).tomatrix())
        self.assertSimilar((-3.0) * mpo.tomatrix(), ((-3) * mpo).tomatrix())

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

    def test_mpo_rejects_non_mps(self):
        mpo = MPO([σx.reshape(1, 2, 2, 1)] * 5)
        with self.assertRaises(Exception):
            mpo.apply([np.zeros(1, 2, 1)] * 3)
        with self.assertRaises(Exception):
            mpo @ [np.zeros(1, 2, 1)]
