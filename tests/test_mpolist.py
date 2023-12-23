import numpy as np
from seemps import MPO, NO_TRUNCATION, MPOList, random_uniform_mps, σx, σy, σz
from seemps.state import MPSSum
from seemps.state.core import DEFAULT_STRATEGY, Simplification, Strategy

from .tools import TestCase, contain_same_objects

TEST_STRATEGY = DEFAULT_STRATEGY.replace(simplify=Simplification.VARIATIONAL)


class TestMPOList(TestCase):
    def test_mpolist_construction(self):
        U = MPO([σx.reshape(1, 2, 2, 1)] * 3)
        V = MPO([σz.reshape(1, 2, 2, 1)] * 3)
        mpos = [U, V]
        UV = MPOList(mpos, NO_TRUNCATION)
        self.assertIsInstance(UV, MPOList)
        self.assertTrue(UV.mpos is not mpos)
        self.assertTrue(contain_same_objects(UV.mpos, mpos))
        self.assertEqual(UV.strategy, NO_TRUNCATION)

    def test_mpolist_copy_is_shallow(self):
        U = MPO([σx.reshape(1, 2, 2, 1)] * 3)
        V = MPO([σz.reshape(1, 2, 2, 1)] * 3)
        mpos = [U, V]
        UV = MPOList(mpos, Strategy())
        UV_copy = UV.copy()
        self.assertIsInstance(UV_copy, MPOList)
        self.assertTrue(UV.mpos is not UV_copy.mpos)
        self.assertTrue(contain_same_objects(UV.mpos, UV_copy.mpos))
        self.assertEqual(UV.mpos, UV_copy.mpos)
        self.assertEqual(UV.size, UV_copy.size)
        self.assertTrue(UV.strategy is UV_copy.strategy)

    def test_mpolist_application_without_errors(self):
        U = MPO([σx.reshape(1, 2, 2, 1)] * 3)
        V = MPO([σz.reshape(1, 2, 2, 1)] * 3)
        UV = MPOList([U, V], NO_TRUNCATION)
        state = random_uniform_mps(2, 3, rng=self.rng)
        self.assertSimilar(UV.apply(state), V.apply(U.apply(state)))
        self.assertSimilar(UV @ state, V.apply(U.apply(state)))
        self.assertEqual(UV.size, U.size)

    def test_mpo_apply_can_simplify(self):
        U = MPO([σx.reshape(1, 2, 2, 1)] * 3)
        V = MPO([σz.reshape(1, 2, 2, 1)] * 3)
        UV = MPOList([U, V], NO_TRUNCATION)
        mps = random_uniform_mps(2, 3, D=2)
        self.assertSimilar(
            UV.apply(mps, strategy=TEST_STRATEGY).to_vector(),
            (UV.tomatrix() @ mps.to_vector()),
        )

    def test_mpo_set_strategy(self):
        new_strategy = Strategy(tolerance=1e-10)
        U = MPO([σx.reshape(1, 2, 2, 1)] * 3)
        V = MPO([σz.reshape(1, 2, 2, 1)] * 3)
        UV = MPOList([U, V], NO_TRUNCATION).set_strategy(new_strategy)
        self.assertTrue(new_strategy, UV.strategy)

    def test_mpolist_application_works_on_mpssum(self):
        U = MPO([σx.reshape(1, 2, 2, 1)] * 3)
        V = MPO([σz.reshape(1, 2, 2, 1)] * 3)
        UV = MPOList([U, V], NO_TRUNCATION)
        state = random_uniform_mps(2, 3, rng=self.rng)
        self.assertSimilar(UV.apply(state + state), V.apply(U.apply(2.0 * state)))
        self.assertSimilar(UV @ (state + state), V.apply(U.apply(2.0 * state)))

    def test_mpo_apply_rejects_non_mps(self):
        U = MPO([σx.reshape(1, 2, 2, 1)] * 3)
        V = MPO([σz.reshape(1, 2, 2, 1)] * 3)
        mpo = MPOList([U, V], NO_TRUNCATION)
        with self.assertRaises(TypeError):
            mpo.apply([np.zeros((1, 2, 1))] * 3)
        with self.assertRaises(Exception):
            mpo @ [np.zeros((1, 2, 1))]

    def test_mpolist_matrix_is_product_of_mpo_matrices(self):
        U = MPO([σx.reshape(1, 2, 2, 1)] * 3)
        V = MPO([σz.reshape(1, 2, 2, 1)] * 3)
        UV = MPOList([U, V], NO_TRUNCATION)
        self.assertSimilar(UV.tomatrix(), V.tomatrix() @ U.tomatrix())
        state = random_uniform_mps(2, 3, rng=self.rng)
        self.assertSimilar(
            UV.apply(state).to_vector(), V.tomatrix() @ U.tomatrix() @ state.to_vector()
        )

    def test_mpolist_can_be_rescaled(self):
        U = MPO([σx.reshape(1, 2, 2, 1)] * 3)
        V = MPO([σz.reshape(1, 2, 2, 1)] * 3)
        UV = MPOList([U, V], NO_TRUNCATION)
        self.assertSimilar(UV.tomatrix() * (-3), (UV * (-3)).tomatrix())
        self.assertSimilar(UV.tomatrix() * (-3), ((-3) * UV).tomatrix())

    def test_mpolist_raises_error_when_rescaling_by_non_number(self):
        U = MPO([σx.reshape(1, 2, 2, 1)] * 3)
        UV = MPOList([U, U], NO_TRUNCATION)
        with self.assertRaises(Exception):
            UV * U
        with self.assertRaises(Exception):
            U * UV

    def test_mpolist_extends_does_so_on_each_operator(self):
        U = MPO([σx.reshape(1, 2, 2, 1)] * 3)
        V = MPO([σz.reshape(1, 2, 2, 1)] * 3)
        UV = MPOList([U, V], NO_TRUNCATION)

        ex_U = U.extend(4, dimensions=[3])
        ex_V = V.extend(4, dimensions=[3])
        ex_UV = UV.extend(4, dimensions=[3])
        self.assertSimilar(ex_UV.tomatrix(), ex_V.tomatrix() @ ex_U.tomatrix())

    def test_mpolist_join_real_mpos(self):
        U = MPO([σx.reshape(1, 2, 2, 1)] * 3)
        V = MPO([σz.reshape(1, 2, 2, 1)] * 3)
        UV = MPOList([U, V], NO_TRUNCATION)
        UV_join = UV.join()
        self.assertSimilar(UV.tomatrix(), UV_join.tomatrix())

    def test_mpolist_join_complex_mpos(self):
        U = MPO([σy.reshape(1, 2, 2, 1)] * 3)
        V = MPO([σz.reshape(1, 2, 2, 1)] * 3)
        UV = MPOList([U, V], NO_TRUNCATION)
        UV_join = UV.join()
        self.assertSimilar(UV.tomatrix(), UV_join.tomatrix())
