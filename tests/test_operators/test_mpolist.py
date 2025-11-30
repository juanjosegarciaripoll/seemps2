import numpy as np
from seemps.tools import σx, σy, σz
from seemps.state import (
    random_uniform_mps,
    DEFAULT_STRATEGY,
    NO_TRUNCATION,
    Simplification,
    Strategy,
)
from seemps.operators import MPO, MPOList

from ..tools import TestCase, contain_same_objects

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
            UV.apply(mps, simplify=True, strategy=TEST_STRATEGY).to_vector(),
            (UV.to_matrix() @ mps.to_vector()),
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
            mpo.apply([np.zeros((1, 2, 1))] * 3)  # type: ignore
        with self.assertRaises(Exception):
            mpo @ [np.zeros((1, 2, 1))]  # type: ignore

    def test_mpolist_matrix_is_product_of_mpo_matrices(self):
        U = MPO([σx.reshape(1, 2, 2, 1)] * 3)
        V = MPO([σz.reshape(1, 2, 2, 1)] * 3)
        UV = MPOList([U, V], NO_TRUNCATION)
        self.assertSimilar(UV.to_matrix(), V.to_matrix() @ U.to_matrix())
        state = random_uniform_mps(2, 3, rng=self.rng)
        self.assertSimilar(
            UV.apply(state).to_vector(),
            V.to_matrix() @ U.to_matrix() @ state.to_vector(),
        )

    def test_mpolist_can_be_rescaled(self):
        U = MPO([σx.reshape(1, 2, 2, 1)] * 3)
        V = MPO([σz.reshape(1, 2, 2, 1)] * 3)
        UV = MPOList([U, V], NO_TRUNCATION)
        self.assertSimilar(UV.to_matrix() * (-3), (UV * (-3)).to_matrix())
        self.assertSimilar(UV.to_matrix() * (-3), ((-3) * UV).to_matrix())

    def test_mpolist_raises_error_when_rescaling_by_non_number(self):
        U = MPO([σx.reshape(1, 2, 2, 1)] * 3)
        UV = MPOList([U, U], NO_TRUNCATION)
        with self.assertRaises(Exception):
            UV * U  # type: ignore
        with self.assertRaises(Exception):
            U * UV  # type: ignore

    def test_mpolist_extends_does_so_on_each_operator(self):
        U = MPO([σx.reshape(1, 2, 2, 1)] * 3)
        V = MPO([σz.reshape(1, 2, 2, 1)] * 3)
        UV = MPOList([U, V], NO_TRUNCATION)

        ex_U = U.extend(4, dimensions=[3])
        ex_V = V.extend(4, dimensions=[3])
        ex_UV = UV.extend(4, dimensions=[3])
        self.assertSimilar(ex_UV.to_matrix(), ex_V.to_matrix() @ ex_U.to_matrix())

    def test_mpolist_join_real_mpos(self):
        U = MPO([σx.reshape(1, 2, 2, 1)] * 3)
        V = MPO([σz.reshape(1, 2, 2, 1)] * 3)
        UV = MPOList([U, V], NO_TRUNCATION)
        UV_join = UV.join()
        self.assertSimilar(UV.to_matrix(), UV_join.to_matrix())

    def test_mpolist_join_complex_mpos(self):
        U = MPO([σy.reshape(1, 2, 2, 1)] * 3)
        V = MPO([σz.reshape(1, 2, 2, 1)] * 3)
        UV = MPOList([U, V], NO_TRUNCATION)
        UV_join = UV.join()
        self.assertSimilar(UV.to_matrix(), UV_join.to_matrix())

    def test_mpolist_T_returns_transpose(self):
        U = MPO([σy.reshape(1, 2, 2, 1)] * 3)
        V = MPO([σz.reshape(1, 2, 2, 1)] * 3)
        UV = MPOList([U, V], NO_TRUNCATION)
        UVT = UV.T
        self.assertSimilar(UV.to_matrix().T, UVT.to_matrix())

    def test_mpolist_dimensions_returns_those_of_first_mpo(self):
        U = MPO([self.rng.random(size=(1, 3, 2, 1))] * 3)
        V = MPO([self.rng.random(size=(1, 4, 3, 1))] * 3)
        UV = MPOList([U, V], NO_TRUNCATION)
        self.assertEqual(UV.dimensions(), U.dimensions())
