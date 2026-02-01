from __future__ import annotations
import numpy as np
from seemps.tools import σx, mkron
from seemps.hamiltonians import InteractionGraph
from ..tools import SeeMPSTestCase


class TestInteractionGraph(SeeMPSTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.sx = σx
        self.id2 = np.eye(2)
        self.SX = np.asarray([[0, 1, 0], [1, 0, 1], [0, 1, 0]])

    def test_ig_requires_integer_dimensions(self):
        with self.assertRaises(AssertionError):
            InteractionGraph([3.0, 2])  # type: ignore
        with self.assertRaises(AssertionError):
            InteractionGraph([3.0, 2, 3 / 2])  # type: ignore

    def test_ig_requires_dimensions_above_one(self):
        with self.assertRaises(AssertionError):
            InteractionGraph([1, 2, 3])
        with self.assertRaises(AssertionError):
            InteractionGraph([0, 2, 3])
        with self.assertRaises(AssertionError):
            InteractionGraph([-1, 2, 3])

    def test_ig_dimension_is_adequately_computed(self):
        self.assertEqual(InteractionGraph([2, 3, 2]).dimension, 2 * 3 * 2)

    def test_one_local_term_at_0(self):
        ig0 = InteractionGraph([2, 2, 2])
        ig0.add_local_term(0, 0.5j * self.sx)
        self.assertEqual(ig0._interactions, ["baa"])

        H = mkron(0.5j * self.sx, self.id2, self.id2)
        self.assertSimilar(ig0.to_matrix(), H, atol=0.0)

        Hmpo = ig0.to_mpo()
        self.assertEqual(Hmpo.physical_dimensions(), [2, 2, 2])
        self.assertEqual(Hmpo.bond_dimensions(), [1, 1])

    def test_one_local_term_at_1(self):
        ig1 = InteractionGraph([2, 2, 2])
        ig1.add_local_term(1, self.sx)
        self.assertEqual(ig1._interactions, ["aba"])

        H = mkron(self.id2, self.sx, self.id2)
        self.assertSimilar(ig1.to_matrix(), H, atol=0.0)

        Hmpo = ig1.to_mpo()
        self.assertEqual(Hmpo.physical_dimensions(), [2, 2, 2])
        self.assertEqual(Hmpo.bond_dimensions(), [1, 1])
        self.assertSimilar(Hmpo.to_matrix(), H, atol=0.0)

    def test_one_local_term_at_2(self):
        ig2 = InteractionGraph([2, 2, 2])
        ig2.add_local_term(2, 0.5 * self.sx)
        self.assertEqual(ig2._interactions, ["aab"])

        H = mkron(self.id2, self.id2, 0.5 * self.sx)
        self.assertSimilar(ig2.to_matrix(), H, atol=0.0)

        Hmpo = ig2.to_mpo()
        self.assertEqual(Hmpo.physical_dimensions(), [2, 2, 2])
        self.assertEqual(Hmpo.bond_dimensions(), [1, 1])
        self.assertSimilar(Hmpo.to_matrix(), H, atol=0.0)

    def test_many_identical_local_terms(self):
        ig2 = InteractionGraph([2, 2, 2])
        ig2.add_identical_local_terms(0.5 * self.sx)
        H = (
            mkron(0.5 * self.sx, self.id2, self.id2)
            + mkron(self.id2, 0.5 * self.sx, self.id2)
            + mkron(self.id2, self.id2, 0.5 * self.sx)
        )
        self.assertSimilar(ig2.to_matrix(), H, atol=0.0)

    def test_one_interaction_01(self):
        ig01 = InteractionGraph([2, 2, 2])
        ig01.add_interaction_term(0, self.sx, 1, self.sx)
        self.assertEqual(ig01._interactions, ["bba"])

        H = mkron(self.sx, self.sx, self.id2)
        self.assertSimilar(ig01.to_matrix(), H, atol=0.0)

        Hmpo = ig01.to_mpo()
        self.assertEqual(Hmpo.physical_dimensions(), [2, 2, 2])
        self.assertEqual(Hmpo.bond_dimensions(), [1, 1])
        self.assertSimilar(Hmpo.to_matrix(), H, atol=0.0)

    def test_one_interaction_12(self):
        ig12 = InteractionGraph([2, 2, 2])
        ig12.add_interaction_term(1, self.sx, 2, self.sx)
        self.assertEqual(ig12._interactions, ["abb"])

        H = mkron(self.id2, self.sx, self.sx)
        self.assertSimilar(ig12.to_matrix(), H, atol=0.0)

        Hmpo = ig12.to_mpo()
        self.assertEqual(Hmpo.physical_dimensions(), [2, 2, 2])
        self.assertEqual(Hmpo.bond_dimensions(), [1, 1])
        self.assertSimilar(Hmpo.to_matrix(), H, atol=0.0)

    def test_interaction_term_sorts_input(self):
        ig21 = InteractionGraph([2, 2, 2])
        ig21.add_interaction_term(2, self.sx, 1, self.sx)
        self.assertEqual(ig21._interactions, ["abb"])

    def test_nearest_neighbor(self):
        ig = InteractionGraph([2, 2, 2])
        ig.add_nearest_neighbor_interaction(self.sx, self.sx)
        H = mkron(self.sx, self.sx, self.id2) + mkron(self.id2, self.sx, self.sx)
        self.assertSimilar(ig.to_matrix(), H, atol=0.0)

        Hmpo = ig.to_mpo(simplify=False)
        self.assertEqual(Hmpo.physical_dimensions(), [2, 2, 2])
        self.assertEqual(Hmpo.bond_dimensions(), [2, 2])
        self.assertSimilar(Hmpo.to_matrix(), H, atol=0)

    def test_large_nearest_neighbor(self):
        L = 11
        ig = InteractionGraph([2] * L)
        ig.add_nearest_neighbor_interaction(self.sx, self.sx)

        Hmpo = ig.to_mpo(simplify=True)
        self.assertEqual(Hmpo.physical_dimensions(), [2] * L)
        self.assertEqual(Hmpo.bond_dimensions(), ([2] + [3] * (L - 3) + [2]))
        self.assertSimilar(Hmpo.to_matrix(), ig.to_matrix(), atol=0)

    def test_long_range_Ising(self):
        J = self.rng.normal(size=(3, 3))
        ig = InteractionGraph([2, 2, 2])
        ig.add_long_range_interaction(J, self.sx)
        self.assertEqual(len(ig._interactions), (3 * 2) / 2)
        H = (
            (J[0, 1] + J[1, 0]) * mkron(self.sx, self.sx, self.id2)
            + (J[1, 2] + J[2, 1]) * mkron(self.id2, self.sx, self.sx)
            + (J[0, 2] + J[2, 0]) * mkron(self.sx, self.id2, self.sx)
        )
        self.assertSimilar(ig.to_matrix(), H, atol=0.0)

        Hmpo = ig.to_mpo(simplify=False)
        self.assertEqual(
            [A.shape for A in Hmpo], [(1, 2, 2, 3), (3, 2, 2, 3), (3, 2, 2, 1)]
        )
        self.assertSimilar(Hmpo.to_matrix(), H, atol=0)
        Hmpo = ig.to_mpo(simplify=True)
        self.assertEqual(
            [A.shape for A in Hmpo], [(1, 2, 2, 3), (3, 2, 2, 2), (2, 2, 2, 1)]
        )
        self.assertSimilar(Hmpo.to_matrix(), H, atol=1e-15)
