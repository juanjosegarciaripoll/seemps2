from __future__ import annotations
import numpy as np
from seemps.tools import σx, σy, σz
from seemps.hamiltonians import (
    NNHamiltonian,
    ConstantNNHamiltonian,
    HeisenbergHamiltonian,
)
from ..tools import TestCase
from math import sqrt

i2 = np.eye(2)
i3 = np.eye(3)
Sx = np.array([[0, 1 / sqrt(2), 0], [1 / sqrt(2), 0, 1 / sqrt(2)], [0, 1 / sqrt(2), 0]])
Sz = np.array([[1, 0, 0], [0, 0, 0], [0, 0, -1]])
Sy = -0.5j * (Sz @ Sx - Sx @ Sz)


class TestConstantNNHamiltonian(TestCase):
    def assertSimilarMatrix(self, H: NNHamiltonian, M: np.ndarray) -> None:
        self.assertSimilar(H.to_matrix(), M)
        self.assertSimilar(H.to_mpo().to_matrix(), M)

    def test_Hamiltonian_size_and_dimension_list_match(self):
        with self.assertRaises(Exception):
            H2 = ConstantNNHamiltonian(3, [2] * 5)  # noqa: F841 # type: ignore

    def test_adding_local_term_to_sides_adds_them_to_interaction(self):
        H2 = ConstantNNHamiltonian(2, 2)
        H2.add_local_term(0, σx)
        self.assertSimilar(H2.interaction_term(0), np.kron(σx, i2))
        self.assertSimilarMatrix(H2, np.kron(σx, i2))

        H2 = ConstantNNHamiltonian(2, 2)
        H2.add_local_term(1, σy)
        self.assertSimilar(H2.interaction_term(0), np.kron(i2, σy))
        self.assertSimilarMatrix(H2, np.kron(i2, σy))

    def test_adding_local_term_inside_splits_in_two_interactions(self):
        H2 = ConstantNNHamiltonian(3, 2)
        H2.add_local_term(1, σx)
        self.assertSimilar(H2.interaction_term(0), np.kron(i2, 0.5 * σx))
        self.assertSimilar(H2.interaction_term(1), np.kron(0.5 * σx, i2))
        self.assertSimilarMatrix(H2, np.kron(i2, np.kron(σx, i2)))

    def test_adding_product_interaction_kronecker_product_matrix(self):
        H2 = ConstantNNHamiltonian(2, 2)
        H2.add_interaction_term(0, σx, σy)
        self.assertSimilar(H2.interaction_term(0), np.kron(σx, σy))
        self.assertSimilarMatrix(H2, np.kron(σx, σy))

    def test_add_interaction_with_one_term_stores_matrix(self):
        H2 = ConstantNNHamiltonian(2, 2)
        H2.add_interaction_term(0, np.kron(σx, σy))
        self.assertSimilar(H2.interaction_term(0), np.kron(σx, σy))
        self.assertSimilarMatrix(H2, np.kron(σx, σy))

    def test_local_non_uniform_Hamiltonian(self):
        H2 = ConstantNNHamiltonian(2, 2)
        H2.add_local_term(0, -0.5 * σx)
        H2.add_local_term(1, +1.2 * σy)
        self.assertSimilarMatrix(H2, np.kron(-0.5 * σx, i2) + np.kron(i2, 1.2 * σy))

        H3 = ConstantNNHamiltonian(3, 2)
        H3.add_local_term(0, -0.5 * σx)
        H3.add_local_term(1, +1.2 * σy)
        H3.add_local_term(2, +0.8 * σz)
        self.assertSimilarMatrix(
            H3,
            np.kron(-0.5 * σx, np.kron(i2, i2))
            + np.kron(i2, np.kron(1.2 * σy, i2))
            + np.kron(i2, np.kron(i2, 0.8 * σz)),
        )

    def test_adding_local_term_with_different_dimensions(self):
        H2 = ConstantNNHamiltonian(2, [3, 2])
        H2.add_local_term(0, Sx)
        self.assertSimilar(H2.interaction_term(0), np.kron(Sx, i2))
        self.assertSimilarMatrix(H2, np.kron(Sx, i2))

        H2 = ConstantNNHamiltonian(2, [2, 3])
        H2.add_local_term(1, Sy)
        self.assertSimilar(H2.interaction_term(0), np.kron(i2, Sy))
        self.assertSimilarMatrix(H2, np.kron(i2, Sy))

        H2 = ConstantNNHamiltonian(2, [2, 3])
        H2.add_local_term(0, σx)
        H2.add_local_term(1, Sy)
        self.assertSimilar(H2.interaction_term(0), np.kron(σx, i3) + np.kron(i2, Sy))
        self.assertSimilarMatrix(H2, np.kron(σx, i3) + np.kron(i2, Sy))

    def test_hamiltonian_to_mpo(self):
        """Check conversion to MPO is accurate by comparing matrices."""
        H2 = HeisenbergHamiltonian(2)
        self.assertSimilar(H2.to_matrix().toarray(), H2.to_mpo().to_matrix())

        H3 = HeisenbergHamiltonian(3)
        self.assertSimilar(H3.to_matrix().toarray(), H3.to_mpo().to_matrix())

        H4 = HeisenbergHamiltonian(4)
        self.assertSimilar(H4.to_matrix().toarray(), H4.to_mpo().to_matrix())
