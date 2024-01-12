import scipy.sparse as sp  # type: ignore
from seemps.tools import σx, σy, σz
from seemps.hamiltonians import ConstantNNHamiltonian, HeisenbergHamiltonian
from ..tools import *

i2 = sp.eye(2)


class TestHamiltonians(TestCase):
    def test_nn_construct(self):
        H2 = ConstantNNHamiltonian(2, 2)
        H2.add_local_term(0, σx)
        M2 = H2.interaction_term(0)
        A2 = sp.kron(σx, i2)
        self.assertTrue(similar(M2, A2))

        H2 = ConstantNNHamiltonian(2, 2)
        H2.add_local_term(1, σy)
        M2 = H2.interaction_term(0)
        A2 = sp.kron(i2, σy)
        self.assertSimilar(M2, A2)

        H3 = ConstantNNHamiltonian(3, 2)
        H3.add_local_term(1, σy)
        M3 = H3.interaction_term(0)
        A3 = sp.kron(i2, 0.5 * σy)
        self.assertSimilar(M3, A3)
        M3 = H3.interaction_term(1)
        A3 = sp.kron(0.5 * σy, i2)
        self.assertSimilar(M3, A3)

    def test_sparse_matrix(self):
        H2 = ConstantNNHamiltonian(2, 2)
        H2.add_interaction_term(0, σz, σz)
        M2 = H2.tomatrix()
        A2 = sp.kron(σz, σz)
        self.assertSimilar(M2, A2)

        H2 = ConstantNNHamiltonian(2, 2)
        H2.add_local_term(0, 3.5 * σx)
        M2 = H2.tomatrix()
        A2 = sp.kron(3.5 * σx, i2)
        self.assertSimilar(M2, A2)

        H2 = ConstantNNHamiltonian(2, 2)
        H2.add_local_term(1, -2.5 * σy)
        M2 = H2.tomatrix()
        A2 = sp.kron(i2, -2.5 * σy)
        self.assertSimilar(M2, A2)

        H2 = ConstantNNHamiltonian(2, 2)
        H2.add_local_term(0, 3.5 * σx)
        H2.add_local_term(1, -2.5 * σy)
        H2.add_interaction_term(0, σz, σz)
        M2 = H2.tomatrix()
        A2 = sp.kron(i2, -2.5 * σy) + sp.kron(σz, σz) + sp.kron(3.5 * σx, i2)
        self.assertSimilar(M2, A2)

    def test_hamiltonian_to_mpo(self):
        """Check conversion to MPO is accurate by comparing matrices."""
        H2 = HeisenbergHamiltonian(2)
        self.assertSimilar(H2.tomatrix().toarray(), H2.to_mpo().tomatrix())

        H3 = HeisenbergHamiltonian(3)
        self.assertSimilar(H3.tomatrix().toarray(), H3.to_mpo().tomatrix())

        H4 = HeisenbergHamiltonian(4)
        self.assertSimilar(H4.tomatrix().toarray(), H4.to_mpo().tomatrix())
