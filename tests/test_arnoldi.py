import numpy as np

from seemps import MPO, product_state
from seemps.hamiltonians import HeisenbergHamiltonian
from seemps.optimization.arnoldi import arnoldi_eigh

from .tools import *


class TestArnoldiEigH(TestCase):
    Sz = np.diag([0.5, -0.5])

    def make_local_Sz_mpo(self, size: int) -> MPO:
        A = np.zeros((2, 2, 2, 2))
        A[0, :, :, 0] = np.eye(2)
        A[1, :, :, 1] = np.eye(2)
        A[0, :, :, 1] = self.Sz
        tensors = [A] * size
        tensors[0] = tensors[0][[0], :, :, :]
        tensors[-1] = tensors[-1][:, :, :, [1]]
        return MPO(tensors)

    def test_arnoldi_eigh_with_local_field(self):
        N = 4
        H = self.make_local_Sz_mpo(N)
        guess = product_state(np.asarray([1, 1]) / np.sqrt(2.0), N)
        exact = product_state([0, 1], N)
        result = arnoldi_eigh(H, guess, maxiter=20, tol=1e-15)
        self.assertAlmostEqual(result.energy, H.expectation(exact))
        self.assertSimilarStates(result.state, exact, atol=1e-7)
