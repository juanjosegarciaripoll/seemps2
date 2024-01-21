import numpy as np

from seemps import MPO, product_state
from seemps.hamiltonians import HeisenbergHamiltonian
from seemps.optimization.arnoldi import arnoldi_eigh

from ..tools import *


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

    def call_arnoldi(self, *args, **kwdargs):
        N = 4
        H = self.make_local_Sz_mpo(N)
        guess = product_state(np.asarray([1, 1]) / np.sqrt(2.0), N)
        exact = product_state([0, 1], N)
        result = arnoldi_eigh(H, guess, *args, **kwdargs)
        exact_energy = H.expectation(exact)
        return result, exact, exact_energy

    def test_arnoldi_solves_small_problem(self):
        result, exact, exact_energy = self.call_arnoldi(
            nvectors=10, maxiter=20, tol=1e-15
        )
        self.assertAlmostEqual(result.energy, exact_energy)
        self.assertSimilarStates(result.state, exact, atol=1e-7)

    def test_arnoldi_stops_if_tolerance_reached(self):
        result, exact, exact_energy = self.call_arnoldi(
            nvectors=10, maxiter=300, tol=1e-15
        )
        self.assertTrue(len(result.trajectory) < 300)

    def test_arnoldi_invokes_callback(self):
        callback_calls = 0

        def callback(state, E, results):
            nonlocal callback_calls
            callback_calls += 1

        result, *_ = self.call_arnoldi(maxiter=20, tol=1e-15, callback=callback)
        self.assertTrue(callback_calls > 0)
        self.assertEqual(callback_calls, len(result.trajectory))

    def test_arnoldi_signals_non_convergence(self):
        result, _, exact_energy = self.call_arnoldi(nvectors=2, maxiter=3, tol=1e-15)
        self.assertFalse(result.converged)
        self.assertFalse(np.isclose(result.energy, exact_energy))
