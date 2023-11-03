import numpy as np
from seemps.analysis.evolution import *
from seemps.hamiltonians import HeisenbergHamiltonian
from seemps import MPO, product_state
from .tools import *

def callback():
        norms = []
        def callback_func(state:MPS):
            norms.append(np.sqrt(state.norm_squared()))
            return None
        return callback_func, norms

class TestEuler(TestCase):
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

    def test_euler_with_local_field(self):
        N = 4
        H = self.make_local_Sz_mpo(N)
        guess = product_state(np.asarray([1, 1]) / np.sqrt(2.0), N)
        exact = product_state([0, 1], N)
        result = euler(H, guess, Δβ=0.1, tol=1e-15)
        self.assertAlmostEqual(result.energy, H.expectation(exact))
        self.assertSimilar(result.state, exact, atol=1e-7)

    def test_euler_acknowledges_tolerance(self):
        """Check that algorithm stops if energy change is below tolerance."""
        N = 4
        H = HeisenbergHamiltonian(N).to_mpo()
        tol = 1e-5
        guess = CanonicalMPS(
            random_uniform_mps(2, N, rng=self.rng), center=0, normalize=True
        )
        result = euler(H, guess, tol=tol, maxiter=1000)
        self.assertTrue(result.converged)
        self.assertTrue(abs(result.trajectory[-1] - result.trajectory[-2]) < tol)

    def test_euler_with_callback(self):
        N = 4
        maxiter = 10
        H = self.make_local_Sz_mpo(N)
        guess = product_state(np.asarray([1, 1]) / np.sqrt(2.0), N)
        callback_func, norms = callback()
        result = euler(H, guess, maxiter=maxiter, tol=1e-15, callback=callback_func)
        self.assertSimilar(norms, np.ones(len(norms)))
        self.assertEqual(maxiter, len(norms))

class TestImprovedEuler(TestCase):
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

    def test_improved_euler_with_local_field(self):
        N = 4
        H = self.make_local_Sz_mpo(N)
        guess = product_state(np.asarray([1, 1]) / np.sqrt(2.0), N)
        exact = product_state([0, 1], N)
        result = improved_euler(H, guess, Δβ=0.1, tol=1e-15)
        self.assertAlmostEqual(result.energy, H.expectation(exact))
        self.assertSimilar(result.state, exact, atol=1e-7)

    def test_improved_euler_acknowledges_tolerance(self):
        """Check that algorithm stops if energy change is below tolerance."""
        N = 4
        H = HeisenbergHamiltonian(N).to_mpo()
        tol = 1e-5
        guess = CanonicalMPS(
            random_uniform_mps(2, N, rng=self.rng), center=0, normalize=True
        )
        result = improved_euler(H, guess, tol=tol, maxiter=1000)
        self.assertTrue(result.converged)
        self.assertTrue(abs(result.trajectory[-1] - result.trajectory[-2]) < tol)

    def test_improved_euler_with_callback(self):
        N = 4
        maxiter = 10
        H = self.make_local_Sz_mpo(N)
        guess = product_state(np.asarray([1, 1]) / np.sqrt(2.0), N)
        callback_func, norms = callback()
        result = improved_euler(H, guess, maxiter=maxiter, tol=1e-15, callback=callback_func)
        self.assertSimilar(norms, np.ones(len(norms)))
        self.assertEqual(maxiter, len(norms))


class TestRungeKutta(TestCase):
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

    def test_runge_kutta_with_local_field(self):
        N = 4
        H = self.make_local_Sz_mpo(N)
        guess = product_state(np.asarray([1, 1]) / np.sqrt(2.0), N)
        exact = product_state([0, 1], N)
        result = runge_kutta(H, guess, Δβ=0.1, tol=1e-15)
        self.assertAlmostEqual(result.energy, H.expectation(exact))
        self.assertSimilar(result.state, exact, atol=1e-7)

    def test_runge_kutta_acknowledges_tolerance(self):
        """Check that algorithm stops if energy change is below tolerance."""
        N = 4
        H = HeisenbergHamiltonian(N).to_mpo()
        tol = 1e-5
        guess = CanonicalMPS(
            random_uniform_mps(2, N, rng=self.rng), center=0, normalize=True
        )
        result = runge_kutta(H, guess, tol=tol, maxiter=1000)
        self.assertTrue(result.converged)
        self.assertTrue(abs(result.trajectory[-1] - result.trajectory[-2]) < tol)

    def test_runge_kutta_with_callback(self):
        N = 4
        maxiter = 10
        H = self.make_local_Sz_mpo(N)
        guess = product_state(np.asarray([1, 1]) / np.sqrt(2.0), N)
        callback_func, norms = callback()
        result = runge_kutta(H, guess, maxiter=maxiter, tol=1e-15, callback=callback_func)
        self.assertSimilar(norms, np.ones(len(norms)))
        self.assertEqual(maxiter, len(norms))

class TestRungeKuttaFehlberg(TestCase):
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

    def test_runge_kutta_fehlberg_with_local_field(self):
        N = 4
        H = self.make_local_Sz_mpo(N)
        guess = product_state(np.asarray([1, 1]) / np.sqrt(2.0), N)
        exact = product_state([0, 1], N)
        result = runge_kutta_fehlberg(H, guess, Δβ=0.1, tol=1e-15, tol_rk=1e-7)
        self.assertAlmostEqual(result.energy, H.expectation(exact))
        self.assertSimilar(result.state, exact, atol=1e-7)

    def test_runge_kutta_fehlberg_acknowledges_tolerance(self):
        """Check that algorithm stops if energy change is below tolerance."""
        N = 4
        H = HeisenbergHamiltonian(N).to_mpo()
        tol = 1e-5
        guess = CanonicalMPS(
            random_uniform_mps(2, N, rng=self.rng), center=0, normalize=True
        )
        result = runge_kutta_fehlberg(H, guess, tol=tol, maxiter=1000)
        self.assertTrue(result.converged)
        self.assertTrue(abs(result.trajectory[-1] - result.trajectory[-2]) < tol)

    def test_runge_kutta_fehlberg_with_callback(self):
        N = 4
        maxiter = 10
        H = self.make_local_Sz_mpo(N)
        guess = product_state(np.asarray([1, 1]) / np.sqrt(2.0), N)
        callback_func, norms = callback()
        result = runge_kutta_fehlberg(H, guess, maxiter=maxiter, tol=1e-15, callback=callback_func)
        self.assertSimilar(norms, np.ones(len(norms)))