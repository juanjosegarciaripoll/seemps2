import numpy as np

from seemps import MPO, product_state
from seemps.analysis.evolution import *
from seemps.hamiltonians import HeisenbergHamiltonian

from ..tools import *


def callback():
    norms = []
    def callback_func(state: MPS):
        if state is None:
            norms.pop()
            return None
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
        result = euler(H, guess)
        self.assertAlmostEqual(result.energy, H.expectation(exact))
        self.assertSimilar(result.state, exact, atol=1e-4)

    def test_euler_with_callback(self):
        N = 4
        maxiter = 10
        H = self.make_local_Sz_mpo(N)
        guess = product_state(np.asarray([1, 1]) / np.sqrt(2.0), N)
        callback_func, norms = callback()
        result = euler(H, guess, maxiter=maxiter, callback=callback_func)
        self.assertSimilar(norms, np.ones(len(norms)))
        self.assertEqual(maxiter + 1, len(norms))


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
        result = improved_euler(H, guess)
        self.assertAlmostEqual(result.energy, H.expectation(exact))
        self.assertSimilar(result.state, exact, atol=1e-4)

    def test_improved_euler_with_callback(self):
        N = 4
        maxiter = 10
        H = self.make_local_Sz_mpo(N)
        guess = product_state(np.asarray([1, 1]) / np.sqrt(2.0), N)
        callback_func, norms = callback()
        result = improved_euler(H, guess, maxiter=maxiter, callback=callback_func)
        self.assertSimilar(norms, np.ones(len(norms)))
        self.assertEqual(maxiter + 1, len(norms))


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
        result = runge_kutta(H, guess)
        self.assertAlmostEqual(result.energy, H.expectation(exact))
        self.assertSimilar(result.state, exact, atol=1e-4)

    def test_runge_kutta_with_callback(self):
        N = 4
        maxiter = 10
        H = self.make_local_Sz_mpo(N)
        guess = product_state(np.asarray([1, 1]) / np.sqrt(2.0), N)
        callback_func, norms = callback()
        result = runge_kutta(H, guess, maxiter=maxiter, callback=callback_func)
        self.assertSimilar(norms, np.ones(len(norms)))
        self.assertEqual(maxiter + 1, len(norms))


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
        result = runge_kutta_fehlberg(H, guess, maxiter=100, tol_rk=1e-7)
        self.assertAlmostEqual(result.energy, H.expectation(exact))
        self.assertSimilar(result.state, exact, atol=1e-4)

    def test_runge_kutta_fehlberg_with_callback(self):
        N = 4
        maxiter = 10
        H = self.make_local_Sz_mpo(N)
        guess = product_state(np.asarray([1, 1]) / np.sqrt(2.0), N)
        callback_func, norms = callback()
        result = runge_kutta_fehlberg(H, guess, maxiter=maxiter, callback=callback_func)
        self.assertSimilar(norms, np.ones(len(norms)))
