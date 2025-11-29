from __future__ import annotations
import numpy as np
import unittest
from abc import abstractmethod
from typing import Any
from math import sqrt
from seemps.state import MPS, product_state
from seemps.operators import MPO
from seemps.analysis.evolution import EvolutionResults
from seemps.optimization.descent import OptimizeResults
from seemps.typing import DenseOperator
from ..tools import TestCase


class TestItimeCase(TestCase):
    Sz: DenseOperator = np.diag([0.5, -0.5])

    def make_problem_and_solution(self, size: int) -> tuple[MPO, MPS]:
        A = np.zeros((2, 2, 2, 2))
        A[0, :, :, 0] = np.eye(2)
        A[1, :, :, 1] = np.eye(2)
        A[0, :, :, 1] = self.Sz
        tensors = [A] * size
        tensors[0] = tensors[0][[0], :, :, :]
        tensors[-1] = tensors[-1][:, :, :, [1]]
        return MPO(tensors), product_state([0, 1], size)

    @classmethod
    def setUpClass(cls):
        if cls is TestItimeCase:
            raise unittest.SkipTest(f"Skip {cls} tests, it's a base class")
        super().setUpClass()

    def make_callback(self):
        norms = []

        def callback_func(state: MPS, results: EvolutionResults):
            self.assertIsInstance(results, EvolutionResults)
            self.assertIsInstance(state, MPS)
            norms.append(sqrt(state.norm_squared()))
            return None

        return callback_func, norms

    @abstractmethod
    def solve(self, H: MPO, state: MPS, **kwdargs) -> Any:
        raise Exception("solve() not implemented")

    def test_eigenvalue_solver_with_local_field(self):
        N = 4
        H, exact = self.make_problem_and_solution(N)
        guess = product_state(np.asarray([1, 1]) / sqrt(2.0), N)
        result = self.solve(H, guess)
        self.assertAlmostEqual(result.energy, H.expectation(exact))
        self.assertSimilarStates(result.state, exact, atol=1e-4)

    def test_eigenvalue_solver_with_callback(self):
        N = 4
        H, _ = self.make_problem_and_solution(N)
        guess = product_state(np.asarray([1, 1]) / sqrt(2.0), N)
        callback_func, norms = self.make_callback()
        self.solve(H, guess, maxiter=10, callback=callback_func)
        self.assertSimilar(norms, np.ones(len(norms)))


class TestOptimizeCase(TestCase):
    Sz = np.diag([0.5, -0.5])

    @classmethod
    def setUpClass(cls):
        if cls is TestOptimizeCase:
            raise unittest.SkipTest("Skip TestOptimizeCase tests, it's a base class")
        super().setUpClass()

    def make_problem_and_solution(self, size: int) -> tuple[MPO, MPS]:
        A = np.zeros((2, 2, 2, 2))
        A[0, :, :, 0] = np.eye(2)
        A[1, :, :, 1] = np.eye(2)
        A[0, :, :, 1] = self.Sz
        tensors = [A] * size
        tensors[0] = tensors[0][[0], :, :, :]
        tensors[-1] = tensors[-1][:, :, :, [1]]
        return MPO(tensors), product_state([0, 1], size)

    def make_callback(self):
        norms = []

        def callback_func(state: MPS, results: OptimizeResults):
            self.assertIsInstance(results, OptimizeResults)
            self.assertIsInstance(state, MPS)
            norms.append(sqrt(state.norm_squared()))
            return None

        return callback_func, norms

    @abstractmethod
    def solve(self, H: MPO, state: MPS, **kwdargs) -> OptimizeResults:
        raise Exception("solve() not implemented")

    def test_eigenvalue_solver_with_local_field(self):
        N = 4
        H, exact = self.make_problem_and_solution(N)
        guess = product_state(np.asarray([1, 1]) / sqrt(2.0), N)
        result = self.solve(H, guess)
        self.assertAlmostEqual(result.energy, H.expectation(exact))
        self.assertSimilarStates(result.state, exact, atol=1e-4)

    def test_eigenvalue_solver_with_callback(self):
        N = 4
        H, _ = self.make_problem_and_solution(N)
        guess = product_state(np.asarray([1, 1]) / sqrt(2.0), N)
        callback_func, norms = self.make_callback()
        self.solve(H, guess, maxiter=10, callback=callback_func)
        self.assertSimilar(norms, np.ones(len(norms)))

    def test_eigenvalue_solver_acknowledges_tolerance(self):
        """Check that algorithm stops if energy change is below tolerance."""
        N = 4
        tol = 1e-5
        H, _ = self.make_problem_and_solution(4)
        guess = product_state(np.asarray([1, 1]) / sqrt(2.0), N)
        result = self.solve(H, guess, tol=tol)
        self.assertTrue(result.converged)
        self.assertTrue(abs(result.trajectory[-1] - result.trajectory[-2]) < tol)
