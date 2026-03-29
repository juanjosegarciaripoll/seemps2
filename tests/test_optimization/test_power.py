from unittest.mock import patch
import numpy as np
from seemps.state import MPS
from seemps.operators import MPO
from seemps.optimization.power import power_method, OptimizeResults
from ..tools import SeeMPSTestCase
from .tools import TestOptimizeCase


class TestPowerMethod(TestOptimizeCase):
    def solve(self, H: MPO, state: MPS, **kwdargs) -> OptimizeResults:
        if "maxiter" not in kwdargs:
            kwdargs["maxiter"] = 100
        shift = 4.0
        results = power_method(H, inverse=True, shift=shift, guess=state, **kwdargs)
        results.energy -= shift
        results.trajectory = [E - shift for E in results.trajectory]
        return results


class FakePowerOperator:
    def __init__(self, energies, image=None, dimensions=None):
        self.energies = list(energies)
        self.image = image
        self.dimensions = [2] if dimensions is None else dimensions
        self.calls = 0
        self.added = None
        self.joined = False

    def physical_dimensions(self):
        return self.dimensions

    def expectation(self, state):
        energy = self.energies[min(self.calls, len(self.energies) - 1)]
        self.calls += 1
        return energy

    def __matmul__(self, state):
        return state if self.image is None else self.image

    def __add__(self, other):
        self.added = other
        return self

    def join(self):
        self.joined = True
        return self


class TestPowerMethodInternals(SeeMPSTestCase):
    def test_power_method_handles_shift_and_upward_fluctuations(self):
        guess = MPS.from_vector(np.array([1.0, 0.0]), [2], normalize=False)
        H = FakePowerOperator([1.0, 2.0])
        callbacks = []

        with patch("seemps.optimization.power.simplify", side_effect=lambda state, strategy: state):
            results = power_method(
                H,
                guess=guess,
                shift=0.25,
                inverse=False,
                maxiter=3,
                tol=1e-9,
                tol_variance=0.0,
                tol_up=0.1,
                upward_moves=0,
                callback=lambda state, results: callbacks.append(results.energy),
            )

        self.assertIsNotNone(H.added)
        self.assertTrue(H.joined)
        self.assertTrue(results.converged)
        self.assertIn("Energy fluctuates upwards", results.message)
        self.assertEqual(len(callbacks), 2)

    def test_power_method_detects_stationary_state(self):
        guess = MPS.from_vector(np.array([1.0, 0.0]), [2], normalize=False)
        H = FakePowerOperator([1.0], image=guess)

        with patch("seemps.optimization.power.simplify", side_effect=lambda state, strategy: state):
            results = power_method(
                H,
                guess=guess,
                inverse=False,
                maxiter=2,
                tol=1e-9,
                tol_variance=1e-12,
            )

        self.assertTrue(results.converged)
        self.assertIn("Stationary state reached", results.message)

    def test_power_method_inverse_defaults_tol_cgs_and_tracks_cgs_steps(self):
        guess = MPS.from_vector(np.array([1.0, 0.0]), [2], normalize=False)
        H = FakePowerOperator([2.0, 1.8], image=guess)
        captured = {}

        def fake_cgs_solve(H, state, guess, maxiter, tolerance, strategy, callback=None):
            captured["tolerance"] = tolerance
            captured["callback_is_none"] = callback is None
            return state, 0.0

        with patch("seemps.optimization.power.cgs_solve", side_effect=fake_cgs_solve):
            results = power_method(
                H,
                guess=guess,
                inverse=True,
                maxiter=2,
                tol=1e-9,
                tol_variance=0.5,
                tol_cgs=None,
            )

        self.assertEqual(captured["tolerance"], 0.5)
        self.assertTrue(captured["callback_is_none"])
        self.assertEqual(results.steps, [0, 0])
