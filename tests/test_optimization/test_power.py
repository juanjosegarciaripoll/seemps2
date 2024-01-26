from seemps.state import MPS
from seemps.operators import MPO
from seemps.optimization.power import power_method, OptimizeResults
from .tools import TestOptimizeCase
import seemps.tools


class TestPowerMethod(TestOptimizeCase):
    def solve(self, H: MPO, state: MPS, maxiter=100, **kwdargs) -> OptimizeResults:
        shift = 4.0
        results = power_method(
            H, inverse=True, shift=shift, guess=state, maxiter=maxiter, **kwdargs
        )
        results.energy -= shift
        results.trajectory = [E - shift for E in results.trajectory]
        return results
