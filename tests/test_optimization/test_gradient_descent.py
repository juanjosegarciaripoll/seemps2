from seemps.state import MPS
from seemps.operators import MPO
from seemps.optimization.descent import gradient_descent, OptimizeResults
from .tools import TestOptimizeCase


class TestGradientDescent(TestOptimizeCase):
    def solve(self, H: MPO, state: MPS, **kwdargs) -> OptimizeResults:
        if "maxiter" not in kwdargs:
            kwdargs["maxiter"] = 100
        return gradient_descent(H, state, **kwdargs)
