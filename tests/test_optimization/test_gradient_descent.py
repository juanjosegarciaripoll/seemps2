from seemps.state import MPS
from seemps.operators import MPO
from seemps.optimization.descent import gradient_descent, OptimizeResults
from .tools import TestOptimizeCase


class TestGradientDescent(TestOptimizeCase):
    def solve(self, H: MPO, state: MPS, maxiter=100, **kwdargs) -> OptimizeResults:
        return gradient_descent(H, state, maxiter=maxiter, **kwdargs)
