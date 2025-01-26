from seemps.state import MPS
from seemps.operators import MPO
from seemps.optimization.descent import gradient_descent, OptimizeResults
from .tools import TestOptimizeCase


class TestGradientDescent(TestOptimizeCase):
    def solve(self, H: MPO, state: MPS, **kwdargs) -> OptimizeResults:
        maxiter = kwdargs.get("maxiter", 100)
        return gradient_descent(H, state, maxiter=maxiter, **kwdargs)
