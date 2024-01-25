from seemps.state import MPS
from seemps.operators import MPO
from seemps.analysis.evolution import runge_kutta
from .tools import TestItimeCase


class TestRungeKutta(TestItimeCase):
    def solve(self, H: MPO, state: MPS, **kwdargs):
        return runge_kutta(H, state, **kwdargs)
