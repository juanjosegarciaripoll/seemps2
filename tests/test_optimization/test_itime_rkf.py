from seemps.state import MPS
from seemps.operators import MPO
from seemps.analysis.evolution import runge_kutta_fehlberg
from .tools import TestItimeCase


class TestRungeKuttaFehlberg(TestItimeCase):
    def solve(self, H: MPO, state: MPS, **kwdargs):
        return runge_kutta_fehlberg(H, state, tol_rk=1e-7, **kwdargs)
