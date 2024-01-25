from seemps.state import MPS
from seemps.operators import MPO
from seemps.analysis.evolution import euler
from .tools import TestItimeCase


class TestEuler(TestItimeCase):
    def solve(self, H: MPO, state: MPS, **kwdargs):
        return euler(H, state, **kwdargs)
