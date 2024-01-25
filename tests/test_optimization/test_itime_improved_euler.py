from seemps.state import MPS
from seemps.operators import MPO
from seemps.analysis.evolution import improved_euler
from .tools import TestItimeCase


class TestImprovedEuler(TestItimeCase):
    def solve(self, H: MPO, state: MPS, **kwdargs):
        return improved_euler(H, state, **kwdargs)
