from seemps.tools import *
from .tools import *
import os


class TestTools(TestCase):
    def test_random_isometry(self):
        for N in range(1, 10):
            for M in range(1, 10):
                A = seemps.tools.random_isometry(N, M)
                self.assertTrue(almostIsometry(A))

    def test_random_Pauli(self):
        for _ in range(100):
            σ = random_Pauli()
            self.assertTrue(almostIdentity(σ @ σ))
            self.assertTrue(np.sum(np.abs(σ.T.conj() - σ)) == 0)


if "DEBUGSEEMPS" in os.environ:
    from seemps import tools

    tools.DEBUG = int(os.environ["DEBUGSEEMPS"])
