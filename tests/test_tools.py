import numpy as np
from seemps.tools import random_isometry, random_Pauli
from .tools import TestCase, almostIsometry, almostIdentity
import os


class TestTools(TestCase):
    def test_random_isometry(self):
        for N in range(1, 10):
            for M in range(1, 10):
                A = random_isometry(N, M)
                self.assertTrue(almostIsometry(A))

    def test_random_Pauli(self):
        for _ in range(100):
            σ = random_Pauli()
            self.assertTrue(almostIdentity(σ @ σ))
            self.assertTrue(np.sum(np.abs(σ.T.conj() - σ)) == 0)


if "DEBUGSEEMPS" in os.environ:
    from seemps import tools

    tools.DEBUG = int(os.environ["DEBUGSEEMPS"])

__all__ = ["TestTools"]
