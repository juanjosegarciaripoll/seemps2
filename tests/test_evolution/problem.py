import numpy as np
from seemps.tools import σx, σy, σz
from seemps.state import product_state, MPS
from ..tools import TestCase


class EvolutionTestCase(TestCase):
    Heisenberg2 = 0.25 * (np.kron(σx, σx) + np.kron(σy, σy) + np.kron(σz, σz))

    def random_initial_state(self, size: int) -> MPS:
        states = np.random.randn(size, 2) + 1j * np.random.randn(size, 2)
        for n in range(size):
            states[n, :] /= np.linalg.norm(states[n, :])
        return product_state(states)
