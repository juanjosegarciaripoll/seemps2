import numpy as np
from .tools import TestCase


class MPSStatesFixture(TestCase):
    def setUp(self):
        self.product_state = [
            np.reshape([1.0, 2.0], (1, 2, 1)),
            np.reshape([3.0, 5.0], (1, 2, 1)),
            np.reshape([7.0, 11.0], (1, 2, 1)),
        ]
        self.product_state_dimension = 2**3
        self.product_state_wavefunction = np.kron(
            [1.0, 2.0], np.kron([3.0, 5.0], [7.0, 11.0])
        )

        self.inhomogeneous_state = [
            np.reshape([1.0, 2.0], (1, 2, 1)),
            np.reshape([3.0, 5.0, -1.0], (1, 3, 1)),
            np.reshape([7.0, -5.0, -9.0, 11.0], (1, 4, 1)),
        ]
        self.inhomogeneous_state_dimension = 2 * 3 * 4
        self.inhomogeneous_state_wavefunction = np.kron(
            [1.0, 2.0], np.kron([3.0, 5.0, -1.0], [7.0, -5.0, -9.0, 11.0])
        )

        self.other_tensor = np.reshape([13, 15], (1, 2, 1))
