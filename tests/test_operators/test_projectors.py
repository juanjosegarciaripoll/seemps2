import numpy as np
from .. import tools
from seemps.operators.projectors import basis_states_projector_mpo, ALL_STATES


def diag_projector(ndx: int, d: int):
    A = np.zeros((d, d))
    A[ndx, ndx] = 1.0
    return A


class TestBasisProjector(tools.TestCase):
    def test_projector_rejects_out_of_bound_indices(self):
        with self.assertRaises(Exception):
            basis_states_projector_mpo([[0, 3]], [2, 2])
        with self.assertRaises(Exception):
            basis_states_projector_mpo([[2, 0]], [2, 2])
        with self.assertRaises(Exception):
            basis_states_projector_mpo([[-2, 0]], [2, 2])
        with self.assertRaises(Exception):
            basis_states_projector_mpo([[0, -2]], [2, 2])

        with self.assertRaises(Exception):
            basis_states_projector_mpo([[0, (0, 3)]], [2, 2])
        with self.assertRaises(Exception):
            basis_states_projector_mpo([[(0, 2), 0]], [2, 2])
        with self.assertRaises(Exception):
            basis_states_projector_mpo([[(-2, 1), 0]], [2, 2])
        with self.assertRaises(Exception):
            basis_states_projector_mpo([[0, (1, -2)]], [2, 2])

    def test_projector_checks_lists_lengths(self):
        with self.assertRaises(Exception):
            basis_states_projector_mpo([[0, 3]], [2, 2, 2])

    def test_projector_00(self):
        P00 = basis_states_projector_mpo([[0, 0]], [2, 2])
        A00 = np.diag([1.0, 0.0, 0.0, 0.0])
        self.assertEqualTensors(P00.to_matrix(), A00)

    def test_projector_01(self):
        P01 = basis_states_projector_mpo([[0, 1]], [2, 2])
        A01 = np.diag([0.0, 1.0, 0.0, 0.0])
        self.assertEqualTensors(P01.to_matrix(), A01)

    def test_projector_10(self):
        P10 = basis_states_projector_mpo([[1, 0]], [2, 2])
        A10 = np.diag([0.0, 0.0, 1.0, 0.0])
        self.assertEqualTensors(P10.to_matrix(), A10)

    def test_projector_11(self):
        P11 = basis_states_projector_mpo([[1, 1]], [2, 2])
        A11 = np.diag([0.0, 0.0, 0.0, 1.0])
        self.assertEqualTensors(P11.to_matrix(), A11)

    def test_projector_0x(self):
        P0x = basis_states_projector_mpo([[0, ALL_STATES]], [2, 2])
        A0x = np.diag([1.0, 1.0, 0.0, 0.0])
        self.assertEqualTensors(P0x.to_matrix(), A0x)
        P0x = basis_states_projector_mpo([[0, (0, 1)]], [2, 2])
        A0x = np.diag([1.0, 1.0, 0.0, 0.0])
        self.assertEqualTensors(P0x.to_matrix(), A0x)

    def test_projector_x0(self):
        Px0 = basis_states_projector_mpo([[(0, 1), 0]], [2, 2])
        Ax0 = np.diag([1.0, 0.0, 1.0, 0.0])
        self.assertEqualTensors(Px0.to_matrix(), Ax0)

    def test_projector_1x(self):
        P1x = basis_states_projector_mpo([[1, (1, 0)]], [2, 2])
        A1x = np.diag([0.0, 0.0, 1.0, 1.0])
        self.assertEqualTensors(P1x.to_matrix(), A1x)

    def test_projector_x1(self):
        Px1 = basis_states_projector_mpo([[(1, 0), 1]], [2, 2])
        Ax1 = np.diag([0.0, 1.0, 0.0, 1.0])
        self.assertEqualTensors(Px1.to_matrix(), Ax1)
