import numpy as np
import scipy.sparse as sp  # type: ignore
from seemps.register.transforms import mpo_weighted_shifts, mpo_shifts
from ..tools import TestCase


class TestShifts(TestCase):
    def shift_matrix(self, L: int, d: int, periodic: bool):
        S = sp.dok_array((L, L))
        for i in range(L):
            j = i + d
            if periodic:
                S[j % L, i] = 1.0
            elif 0 <= j < L:
                S[j, i] = 1.0
        return S.toarray()

    def weighted_shift_matrix(
        self, L: int, weights: np.ndarray, shifts: np.ndarray, periodic: bool
    ):
        return sum(
            w * self.shift_matrix(L, d, periodic) for w, d in zip(weights, shifts)
        )

    def test_shift_by_one(self):
        S = mpo_shifts(3, shifts=[1])
        matrix = S.to_matrix()
        target = [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
        ]
        self.assertSimilar(matrix, target)

    def test_shift_by_two(self):
        S = mpo_shifts(3, shifts=[2])
        matrix = S.to_matrix()
        target = [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
        ]
        self.assertSimilar(matrix, target)

    def test_shift_by_minus_one(self):
        S = mpo_shifts(3, shifts=[-1])
        matrix = S.to_matrix()
        target = [
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ]
        self.assertSimilar(matrix, target)

    def test_shift_by_minus_two(self):
        S = mpo_shifts(3, shifts=[-2])
        matrix = S.to_matrix()
        target = [
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ]
        self.assertSimilar(matrix, target)

    def test_shift_by_one_periodic(self):
        S = mpo_shifts(3, shifts=[1], periodic=True)
        matrix = S.to_matrix()
        target = [
            [0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
        ]
        self.assertSimilar(matrix, target)

    def test_shift_by_two_periodic(self):
        S = mpo_shifts(3, shifts=[2], periodic=True)
        matrix = S.to_matrix()
        target = [
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
        ]
        self.assertSimilar(matrix, target)

    def test_shift_by_minus_one_periodic(self):
        S = mpo_shifts(3, shifts=[-1], periodic=True)
        matrix = S.to_matrix()
        target = [
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0],
        ]
        self.assertSimilar(matrix, target)

    def test_shift_by_minus_two_periodic(self):
        S = mpo_shifts(3, shifts=[-2], periodic=True)
        matrix = S.to_matrix()
        target = [
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
        ]
        self.assertSimilar(matrix, target)

    def test_shift_by_all_integers(self):
        for N in range(1, 4):
            for d in range(-(2**N), 2**N + 1):
                S = mpo_shifts(N, shifts=[d], periodic=False)
                target = self.shift_matrix(2**N, d, periodic=False)
                self.assertSimilar(S.to_matrix(), target)

    def test_shift_by_all_integers_periodic(self):
        for N in range(1, 4):
            for d in range(-(2**N), 2**N + 1):
                S = mpo_shifts(N, shifts=[d], periodic=True)
                target = self.shift_matrix(2**N, d, periodic=True)
                self.assertSimilar(S.to_matrix(), target)

    def test_weighted_shifts_by_all_integers(self):
        for N in range(1, 4):
            for d in range(1, 2**N + 1):
                shifts = np.arange(-d, d + 1)
                weights = self.rng.normal(size=shifts.shape)
                S = mpo_weighted_shifts(N, weights, shifts.tolist(), periodic=False)
                target = self.weighted_shift_matrix(
                    2**N, weights, shifts, periodic=False
                )
                self.assertSimilar(S.to_matrix(), target)
