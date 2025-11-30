import numpy as np
from abc import abstractmethod
import unittest

import seemps
from seemps.state import MPS, scprod
from seemps.truncate.simplify_mpo import mps_as_mpo
from seemps.analysis.mesh import Mesh, RegularInterval
from seemps.analysis.factories import mps_tensor_product
from seemps.analysis.integration import mps_fifth_order
from seemps.analysis.cross import (
    BlackBoxLoadMPS,
    BlackBoxLoadTT,
    BlackBoxLoadMPO,
    BlackBoxComposeMPS,
    cross_maxvol,
    cross_dmrg,
    cross_greedy,
    CrossStrategyGreedy,
)
from seemps.analysis.cross.cross import maxvol_square
from seemps.analysis.cross.cross_maxvol import maxvol_rectangular

from .tools_analysis import reorder_tensor
from ..tools import TestCase


def integration_callback(mps_quadrature: MPS):
    """
    Returns a callback function that can be used to compute the
    integral of the intermediate MPS in TCI.
    """

    def callback(mps: MPS, **kwargs) -> float:
        integral = scprod(mps, mps_quadrature)
        logger = kwargs.get("logger")
        if logger:
            logger(f"MPS integral={integral}")
        return integral  # type: ignore

    return callback


def gaussian_setup_mps(dims, n=5, a=-1, b=1):
    func = lambda tensor: np.exp(-(np.sum(tensor, axis=0) ** 2))
    intervals = [RegularInterval(a, b, 2**n) for _ in range(dims)]
    mesh = Mesh(intervals)  # type: ignore
    mesh_tensor = mesh.to_tensor()
    func_vector = func(mesh_tensor.T).reshape(-1)
    mps = MPS.from_vector(func_vector, [2] * (n * dims), normalize=False)
    return func, mesh, mps, func_vector


def gaussian_setup_1d_mpo(is_diagonal, n=5, a=-1, b=1):
    interval = RegularInterval(a, b, 2**n)
    vec_x = interval.to_vector()
    mps_identity = MPS([np.ones((1, 2, 1))] * n)
    mesh = Mesh([interval, interval])
    if is_diagonal:
        delta = lambda z: (z == 0).astype(int)
        func = lambda x, y: np.exp(-(x**2)) * delta(x - y)
    else:
        func = lambda x, y: np.exp(-(x**2 + y**2))
    return func, vec_x, mesh, mps_identity


class CrossTests(TestCase):
    @classmethod
    def setUpClass(cls):
        if cls is CrossTests:
            raise unittest.SkipTest(f"Skip {cls} tests, it's a base class")
        super().setUpClass()

    @abstractmethod
    def cross_method(self, function, *args, **kwdargs):
        raise Exception("cross_method not implemented in " + str(type(self)))

    def test_load_1d_mps(self, n=5):
        func, mesh, _, y = gaussian_setup_mps(1, n=n)
        black_box = BlackBoxLoadMPS(func, mesh)
        cross_results = self.cross_method(black_box)
        self.assertSimilar(y, cross_results.mps.to_vector())

    def test_load_2d_mps(self, n=5):
        func, mesh, _, y = gaussian_setup_mps(2, n=n)
        black_box = BlackBoxLoadMPS(func, mesh)
        cross_results = self.cross_method(black_box)
        self.assertSimilar(y, cross_results.mps.to_vector())

    def test_load_2d_mps_with_order_B(self, n=5):
        func, mesh, _, y = gaussian_setup_mps(2, n=n)
        black_box = BlackBoxLoadMPS(func, mesh, mps_order="B")
        cross_results = self.cross_method(black_box)
        qubits = [int(np.log2(s)) for s in mesh.dimensions]
        tensor = reorder_tensor(cross_results.mps.to_vector(), qubits)
        self.assertSimilar(y, tensor)

    def test_load_2d_tt(self, n=5):
        func, mesh, _, y = gaussian_setup_mps(2, n=n)
        black_box = BlackBoxLoadTT(func, mesh)
        cross_results = self.cross_method(black_box)
        vector = cross_results.mps.to_vector()
        self.assertSimilar(y, vector)

    def test_2d_integration_callback(self, n=8):
        a, b = -1, 1
        func = lambda tensor: np.exp(tensor[0] + tensor[1])  # f(x,y) = e^(x+y)
        interval = RegularInterval(a, b, 2**n, endpoint_right=True)
        mesh = Mesh([interval, interval])
        mps_quad_1d = mps_fifth_order(-1, 1, n)
        mps_quad = mps_tensor_product([mps_quad_1d, mps_quad_1d])
        exact_integral = (np.exp(b) - np.exp(a)) ** 2
        callback = integration_callback(mps_quad)
        black_box = BlackBoxLoadMPS(func, mesh)
        cross_results = self.cross_method(black_box, callback=callback)
        integral = cross_results.callback_output[-1]  # type: ignore
        self.assertAlmostEqual(integral, exact_integral)

    def test_load_1d_mpo_diagonal(self, n=5):
        func, x, mesh, mps_I = gaussian_setup_1d_mpo(is_diagonal=True, n=n)
        black_box = BlackBoxLoadMPO(func, mesh, is_diagonal=True)
        cross_results = self.cross_method(black_box)
        mps_diagonal = mps_as_mpo(cross_results.mps).apply(mps_I)
        self.assertSimilar(func(x, x), mps_diagonal.to_vector())

    def test_load_1d_mpo_nondiagonal(self, n=5):
        func, x, mesh, _ = gaussian_setup_1d_mpo(is_diagonal=False, n=n)
        black_box = BlackBoxLoadMPO(func, mesh)
        cross_results = self.cross_method(black_box)
        y_mps = mps_as_mpo(cross_results.mps).to_matrix()
        xx, yy = np.meshgrid(x, x)
        self.assertSimilar(func(xx, yy), y_mps)

    def test_compose_1d_mps_list(self, n=5):
        _, _, mps_0, y_0 = gaussian_setup_mps(1, n=n)
        func = lambda v: v[0] + np.sin(v[1]) + np.cos(v[2])
        black_box = BlackBoxComposeMPS(func, [mps_0, mps_0, mps_0])
        cross_results = self.cross_method(black_box)
        self.assertSimilar(func([y_0, y_0, y_0]), cross_results.mps.to_vector())

    def test_compose_2d_mps_list(self, n=5):
        _, _, mps_0, y_0 = gaussian_setup_mps(2, n=n)
        func = lambda v: v[0] + np.sin(v[1]) + np.cos(v[2])
        black_box = BlackBoxComposeMPS(func, [mps_0, mps_0, mps_0])
        cross_results = self.cross_method(black_box)
        self.assertSimilar(func([y_0, y_0, y_0]), cross_results.mps.to_vector())


class TestCrossMaxvol(CrossTests):
    def cross_method(self, function, *args, **kwdargs):
        return cross_maxvol(function, *args, **kwdargs)


class TestCrossDMRG(CrossTests):
    def cross_method(self, function, *args, **kwdargs):
        return cross_dmrg(function, *args, **kwdargs)


class TestCrossGreedyFull(CrossTests):
    def cross_method(self, function, *args, **kwdargs):
        return cross_greedy(
            function,
            *args,
            cross_strategy=CrossStrategyGreedy(partial=False),
            **kwdargs,
        )


class TestCrossGreedyPartial(CrossTests):
    def cross_method(self, function, *args, **kwdargs):
        return cross_greedy(
            function,
            *args,
            cross_strategy=CrossStrategyGreedy(partial=True),
            **kwdargs,
        )


class TestSkeleton(TestCase):
    @staticmethod
    def random_matrix(m=1000, n=1000, r=5):
        """Computes a m x n random matrix of rank r"""
        return np.dot(np.random.rand(m, r), np.random.rand(r, n))

    def test_maxvol_square(self):
        A = self.random_matrix()
        J = np.random.choice(A.shape[1], 5, replace=False)
        I = 0  # unused, to keep typechecker happy
        for _ in range(1):
            C = A[:, J]
            I, _ = maxvol_square(C)
            R = A[I, :]
            J, _ = maxvol_square(R.T)
        A_new = A[:, J] @ np.linalg.inv(A[I, :][:, J]) @ A[I, :]
        self.assertSimilar(A, A_new)

    def test_maxvol_rectangular(self):
        A = self.random_matrix()
        J = np.random.choice(A.shape[1], 1, replace=False)
        I = 0  # unused, to keep typechecker happy
        for _ in range(2):
            C = A[:, J]
            I, _ = maxvol_rectangular(C, max_rank_kick=1)
            R = A[I, :]
            J, _ = maxvol_rectangular(R.T, max_rank_kick=1)
        I, _ = maxvol_square(A[:, J])
        A_new = A[:, J] @ np.linalg.inv(A[I, :][:, J]) @ A[I, :]
        self.assertSimilar(A, A_new)
