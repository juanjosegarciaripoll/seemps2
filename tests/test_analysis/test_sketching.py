import numpy as np
from scipy.stats import norm
from ..tools import TestCase

from seemps.state import DEFAULT_STRATEGY
from seemps.analysis.mesh import RegularInterval, Mesh, mps_to_mesh_matrix
from seemps.analysis.sketching import tt_rss, BlackBoxLoadMPS


class TestTTRSS(TestCase):
    def test_1d_gaussian(self):
        n = 10
        N = 2**n
        a, b = -1.0, 1.0
        interval = RegularInterval(a, b, N)
        x = interval.to_vector()

        loc, scale = 0.0, 1.0
        f = lambda x: norm.pdf(x, loc=loc, scale=scale)  # noqa: E731
        y_vec = f(x)

        num_samples = 100
        rng = np.random.default_rng(0)
        samples = rng.normal(loc, scale, num_samples).reshape(-1, 1)

        mesh = Mesh([interval])
        map_matrix = mps_to_mesh_matrix([n])
        physical_dimensions = [2] * n
        black_box = BlackBoxLoadMPS(f, mesh, map_matrix, physical_dimensions)

        max_bonds = np.array([10] * n, dtype=int)
        mps = tt_rss(black_box, samples, max_bonds, strategy=DEFAULT_STRATEGY)
        y_mps = mps.to_vector()
        self.assertTrue(np.allclose(y_vec, y_mps, atol=1e-7))

    def test_2d_separable_gaussian(self):
        n = 8
        N = 2**n
        a, b = -1.0, 1.0
        interval = RegularInterval(a, b, N)
        x = interval.to_vector()

        f = lambda x, y: np.exp(-0.5 * (x**2 + y**2))  # noqa: E731
        X, Y = np.meshgrid(x, x, indexing="ij")
        z_vec = f(X, Y).reshape(-1)

        num_samples = 100
        rng = np.random.default_rng(0)
        samples = rng.normal(loc=0.0, scale=1.0, size=(num_samples, 2))

        mesh = Mesh([interval, interval])
        map_matrix = mps_to_mesh_matrix([n, n])
        physical_dimensions = [2] * (2 * n)
        func = lambda tensor: f(tensor[0, ...], tensor[1, ...])  # noqa: E731
        black_box = BlackBoxLoadMPS(func, mesh, map_matrix, physical_dimensions)

        max_bonds = np.array([10] * (2 * n))
        mps = tt_rss(black_box, samples, max_bonds, strategy=DEFAULT_STRATEGY)
        z_mps = mps.to_vector()
        self.assertTrue(np.allclose(z_vec, z_mps, atol=1e-7))

    def test_3d_rotated_gaussian(self):
        n = 6
        N = 2**n
        a, b = -1.0, 1.0
        interval = RegularInterval(a, b, N)
        x = interval.to_vector()

        μ = np.zeros(3)
        Σ = np.array([[1.0, 0.6, 0.3], [0.6, 1.5, 0.4], [0.3, 0.4, 1.2]])
        Σ_inv = np.linalg.inv(Σ)

        def f(x, y, z):
            X = np.stack([x, y, z], axis=-1)
            q = np.einsum("...i,ij,...j->...", X, Σ_inv, X)
            return np.exp(-0.5 * q)

        X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
        z_vec = f(X, Y, Z).reshape(-1)

        num_samples = 1000
        rng = np.random.default_rng(0)
        samples = rng.multivariate_normal(μ, Σ, size=num_samples)

        mesh = Mesh([interval] * 3)
        map_matrix = mps_to_mesh_matrix([n] * 3)
        physical_dimensions = [2] * (3 * n)
        func = lambda tensor: f(tensor[0, ...], tensor[1, ...], tensor[2, ...])  # noqa: E731
        black_box = BlackBoxLoadMPS(func, mesh, map_matrix, physical_dimensions)

        max_bonds = np.array([100] * (3 * n))
        mps = tt_rss(black_box, samples, max_bonds, strategy=DEFAULT_STRATEGY)
        z_mps = mps.to_vector()
        self.assertTrue(np.allclose(z_vec, z_mps, atol=1e-6))
