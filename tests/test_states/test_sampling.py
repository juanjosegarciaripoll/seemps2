import numpy as np
from math import sqrt
from seemps.state.sampling import sample_mps
from seemps.state import product_state, CanonicalMPS
from ..tools import TestCase


class TestSampling(TestCase):
    def test_sample_mps_sizes(self):
        mps = product_state([1.0, 0.0], 10)

        instances = sample_mps(mps, size=30)
        self.assertIsInstance(instances, np.ndarray)
        self.assertEqual(instances.dtype, int)
        self.assertEqual(instances.shape, (30, 10))

        instances = sample_mps(mps)
        self.assertIsInstance(instances, np.ndarray)
        self.assertEqual(instances.dtype, int)
        self.assertEqual(instances.shape, (1, 10))

    def test_sample_mps_product_state_all_zeros(self):
        mps = product_state([1.0, 0.0], 10)
        instances = sample_mps(mps, size=100, rng=self.rng)
        self.assertTrue(np.all(instances == 0))

    def test_sample_mps_product_state_all_ones(self):
        mps = product_state([0.0, 1.0], 10)
        instances = sample_mps(mps, size=100, rng=self.rng)
        self.assertTrue(np.all(instances == 1))

    def test_sample_mps_Hadamard_state(self):
        mps = product_state(np.ones(2) / sqrt(2), 10)
        instances = sample_mps(mps, size=10000, rng=self.rng)
        ones = np.sum(instances)
        self.assertTrue((ones / instances.size - 0.5) < 0.01)

    def test_sample_mps_product_state_all_zeros_end_center(self):
        mps = CanonicalMPS(product_state([1.0, 0.0], 10), center=-1)
        instances = sample_mps(mps, size=100, rng=self.rng)
        self.assertTrue(np.all(instances == 0))

    def test_sample_mps_product_state_all_ones_end_center(self):
        mps = CanonicalMPS(product_state([0.0, 1.0], 10), center=-1)
        instances = sample_mps(mps, size=100, rng=self.rng)
        self.assertTrue(np.all(instances == 1))

    def test_sample_mps_product_state_all_ones_random_center(self):
        mps = CanonicalMPS(product_state([0.0, 1.0], 10), center=4)
        instances = sample_mps(mps, size=100, rng=self.rng)
        self.assertTrue(np.all(instances == 1))
