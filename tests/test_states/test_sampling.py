import numpy as np
from math import sqrt
from seemps.state.sampling import sample_mps
from seemps.state import product_state, CanonicalMPS
from ..tools import SeeMPSTestCase


class TestSampling(SeeMPSTestCase):
    def make_product_state(self, amplitudes, center=None):
        state = product_state(amplitudes, 10)
        if center is not None:
            state = CanonicalMPS(state, center=center)
        return state

    def assert_constant_samples(self, amplitudes, expected_bit, center=None):
        instances = sample_mps(
            self.make_product_state(amplitudes, center=center),
            size=100,
            rng=self.rng,
        )
        self.assertTrue(np.all(instances == expected_bit))

    def test_sample_mps_sizes(self):
        mps = self.make_product_state([1.0, 0.0])

        instances = sample_mps(mps, size=30)
        self.assertIsInstance(instances, np.ndarray)
        self.assertEqual(instances.dtype, int)
        self.assertEqual(instances.shape, (30, 10))

        instances = sample_mps(mps)
        self.assertIsInstance(instances, np.ndarray)
        self.assertEqual(instances.dtype, int)
        self.assertEqual(instances.shape, (1, 10))

    def test_sample_mps_product_state_all_zeros(self):
        self.assert_constant_samples([1.0, 0.0], 0)

    def test_sample_mps_product_state_all_ones(self):
        self.assert_constant_samples([0.0, 1.0], 1)

    def test_sample_mps_Hadamard_state(self):
        mps = product_state(np.ones(2) / sqrt(2), 10)
        instances = sample_mps(mps, size=10000, rng=self.rng)
        ones = np.sum(instances)
        self.assertTrue((ones / instances.size - 0.5) < 0.01)

    def test_sample_mps_product_state_all_zeros_end_center(self):
        self.assert_constant_samples([1.0, 0.0], 0, center=-1)

    def test_sample_mps_product_state_all_ones_end_center(self):
        self.assert_constant_samples([0.0, 1.0], 1, center=-1)

    def test_sample_mps_product_state_all_ones_random_center(self):
        self.assert_constant_samples([0.0, 1.0], 1, center=4)
