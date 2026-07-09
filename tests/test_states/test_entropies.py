import numpy as np

from seemps.state import CanonicalMPS, product_state
from seemps.state.entropies import (
    all_entanglement_entropies,
    all_Renyi_entropies,
)

from .. import tools


class TestAllEntropies(tools.SeeMPSTestCase):
    def test_product_state_has_zero_entanglement(self):
        state = product_state(np.array([1.0, 0.0]), 4)
        entropies = all_entanglement_entropies(state)
        self.assertEqual(len(entropies), 4)
        self.assertTrue(np.allclose(entropies, 0.0, atol=1e-12))

    def test_all_entanglement_entropies_match_per_site(self):
        state = self.random_uniform_mps(2, 5, D=4)
        entropies = all_entanglement_entropies(state)
        reference = [
            CanonicalMPS(state, center=i).entanglement_entropy(i)
            for i in range(len(state))
        ]
        self.assertSimilar(entropies, reference)

    def test_all_Renyi_entropies_match_per_site(self):
        alpha = 2.0
        state = self.random_uniform_mps(2, 5, D=4)
        entropies = all_Renyi_entropies(state, alpha)
        reference = [
            CanonicalMPS(state, center=i).Renyi_entropy(i, alpha)
            for i in range(len(state))
        ]
        self.assertSimilar(entropies, reference)
