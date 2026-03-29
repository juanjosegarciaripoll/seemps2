import numpy as np
from seemps.state.entropies import all_entanglement_entropies, all_Renyi_entropies
from seemps.state.factories import GHZ, product_state
from ..tools import SeeMPSTestCase


class TestEntropies(SeeMPSTestCase):
    def test_all_entanglement_entropies_for_product_state_are_zero(self):
        state = product_state([1.0, 0.0], 5)
        self.assertSimilar(all_entanglement_entropies(state), np.zeros(5))

    def test_all_entanglement_entropies_for_ghz_state_are_nontrivial(self):
        state = GHZ(4)
        self.assertSimilar(all_entanglement_entropies(state), [1.0, 1.0, 1.0, 0.0])

    def test_all_Renyi_entropies_for_ghz_state_match_expected_values(self):
        state = GHZ(4)
        expected = np.array([np.log(2.0), np.log(2.0), np.log(2.0), 0.0])
        self.assertSimilar(all_Renyi_entropies(state, alpha=2.0), expected)
