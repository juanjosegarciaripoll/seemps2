from typing import Any, Callable
from .tools import *


class TestMPOTensorFold(TestCase):
    def test_contract_A_B(self):
        A = self.rng.normal(size=(30, 2, 2, 30))
        B = self.rng.normal(size=(10, 2, 13))

        exact_contraction = np.einsum("cjd,aijb->caidb", B, A).reshape(
            30 * 10, 2, 30 * 13
        )
        fast_contraction = seemps.state.core._mpo_multiply_tensor(A, B)
        self.assertSimilar(exact_contraction, fast_contraction)


class TestTwoSiteEvolutionFold(TestCase):
    def test_contract_U_A_B(self):
        A = self.rng.normal(size=(10, 2, 15))
        B = self.rng.normal(size=(15, 3, 13))
        U = self.rng.normal(size=(2 * 3, 2 * 3))

        exact_contraction = np.einsum(
            "ijk,klm,nrjl -> inrm", A, B, U.reshape(2, 3, 2, 3)
        )
        fast_contraction = seemps.state._contractions._contract_nrjl_ijk_klm(U, A, B)
        self.assertSimilar(exact_contraction, fast_contraction)
