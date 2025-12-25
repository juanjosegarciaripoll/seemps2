from .tools import TestCase
import numpy as np
import seemps
import seemps.cython


class TestMPOTensorFold(TestCase):
    def test_contract_A_B(self):
        A = self.rng.normal(size=(30, 2, 2, 30))
        B = self.rng.normal(size=(10, 2, 13))

        exact_contraction = np.einsum("cjd,aijb->caidb", B, A).reshape(
            30 * 10, 2, 30 * 13
        )
        fast_contraction = seemps.operators.mpo._mpo_multiply_tensor(A, B)
        self.assertSimilar(exact_contraction, fast_contraction)


class TestTwoSiteEvolutionFold(TestCase):
    def test_contract_U_A_B(self):
        A = self.rng.normal(size=(10, 2, 15))
        B = self.rng.normal(size=(15, 3, 13))
        U = self.rng.normal(size=(2 * 3, 2 * 3))

        exact_contraction = np.einsum(
            "ijk,klm,nrjl -> inrm", A, B, U.reshape(2, 3, 2, 3)
        )
        fast_contraction = seemps.cython._contract_nrjl_ijk_klm(U, A, B)
        self.assertSimilar(exact_contraction, fast_contraction)


class TestDMRGHamiltonianContraction(TestCase):
    def test_contract_A_B(self):
        from seemps.optimization.dmrg import DMRGMatrixOperator

        L = self.rng.normal(size=(10, 5, 10))
        R = self.rng.normal(size=(13, 6, 13))
        H12 = self.rng.normal(size=(5, 3, 3, 2, 2, 6))
        v = self.rng.normal(size=(10, 3, 2, 13))

        exact_contraction = np.einsum(
            "acb,cikjld,edf,bklf->aije", L, H12, R, v
        ).reshape(-1)
        dmrg_contractor = DMRGMatrixOperator(L, H12, R)  # type: ignore
        fast_contraction = dmrg_contractor(v.reshape(-1))
        self.assertSimilar(exact_contraction, fast_contraction)
