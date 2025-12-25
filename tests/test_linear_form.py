import numpy as np
from .tools import TestCase, almostIdentity, run_over_random_uniform_mps
from seemps.state import CanonicalMPS, MPS
from seemps.state.simplification import AntilinearForm
from seemps.expectation import expectation1


class TestLinearForm(TestCase):
    def test_canonical_env(self):
        #
        # When we pass two identical canonical form MPS to LinearForm, the
        # left and right environments are the identity
        #
        def ok(ψ):
            global foo
            for center in range(ψ.size):
                ϕ = CanonicalMPS(ψ, center=center).normalize_inplace()
                LF = AntilinearForm(ϕ, ϕ, center)
                for i in range(ϕ.size):
                    if i <= center:
                        self.assertSimilar(LF.L[i], ϕ.left_environment(i))
                        self.assertTrue(almostIdentity(LF.L[i], +1))
                    if i >= center:
                        self.assertSimilar(LF.R[i], ϕ.right_environment(i))
                        self.assertTrue(almostIdentity(LF.R[i], +1))

        run_over_random_uniform_mps(ok)

    def tensor1siteok(self, aϕ, O):
        for center in range(aϕ.size):
            ϕ = CanonicalMPS(aϕ, center=center).normalize_inplace()
            for n in range(ϕ.size):
                #
                # Take an MPS Φ, construct a new state ψ = O1*ϕ with a local
                # operator on the 'n-th' site
                #
                ψ = MPS(ϕ)
                ψ[n] = np.einsum("ij,ajb->aib", O, ψ[n])
                #
                # and make sure that the AntilinearForm provides the right tensor to
                # compute <ϕ|ψ> = <ϕ|O1|ϕ>
                #
                Odesired = expectation1(ϕ, O, n)
                LF = AntilinearForm(ϕ, ψ, center)
                Oestimate = np.einsum("aib,aib", ϕ[center].conj(), LF.tensor1site())
                self.assertAlmostEqual(Oestimate, Odesired)
                if n >= center:
                    self.assertTrue(almostIdentity(LF.L[center], +1))
                if n <= center:
                    self.assertTrue(almostIdentity(LF.R[center], +1))

    def test_tensor1site_product(self):
        O = np.array([[0.3, 0.2 + 1.0j], [0.2 - 1.0j, 2.0]])
        run_over_random_uniform_mps(lambda ϕ: self.tensor1siteok(ϕ, O), D=1)

    def test_tensor1site(self):
        O = np.array([[0.3, 0.2 + 1.0j], [0.2 - 1.0j, 2.0]])
        run_over_random_uniform_mps(lambda ϕ: self.tensor1siteok(ϕ, O))

    def tensor2siteok(self, aϕ, O1, O2):
        for center in range(aϕ.size):
            ϕ = CanonicalMPS(aϕ, center=center).normalize_inplace()
            for n in range(ϕ.size - 1):
                #
                # Take an MPS Φ, construct a new state ψ = O1*ϕ with a local
                # operator on the 'n-th' site
                #
                ψ = MPS(ϕ)
                ψ[n] = np.einsum("ij,ajb->aib", O1, ψ[n])
                ψ[n + 1] = np.einsum("ij,ajb->aib", O2, ψ[n + 1])
                #
                # and make sure that the AntilinearForm provides the right tensor to
                # compute <ϕ|ψ> = <ϕ|O1|ϕ>
                #
                Odesired = ϕ.expectation2(O1, O2, n)
                LF = AntilinearForm(ϕ, ψ, center)
                if center + 1 < ϕ.size:
                    D = np.einsum("ijk,klm->ijlm", ϕ[center], ϕ[center + 1])
                    Oestimate = np.einsum("aijb,aijb", D.conj(), LF.tensor2site(+1))
                    self.assertAlmostEqual(Oestimate, Odesired)
                if center > 0:
                    D = np.einsum("ijk,klm->ijlm", ϕ[center - 1], ϕ[center])
                    Oestimate = np.einsum("aijb,aijb", D.conj(), LF.tensor2site(-1))
                    self.assertAlmostEqual(Oestimate, Odesired)
                if n >= center:
                    self.assertTrue(almostIdentity(LF.L[center], +1))
                if n + 1 <= center:
                    self.assertTrue(almostIdentity(LF.R[center], +1))

    def test_tensor2site_product(self):
        O1 = np.array([[0.3, 0.2 + 1.0j], [0.2 - 1.0j, 2.0]])
        O2 = np.array([[0.34, 0.4 - 0.7j], [0.4 + 0.7j, -0.6]])
        run_over_random_uniform_mps(lambda ϕ: self.tensor2siteok(ϕ, O1, O2), D=1)

    def test_tensor2site(self):
        O1 = np.array([[0.3, 0.2 + 1.0j], [0.2 - 1.0j, 2.0]])
        O2 = np.array([[0.34, 0.4 - 0.7j], [0.4 + 0.7j, -0.6]])
        run_over_random_uniform_mps(lambda ϕ: self.tensor2siteok(ϕ, O1, O2))
