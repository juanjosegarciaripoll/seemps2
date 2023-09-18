from .tools import *
from seemps.state import (
    DEFAULT_STRATEGY,
    CanonicalMPS,
)
from seemps.state.canonical_mps import (
    _update_in_canonical_form_left,
    _update_in_canonical_form_right,
    _canonicalize,
)


class TestCanonicalForm(TestCase):
    def test_local_update_canonical(self):
        #
        # We verify that _update_in_canonical_form() leaves a tensor that
        # is an approximate isometry.
        #
        def ok(Ψ, normalization=False):
            strategy = DEFAULT_STRATEGY.replace(normalize=normalization)
            for i in range(Ψ.size - 1):
                ξ = Ψ.copy()
                _update_in_canonical_form_right(ξ, ξ[i], i, strategy)
                self.assertTrue(approximateIsometry(ξ[i], +1))
            for i in range(1, Ψ.size):
                ξ = Ψ.copy()
                _update_in_canonical_form_left(ξ, ξ[i], i, strategy)
                self.assertTrue(approximateIsometry(ξ[i], -1))

        run_over_random_uniform_mps(ok)
        run_over_random_uniform_mps(lambda ψ: ok(ψ, normalization=True))

    def test_canonicalize(self):
        #
        # We verify _canonicalize() transforms an MPS into an equivalent one
        # that is in canonical form and represents the same state, up to
        # a reasonable tolerance.
        #
        def ok(Ψ):
            for center in range(Ψ.size):
                ξ = Ψ.copy()
                _canonicalize(ξ, center, DEFAULT_STRATEGY)
                #
                # All sites to the left and to the right are isometries
                #
                for i in range(center):
                    self.assertTrue(approximateIsometry(ξ[i], +1))
                for i in range(center + 1, ξ.size):
                    self.assertTrue(approximateIsometry(ξ[i], -1))
                #
                # Both states produce the same wavefunction
                #
                self.assertSimilar(ξ.to_vector(), Ψ.to_vector())

        run_over_random_uniform_mps(ok)

    def test_canonical_mps(self):
        #
        # We verify _canonicalize() transforms an MPS into an equivalent one
        # that is in canonical form and represents the same state, up to
        # a reasonable tolerance.
        #
        def ok(Ψ):
            for center in range(Ψ.size):
                ξ = CanonicalMPS(Ψ, center=center)
                #
                # All sites to the left and to the right are isometries
                #
                for i in range(center):
                    self.assertTrue(approximateIsometry(ξ[i], +1))
                for i in range(center + 1, ξ.size):
                    self.assertTrue(approximateIsometry(ξ[i], -1))
                #
                # Both states produce the same wavefunction
                #
                self.assertTrue(similar(ξ.to_vector(), Ψ.to_vector()))
                #
                # The norm is correct
                #
                self.assertAlmostEqual(ξ.norm_squared() / Ψ.norm_squared(), 1.0)
                #
                # Local observables give the same
                #
                O = np.array([[0, 0], [0, 1]])
                nrm2 = ξ.norm_squared()
                self.assertAlmostEqual(
                    ξ.expectation1(O, center) / nrm2, Ψ.expectation1(O, center) / nrm2
                )
                #
                # The canonical form is the same when we use the
                # corresponding negative indices of 'center'
                #
                χ = CanonicalMPS(Ψ, center=center - Ψ.size)
                for i in range(Ψ.size):
                    self.assertTrue(similar(ξ[i], χ[i]))

        run_over_random_uniform_mps(ok)

    def test_environments(self):
        #
        # Verify that the canonical form is indeed canonical and the
        # environment is orthogonal
        #
        def ok(Ψ):
            for center in range(Ψ.size):
                ξ = CanonicalMPS(Ψ, center=center)
                Lenv = super(CanonicalMPS, ξ).left_environment(center)
                Renv = super(CanonicalMPS, ξ).left_environment(center)
                self.assertTrue(almostIdentity(Lenv))
                self.assertTrue(almostIdentity(Renv))

        run_over_random_uniform_mps(ok)

    def test_canonical_mps_normalization(self):
        #
        # We verify normalize_inplace() normalizes the
        # vector without really changing it.
        #
        def ok(Ψ):
            for center in range(Ψ.size):
                ξ1 = CanonicalMPS(Ψ, center=center)
                ξ2 = CanonicalMPS(Ψ, center=center).normalize_inplace()
                self.assertAlmostEqual(ξ2.norm_squared(), 1.0)
                self.assertTrue(
                    similar(ξ1.to_vector() / np.sqrt(ξ1.norm_squared()), ξ2.to_vector())
                )

        run_over_random_uniform_mps(ok)

    def test_canonical_mps_copy(self):
        #
        # Copying a class does not invoke _canonicalize and does not
        # change the tensors in any way
        #
        def ok(Ψ):
            for center in range(Ψ.size):
                ψ = CanonicalMPS(Ψ, center=center).normalize_inplace()
                ξ = ψ.copy()
                self.assertEqual(ξ.size, ψ.size)
                self.assertEqual(ξ.center, ψ.center)
                for i in range(ξ.size):
                    self.assertTrue(np.all(np.equal(ξ[i], ψ[i])))

        run_over_random_uniform_mps(ok)
