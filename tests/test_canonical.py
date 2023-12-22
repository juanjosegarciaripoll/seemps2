from .tools import *
from seemps.state import (
    DEFAULT_STRATEGY,
    CanonicalMPS,
    product_state,
    random_uniform_mps,
)
from seemps.state.core import (
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
        def ok(psi, normalization=False):
            strategy = DEFAULT_STRATEGY.replace(normalize=normalization)
            for i in range(psi.size - 1):
                xi = psi.copy()
                _update_in_canonical_form_right(xi.data, xi[i], i, strategy)
                self.assertTrue(approximateIsometry(xi[i], +1))
            for i in range(1, psi.size):
                xi = psi.copy()
                _update_in_canonical_form_left(xi.data, xi[i], i, strategy)
                self.assertTrue(approximateIsometry(xi[i], -1))

        run_over_random_uniform_mps(ok)
        run_over_random_uniform_mps(lambda ψ: ok(ψ, normalization=True))

    def test_canonicalize(self):
        #
        # We verify _canonicalize() transforms an MPS into an equivalent one
        # that is in canonical form and represents the same state, up to
        # a reasonable tolerance.
        #
        def ok(psi):
            for center in range(psi.size):
                xi = psi.copy()
                _canonicalize(xi.data, center, DEFAULT_STRATEGY)
                #
                # All sites to the left and to the right are isometries
                #
                for i in range(center):
                    self.assertTrue(approximateIsometry(xi[i], +1))
                for i in range(center + 1, xi.size):
                    self.assertTrue(approximateIsometry(xi[i], -1))
                #
                # Both states produce the same wavefunction
                #
                self.assertSimilar(xi.to_vector(), psi.to_vector())

        run_over_random_uniform_mps(ok)

    def test_canonical_mps(self):
        #
        # We verify _canonicalize() transforms an MPS into an equivalent one
        # that is in canonical form and represents the same state, up to
        # a reasonable tolerance.
        #
        def ok(psi):
            for center in range(psi.size):
                xi = CanonicalMPS(psi, center=center)
                #
                # All sites to the left and to the right are isometries
                #
                for i in range(center):
                    self.assertTrue(approximateIsometry(xi[i], +1))
                for i in range(center + 1, xi.size):
                    self.assertTrue(approximateIsometry(xi[i], -1))
                #
                # Both states produce the same wavefunction
                #
                self.assertTrue(similar(xi.to_vector(), psi.to_vector()))
                #
                # The norm is correct
                #
                self.assertAlmostEqual(xi.norm_squared() / psi.norm_squared(), 1.0)
                #
                # Local observables give the same
                #
                O = np.array([[0, 0], [0, 1]])
                nrm2 = xi.norm_squared()
                self.assertAlmostEqual(
                    xi.expectation1(O, center) / nrm2,
                    psi.expectation1(O, center) / nrm2,
                )
                #
                # The canonical form is the same when we use the
                # corresponding negative indices of 'center'
                #
                χ = CanonicalMPS(psi, center=center - psi.size)
                for i in range(psi.size):
                    self.assertTrue(similar(xi[i], χ[i]))

        run_over_random_uniform_mps(ok)

    def test_environments(self):
        #
        # Verify that the canonical form is indeed canonical and the
        # environment is orthogonal
        #
        def ok(psi):
            for center in range(psi.size):
                xi = CanonicalMPS(psi, center=center)
                Lenv = super(CanonicalMPS, xi).left_environment(center)
                Renv = super(CanonicalMPS, xi).right_environment(center)
                self.assertTrue(almostIdentity(Lenv))
                self.assertTrue(almostIdentity(Renv))

        run_over_random_uniform_mps(ok)

    def test_canonical_mps_normalization(self):
        #
        # We verify normalize_inplace() normalizes the
        # vector without really changing it.
        #
        def ok(psi):
            for center in range(psi.size):
                xi1 = CanonicalMPS(psi, center=center)
                xi2 = CanonicalMPS(psi, center=center).normalize_inplace()
                self.assertAlmostEqual(xi2.norm_squared(), 1.0)
                self.assertTrue(
                    similar(
                        xi1.to_vector() / np.sqrt(xi1.norm_squared()), xi2.to_vector()
                    )
                )

        run_over_random_uniform_mps(ok)

    def test_canonical_mps_copy(self):
        #
        # Copying a class does not invoke _canonicalize and does not
        # change the tensors in any way
        #
        def ok(psi):
            for center in range(psi.size):
                ψ = CanonicalMPS(psi, center=center).normalize_inplace()
                xi = ψ.copy()
                self.assertEqual(xi.size, ψ.size)
                self.assertEqual(xi.center, ψ.center)
                for i in range(xi.size):
                    self.assertTrue(np.all(np.equal(xi[i], ψ[i])))

        run_over_random_uniform_mps(ok)

    def test_canonical_complains_if_center_out_of_bounds(self):
        mps = random_uniform_mps(2, 10, rng=self.rng)
        state = CanonicalMPS(mps)
        with self.assertRaises(Exception):
            CanonicalMPS(mps, center=10)
        with self.assertRaises(Exception):
            CanonicalMPS(mps, center=-11)

    def test_canonical_entanglement_entropy(self):
        mps = CanonicalMPS(product_state([1.0, 0.0], 10), center=0)
        self.assertAlmostEqual(mps.entanglement_entropy(), 0.0)
        self.assertAlmostEqual(mps.entanglement_entropy(0), 0.0)
        self.assertAlmostEqual(mps.entanglement_entropy(-1), 0.0)

    def test_canonical_from_vector(self):
        state = self.rng.normal(size=2**8)
        state /= np.linalg.norm(state)
        mps = CanonicalMPS.from_vector(state, [2] * 8)
        self.assertSimilar(state, mps.to_vector())
