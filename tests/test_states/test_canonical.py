import numpy as np
from math import sqrt
from seemps.state import DEFAULT_STRATEGY, CanonicalMPS, product_state
from seemps.state.canonical_mps import (
    _update_in_canonical_form_left,
    _update_in_canonical_form_right,
    _canonicalize,
)
from ..fixture_mps_states import MPSStatesFixture
from ..tools import (
    approximateIsometry,
    similar,
    almostIdentity,
)


class TestCanonicalForm(MPSStatesFixture):
    def test_local_update_canonical(self):
        #
        # We verify that _update_in_canonical_form() leaves a tensor that
        # is an approximate isometry.
        #
        def ok(Ψ, normalization=False):
            strategy = DEFAULT_STRATEGY.replace(normalize=normalization)
            for i in range(Ψ.size - 1):
                ξ = Ψ.copy()
                if "c++" in self.seemps_version:
                    _update_in_canonical_form_right(ξ, ξ[i], i, strategy)
                else:
                    _update_in_canonical_form_right(ξ._data, ξ[i], i, strategy)
                self.assertTrue(approximateIsometry(ξ[i], +1))
            for i in range(1, Ψ.size):
                ξ = Ψ.copy()
                if "c++" in self.seemps_version:
                    _update_in_canonical_form_left(ξ, ξ[i], i, strategy)
                else:
                    _update_in_canonical_form_left(ξ._data, ξ[i], i, strategy)
                self.assertTrue(approximateIsometry(ξ[i], -1))

        self.run_over_random_uniform_mps(ok)
        self.run_over_random_uniform_mps(lambda ψ: ok(ψ, normalization=True))

    def test_canonicalize(self):
        #
        # We verify _canonicalize() transforms an MPS into an equivalent one
        # that is in canonical form and represents the same state, up to
        # a reasonable tolerance.
        #
        def ok(Ψ):
            for center in range(Ψ.size):
                ξ = Ψ.copy()
                if "c++" in self.seemps_version:
                    _canonicalize(ξ, center, DEFAULT_STRATEGY)
                else:
                    _canonicalize(ξ._data, center, DEFAULT_STRATEGY)
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

        self.run_over_random_uniform_mps(ok)

    def test_canonical_mps(self):
        #
        # We verify _canonicalize() transforms an MPS into an equivalent one
        # that is in canonical form and represents the same state, up to
        # a reasonable tolerance.
        #
        def ok(mps_state):
            for center in range(mps_state.size):
                canonical_state = CanonicalMPS(mps_state, center=center)
                #
                # All sites to the left and to the right are isometries
                #
                for i in range(center):
                    self.assertTrue(approximateIsometry(canonical_state[i], +1))
                for i in range(center + 1, canonical_state.size):
                    self.assertTrue(approximateIsometry(canonical_state[i], -1))
                #
                # Both states produce the same wavefunction
                #
                self.assertTrue(
                    similar(canonical_state.to_vector(), mps_state.to_vector())
                )
                #
                # The norm is correct
                #
                self.assertAlmostEqual(
                    canonical_state.norm_squared() / mps_state.norm_squared(), 1.0
                )
                #
                # Local observables give the same
                #
                O = np.array([[0, 0], [0, 1]])
                nrm2 = canonical_state.norm_squared()
                mps_expectation = mps_state.expectation1(O, center) / nrm2
                canonical_expectation = canonical_state.expectation1(O, center) / nrm2
                self.assertAlmostEqual(canonical_expectation, mps_expectation)
                #
                # The canonical form is the same when we use the
                # corresponding negative indices of 'center'
                #
                χ = CanonicalMPS(mps_state, center=center - mps_state.size)
                for i in range(mps_state.size):
                    self.assertTrue(similar(canonical_state[i], χ[i]))

        self.run_over_random_uniform_mps(ok)

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

        self.run_over_random_uniform_mps(ok)

    def test_canonical_mps_normalization(self):
        #
        # We verify normalize_inplace() normalizes the
        # vector without really changing it.
        #
        def ok(Ψ):
            for center in range(Ψ.size):
                ξ1 = CanonicalMPS(Ψ, center=center)
                ξ2 = CanonicalMPS(Ψ, center=center)
                ξ2.normalize_inplace()
                self.assertAlmostEqual(ξ2.norm_squared(), 1.0)
                self.assertTrue(
                    similar(ξ1.to_vector() / sqrt(ξ1.norm_squared()), ξ2.to_vector())
                )

        self.run_over_random_uniform_mps(ok)

    def test_canonical_mps_copy(self):
        #
        # Copying a class does not invoke _canonicalize and does not
        # change the tensors in any way
        #
        def ok(Ψ):
            for center in range(Ψ.size):
                ψ = CanonicalMPS(Ψ, center=center)
                ξ = ψ.copy()
                self.assertEqual(ξ.size, ψ.size)
                self.assertEqual(ξ.center, ψ.center)
                for i in range(ξ.size):
                    self.assertTrue(np.all(np.equal(ξ[i], ψ[i])))

        self.run_over_random_uniform_mps(ok)

    def test_canonical_complains_if_center_out_of_bounds(self):
        mps = self.random_uniform_mps(2, 10)
        with self.assertRaises(Exception):
            CanonicalMPS(mps, center=10)
        with self.assertRaises(Exception):
            CanonicalMPS(mps, center=-11)

    def test_canonical_recenter_returns_same_object(self):
        mps = CanonicalMPS(product_state([1.0, 0.0], 10), center=0)
        self.assertIs(mps, mps.recenter(-1))

    def test_canonical_Schmidt_weights(self):
        mps = CanonicalMPS(product_state([1.0, 0.0], 10), center=0)
        self.assertSimilar(mps.Schmidt_weights(), [1.0])
        self.assertSimilar(mps.Schmidt_weights(0), [1.0])
        self.assertSimilar(mps.Schmidt_weights(-1), [1.0])

    def test_canonical_entanglement_entropy(self):
        mps = CanonicalMPS(product_state([1.0, 0.0], 10), center=0)
        self.assertAlmostEqual(mps.entanglement_entropy(), 0.0)
        self.assertAlmostEqual(mps.entanglement_entropy(0), 0.0)
        self.assertAlmostEqual(mps.entanglement_entropy(-1), 0.0)

    def test_canonical_Renyi_entropy(self):
        mps = CanonicalMPS(product_state([1.0, 0.0], 10), center=0)
        self.assertAlmostEqual(mps.Renyi_entropy(alpha=2), 0.0)
        self.assertAlmostEqual(mps.Renyi_entropy(0, alpha=2), 0.0)
        self.assertAlmostEqual(mps.Renyi_entropy(-1, alpha=2), 0.0)

    def test_canonical_from_vector(self):
        state = self.rng.normal(size=2**8)
        state /= np.linalg.norm(state)
        mps = CanonicalMPS.from_vector(state, [2] * 8)
        self.assertSimilar(state, mps.to_vector())

    def test_multiplying_by_zero_returns_zero_state(self):
        A = CanonicalMPS(self.inhomogeneous_state)
        for factor in [0, 0.0, 0.0j]:
            B = factor * A
            self.assertIsInstance(B, CanonicalMPS)
            self.assertTrue(B is not A)
            for Ai, Bi in zip(A, B):
                self.assertTrue(Ai is not Bi)
                self.assertTrue(np.all(Bi == 0))
                self.assertEqual(Bi.shape[0], 1)
                self.assertEqual(Bi.shape[2], 1)
            self.assertEqual(A.physical_dimensions(), B.physical_dimensions())
