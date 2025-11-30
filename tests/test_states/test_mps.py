import numpy as np
from seemps.state import MPS, MPSSum
from ..fixture_mps_states import MPSStatesFixture
from ..tools import (
    contain_same_objects,
    contain_different_objects,
    random_uniform_mps,
    similar,
)


class TestMPS(MPSStatesFixture):
    def test_initial_data_is_copied(self):
        data = [np.zeros((1, 2, 1))] * 10
        A = MPS(data)
        self.assertFalse(A._data is data)
        self.assertEqual(A._data, data)

    def test_initial_copy_is_shallow(self):
        A = MPS(self.product_state.copy(), 0.1)
        B = A.copy()
        self.assertTrue(A._data is not B._data)
        self.assertEqual(A._error, B._error)
        self.assertTrue(contain_same_objects(A, B))
        A[::] = [self.other_tensor] * len(A)
        self.assertTrue(contain_different_objects(A, B))

    def test_total_dimension_is_product_of_physical_dimensions(self):
        A = MPS(self.inhomogeneous_state)
        self.assertEqual(A.dimension(), self.inhomogeneous_state_dimension)

    def test_to_vector_creates_correct_wavefunction(self):
        A = MPS(self.inhomogeneous_state)
        ψ = A.to_vector()
        self.assertEqual(ψ.shape, (self.inhomogeneous_state_dimension,))
        self.assertTrue(similar(ψ, self.inhomogeneous_state_wavefunction))

    def test_from_vector_recreates_product_states(self):
        A = MPS.from_vector(
            self.inhomogeneous_state_wavefunction, [2, 3, 4], normalize=False
        )
        self.assertTrue(
            all(a.shape == b.shape for a, b in zip(A, self.inhomogeneous_state))
        )
        self.assertSimilar(A.to_vector(), self.inhomogeneous_state_wavefunction)

    def test_from_tensor_recreates_product_states(self):
        A = MPS.from_tensor(
            self.inhomogeneous_state_wavefunction.reshape(2, 3, 4),
            normalize=False,
        )
        self.assertTrue(
            all(a.shape == b.shape for a, b in zip(A, self.inhomogeneous_state))
        )
        self.assertSimilar(A.to_vector(), self.inhomogeneous_state_wavefunction)

    def test_mps_bond_dimensions_returns_first_dimension(self):
        shapes = [(1, 2, 2), (2, 2, 3), (3, 2, 1), (1, 2, 2), (2, 2, 1)]
        mps = MPS([np.ones(shape) for shape in shapes])
        self.assertEqual(len(mps.bond_dimensions()), mps.size + 1)
        self.assertEqual(mps.bond_dimensions(), [1, 2, 3, 1, 2, 1])


class TestMPSOperations(MPSStatesFixture):
    def test_norm2_is_deprecated(self):
        with self.assertWarns(DeprecationWarning):
            MPS(self.inhomogeneous_state).norm2()

    def test_norm_returns_real_nonnegative_values(self):
        complex_mps = MPS([-1j * x for x in self.inhomogeneous_state])
        complex_mps_norm = complex_mps.norm()
        self.assertTrue(complex_mps_norm > 0)
        self.assertTrue(isinstance(complex_mps_norm, float))

    def test_norm_returns_wavefunction_norm(self):
        self.assertAlmostEqual(
            MPS(self.inhomogeneous_state).norm(),
            np.linalg.norm(self.inhomogeneous_state_wavefunction),
        )

    def test_norm_squared_returns_wavefunction_norm_squared(self):
        self.assertAlmostEqual(
            MPS(self.inhomogeneous_state).norm_squared(),
            np.linalg.norm(self.inhomogeneous_state_wavefunction) ** 2,
        )

    def test_adding_mps_creates_mps_list(self):
        A = MPS(self.inhomogeneous_state)
        B = MPS(self.inhomogeneous_state)
        C = A + B
        self.assertTrue(isinstance(C, MPSSum))

    def test_adding_mps_with_non_mps_raises_error(self):
        A = MPS(self.inhomogeneous_state)
        with self.assertRaises(TypeError):
            A = A + 2.0  # type: ignore
        with self.assertRaises(TypeError):
            A = 2.0 + A  # type: ignore

    def test_subtracting_mps_creates_mps_list(self):
        A = MPS(self.inhomogeneous_state)
        B = MPS(self.inhomogeneous_state)
        C = A - B
        self.assertTrue(isinstance(C, MPSSum))

    def test_subtracting_mps_and_non_mps_raises_error(self):
        A = MPS(self.inhomogeneous_state)
        with self.assertRaises(TypeError):
            A = A - 2.0  # type: ignore
        with self.assertRaises(TypeError):
            A = 2.0 - A  # type: ignore

    def test_scaling_mps_creates_new_object(self):
        A = MPS(self.inhomogeneous_state)
        B = 3.0 * A
        self.assertTrue(B is not A)
        self.assertTrue(contain_different_objects(B[0], A[0]))

    def test_multiplying_by_zero_returns_zero_state(self):
        A = MPS(self.inhomogeneous_state)
        for factor in [0, 0.0, 0.0j]:
            B = factor * A
            self.assertIsInstance(B, MPS)
            self.assertTrue(B is not A)
            for Ai, Bi in zip(A, B):
                self.assertTrue(Ai is not Bi)
                self.assertTrue(np.all(Bi == 0))
                self.assertEqual(Bi.shape[0], 1)
                self.assertEqual(Bi.shape[2], 1)
            self.assertEqual(A.physical_dimensions(), B.physical_dimensions())

    def test_multiplying_mps_by_non_scalar_raises_exception(self):
        A = MPS(self.inhomogeneous_state)
        with self.assertRaises(TypeError):
            B = A * np.array([1.0])  # type: ignore # noqa: F841
        with self.assertRaises(TypeError):
            B = A * np.zeros((3, 3))  # type: ignore # noqa: F841
        with self.assertRaises(TypeError):
            B = np.array([1.0]) * A  # type: ignore # noqa: F841
        with self.assertRaises(TypeError):
            B = np.zeros((3, 3)) * A  # type: ignore # noqa: F841

    def test_scaled_mps_produces_scaled_wavefunction(self):
        factor = 1.0 + 3.0j
        A = MPS(self.inhomogeneous_state)
        self.assertTrue(similar(factor * A.to_vector(), (factor * A).to_vector()))
        factor = 1.0 + 3.0j
        A = MPS(self.inhomogeneous_state)
        self.assertTrue(similar(factor * A.to_vector(), (A * factor).to_vector()))

    def test_scaled_mps_produces_scaled_norm(self):
        factor = 1.0 + 3.0j
        A = MPS(self.inhomogeneous_state)
        self.assertAlmostEqual(abs(factor) * A.norm(), (factor * A).norm())

    def test_multiplying_two_mps_produces_product_wavefunction(self):
        A = MPS(self.inhomogeneous_state)
        self.assertSimilar(A.to_vector() * A.to_vector(), (A * A).to_vector())
        with self.assertRaises(Exception):
            random_uniform_mps(2, 3) * random_uniform_mps(3, 3)  # type: ignore

    def test_mps_complex_conjugate(self):
        A = MPS(self.inhomogeneous_state)
        self.assertSimilar(A.to_vector().conj(), A.conj().to_vector())

    def test_mps_extend(self):
        mps = random_uniform_mps(2, 5, D=5, truncate=False)
        new_mps = mps.extend(7, sites=[0, 2, 4, 5, 6], dimensions=3)
        self.assertTrue(mps[0] is new_mps[0])
        self.assertEqual(new_mps[1].shape, (5, 3, 5))
        self.assertTrue(mps[1] is new_mps[2])
        self.assertEqual(new_mps[3].shape, (5, 3, 5))
        self.assertTrue(mps[2] is new_mps[4])
        self.assertTrue(mps[3] is new_mps[5])
        self.assertTrue(mps[4] is new_mps[6])

    def test_mps_extend_accepts_dimensions_list_with_proper_size(self):
        mps = random_uniform_mps(2, 5, D=5, truncate=False)
        new_mps = mps.extend(7, sites=[0, 2, 4, 5, 6], dimensions=[5, 4])
        self.assertEqual(new_mps.physical_dimensions(), [2, 5, 2, 4, 2, 2, 2])
        with self.assertRaises(Exception):
            mps.extend(7, sites=[0, 2, 4, 5, 6], dimensions=[5])
        with self.assertRaises(Exception):
            mps.extend(7, sites=[0, 2, 4, 5, 6], dimensions=[5, 6, 8])

    def test_mps_extend_cannot_shrink_mps(self):
        mps = random_uniform_mps(2, 5, D=5, truncate=False)
        with self.assertRaises(Exception):
            mps.extend(3)
