import numpy as np
from seemps.operators.quadratic import QuadraticForm
from seemps.hamiltonians import ConstantTIHamiltonian
from seemps.cython import _contract_last_and_first
from seemps.state import CanonicalMPS, NO_TRUNCATION
from seemps.operators import MPO
from ..tools import SeeMPSTestCase


class TestQuadraticForm(SeeMPSTestCase):
    def setUp(self) -> None:
        return super().setUp()

    def _H(self, size: int) -> MPO:
        Sz = np.diag([1, -1])
        Sx = np.array([[0, 1], [1, 0]])
        H = ConstantTIHamiltonian(size=size, interaction=np.kron(Sz, Sx))
        return H.to_mpo()

    def _random_canonical(
        self, size: int, D: int = 2, center: int = 0, complex: bool = False
    ) -> CanonicalMPS:
        return CanonicalMPS(
            self.random_uniform_mps(2, size, D=D, complex=complex), center=center
        )

    def _assert_adjoint_consistent(self, op):
        n = op.shape[0]
        rng = np.random.default_rng(42)
        u = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        v = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        lhs = np.vdot(u, op.matvec(v))
        rhs = np.vdot(op.rmatvec(u), v)
        self.assertAlmostEqual(lhs, rhs)

    def _dense_trace(self, op):
        n = op.shape[0]
        return sum((op @ np.eye(n)[i])[i] for i in range(n))

    def test_quadratic_form_checks_mpo_size(self):
        mpo = MPO([np.ones((1, 2, 2, 1))] * 3)
        mps = self._random_canonical(4)
        with self.assertRaises(Exception):
            QuadraticForm(mpo, mps)  # type: ignore

    def test_quadratic_form_checks_mpo_dimensions(self):
        mpo = MPO([np.ones((1, 2, 2, 1))] * 3)
        mps = CanonicalMPS(self.random_uniform_mps(3, 3), center=0)
        with self.assertRaises(Exception):
            QuadraticForm(mpo, mps)  # type: ignore

    def test_quadratic_form_checks_state_is_canonical_and_centered(self):
        H = self._H(3)
        state = self.random_uniform_mps(2, 3, D=2)
        with self.assertRaises(Exception):
            QuadraticForm(H, state, start=1)

        state_canonical = CanonicalMPS(state, center=0)
        with self.assertRaises(Exception):
            QuadraticForm(H, state_canonical, start=1)

    # Blocks shapes
    def test_blocks_shapes(self):
        H = self._H(3)
        state = self._random_canonical(3, center=0)
        Q = QuadraticForm(H, state, start=0)
        Q.move_right(1)
        state = self._random_canonical(3, center=1)
        Q = QuadraticForm(H, state, start=1)

        L, R = Q.zero_site_block(1)
        self.assertEqual(L.ndim, 3)
        self.assertEqual(R.ndim, 3)

        L, O, R = Q.one_site_block(1)
        self.assertEqual(L.ndim, 3)
        self.assertEqual(O.ndim, 4)
        self.assertEqual(R.ndim, 3)

        L, O12, R = Q.two_site_block(1)
        self.assertEqual(L.ndim, 3)
        self.assertEqual(O12.ndim, 6)
        self.assertEqual(R.ndim, 3)

    # Environment updates (move_right / move_left)
    def test_move_right_preserves_expectation(self):
        Hmpo = self._H(4)
        state = self._random_canonical(4, center=0)
        Q = QuadraticForm(Hmpo, state, start=0)
        Q.move_right(1)
        AB = Q.ket_2site(1)
        HopAB = Q.two_site_operator(1) @ AB.reshape(-1)
        self.assertAlmostEqual(np.vdot(AB, HopAB), Hmpo.expectation(state))

    def test_move_left_preserves_expectation(self):
        H = self._H(4)
        state = self._random_canonical(4, center=3)
        Q = QuadraticForm(H, state, start=3)
        Q.move_left(2)
        AB = Q.ket_2site(1)
        HopAB = Q.two_site_operator(1) @ AB.reshape(-1)
        self.assertAlmostEqual(np.vdot(AB, HopAB), H.expectation(state))

    def test_gradient_shapes(self):
        H = self._H(4)
        state = self._random_canonical(4, center=1)
        Q = QuadraticForm(H, state, start=1)

        grad1 = Q.gradient_1site()
        self.assertEqual(grad1.shape, state[1].shape)

        grad2_right = Q.gradient_2site(1)
        self.assertEqual(grad2_right.shape, Q.ket_2site(1).shape)

        grad2_left = Q.gradient_2site(-1)
        self.assertEqual(grad2_left.shape, Q.ket_2site(0).shape)

    # Left/right updates
    def test_update_2site_right(self):
        H = self._H(4)
        state = self._random_canonical(4, center=1)
        Q = QuadraticForm(H, state, start=1)
        AB = Q.ket_2site(1)
        expected_energy = H.expectation(state)

        Q.update_2site_right(AB, 1, NO_TRUNCATION)
        self.assertEqual(Q.site, 2)
        self.assertAlmostEqual(expected_energy, H.expectation(Q.state))

    def test_update_2site_left(self):
        H = self._H(4)
        state = self._random_canonical(4, center=2)
        Q = QuadraticForm(H, state, start=2)
        AB = Q.ket_2site(1)
        expected_energy = H.expectation(state)

        Q.update_2site_left(AB, 1, NO_TRUNCATION)
        self.assertEqual(Q.site, 0)
        self.assertAlmostEqual(expected_energy, H.expectation(Q.state))

    def test_update_2site_right_boundary(self):
        H = self._H(4)
        state = self._random_canonical(4, center=2)
        Q = QuadraticForm(H, state, start=2)
        AB = Q.ket_2site(2)
        Q.update_2site_right(AB, 2, NO_TRUNCATION)
        self.assertEqual(Q.site, 2)

    def test_update_2site_left_boundary(self):
        H = self._H(4)
        state = self._random_canonical(4, center=0)
        Q = QuadraticForm(H, state, start=0)
        AB = Q.ket_2site(0)
        Q.update_2site_left(AB, 0, NO_TRUNCATION)
        self.assertEqual(Q.site, 0)

    # Ket different from bra
    def test_quadratic_form_with_separate_ket(self):
        H = self._H(2)
        state = self._random_canonical(2, center=0)
        ket = self._random_canonical(2, center=0)
        Q = QuadraticForm(H, state, start=0, ket=ket)
        AB_bra = _contract_last_and_first(Q.state[0], Q.state[1])
        AB_ket = Q.ket_2site(0)
        HopAB = Q.two_site_operator(0) @ AB_ket.reshape(-1)
        expected = H.expectation(state, ket)
        self.assertAlmostEqual(np.vdot(AB_bra, HopAB), expected)

    # Zero-site operator
    def test_zero_site_operator_shape_and_matvec(self):
        H = self._H(3)
        state = self._random_canonical(3, center=0)
        Q = QuadraticForm(H, state, start=0)
        Q.move_right(1)
        state = self._random_canonical(3, center=1)
        Q = QuadraticForm(H, state, start=1)
        op = Q.zero_site_operator(1)
        n = op.shape[0]
        v = np.random.randn(n)
        self.assertEqual((op @ v).shape, (n,))

    def test_zero_site_operator_trace(self):
        H = self._H(3)
        state = self._random_canonical(3, center=0)
        Q = QuadraticForm(H, state, start=0)
        Q.move_right(1)
        state = self._random_canonical(3, center=1)
        Q = QuadraticForm(H, state, start=1)
        op = Q.zero_site_operator(1)
        self.assertAlmostEqual(op.trace(), self._dense_trace(op))

    def test_zero_site_operator_matvec(self):
        H = self._H(3)
        state = self._random_canonical(3, center=0)
        Q = QuadraticForm(H, state, start=0)
        Q.move_right(1)
        state = self._random_canonical(3, center=1)
        Q = QuadraticForm(H, state, start=1)
        L, R = Q.zero_site_block(1)
        op = Q.zero_site_operator(1)

        b, f = L.shape[2], R.shape[2]
        v = np.random.randn(b, f)

        result_fast = (op @ v.reshape(-1)).reshape(L.shape[0], R.shape[0])
        result_exact = np.einsum("acb,bf,dcf->ad", L, v, R)

        self.assertSimilar(result_fast, result_exact)

    def test_zero_site_operator_adjoint(self):
        H = self._H(4)
        state = self._random_canonical(4, center=1, complex=True)
        Q = QuadraticForm(H, state, start=1)
        self._assert_adjoint_consistent(Q.zero_site_operator(1))

    # One-site operator
    def test_one_site_operator_expectation(self):
        H = self._H(3)
        state = self._random_canonical(3, center=1)
        Q = QuadraticForm(H, state, start=1)
        A = state[1]
        HopA = Q.one_site_operator(1) @ A.reshape(-1)
        self.assertAlmostEqual(np.vdot(A, HopA), H.expectation(state))

    def test_one_site_operator_trace(self):
        Hmpo = self._H(2)
        state = self._random_canonical(2)
        Q = QuadraticForm(Hmpo, state)
        op = Q.one_site_operator(0)
        self.assertAlmostEqual(op.trace(), self._dense_trace(op))

    def test_gradient_1site_matches_operator(self):
        H = self._H(3)
        state = self._random_canonical(3, center=1)
        Q = QuadraticForm(H, state, start=1)
        grad = Q.gradient_1site()
        A = state[1]
        via_op = Q.one_site_operator(1) @ A.reshape(-1)
        self.assertSimilar(grad.reshape(-1), via_op)

    def test_one_site_operator_matvec(self):
        H = self._H(3)
        state = self._random_canonical(3, center=1)
        Q = QuadraticForm(H, state, start=1)
        L, O, R = Q.one_site_block(1)
        op = Q.one_site_operator(1)

        b, s, f = L.shape[2], O.shape[2], R.shape[2]
        v = np.random.randn(b, s, f)

        result_fast = (op @ v.reshape(-1)).reshape(L.shape[0], O.shape[1], R.shape[0])
        result_exact = np.einsum("acb,cjie,bif,def->ajd", L, O, v, R)

        self.assertSimilar(result_fast, result_exact)

    def test_one_site_operator_adjoint(self):
        H = self._H(4)
        state = self._random_canonical(4, center=1, complex=True)
        Q = QuadraticForm(H, state, start=1)
        self._assert_adjoint_consistent(Q.one_site_operator(1))

    def test_gradient_1site(self):
        H = self._H(3)
        state = self._random_canonical(3, center=1, complex=True)
        Q = QuadraticForm(H, state, start=1)

        grad_fast = Q.gradient_1site()
        L, O, R = Q.left_env[1], Q.O[1], Q.right_env[1]
        A = Q.ket[1]

        grad_exact = np.einsum("acb,cjie,bif,def->ajd", L, O, A, R)
        self.assertSimilar(grad_fast, grad_exact)

    # Two-site operator
    def test_quadratic_form_two_sites(self):
        H = self._H(2)
        state = CanonicalMPS(self.random_uniform_mps(2, 2, D=2), center=0)
        Q = QuadraticForm(H, state, start=0)
        Hop = Q.two_site_operator(0)
        AB = Q.ket_2site(0)
        HopAB = Hop @ AB.reshape(-1)
        self.assertEqual(HopAB.shape, (AB.size,))
        self.assertAlmostEqual(np.vdot(AB, HopAB), H.expectation(state))  # type: ignore
        self.assertSimilar(H.to_matrix() @ AB.reshape(-1), HopAB)

    def test_quadratic_form_start_zero(self):
        H = self._H(3)
        state = CanonicalMPS(self.random_uniform_mps(2, 3, D=2), center=0)
        Q = QuadraticForm(H, state, start=0)
        AB = Q.ket_2site(0)
        HopAB = Q.two_site_operator(0) @ AB.reshape(-1)
        self.assertAlmostEqual(np.vdot(AB, HopAB), H.expectation(state))  # type: ignore

    def test_quadratic_form_start_one(self):
        H = self._H(3)
        state = CanonicalMPS(self.random_uniform_mps(2, 3, D=2), center=1)
        Q = QuadraticForm(H, state, start=1)
        AB = Q.ket_2site(1)
        HopAB = Q.two_site_operator(1) @ AB.reshape(-1)
        self.assertAlmostEqual(np.vdot(AB, HopAB), H.expectation(state))  # type: ignore

    def test_two_site_operator_trace(self):
        H = self._H(2)
        state = self._random_canonical(2)
        Q = QuadraticForm(H, state)
        op = Q.two_site_operator(0)
        self.assertAlmostEqual(op.trace(), self._dense_trace(op))

    def test_gradient_2site_right_matches_operator(self):
        H = self._H(4)
        state = self._random_canonical(4, center=1)
        Q = QuadraticForm(H, state, start=1)
        grad = Q.gradient_2site(direction=+1)
        AB = Q.ket_2site(1)
        via_op = Q.two_site_operator(1) @ AB.reshape(-1)
        self.assertSimilar(grad.reshape(-1), via_op)

    def test_gradient_2site_left_matches_operator(self):
        H = self._H(4)
        state = self._random_canonical(4, center=2)
        Q = QuadraticForm(H, state, start=2)
        grad = Q.gradient_2site(direction=-1)
        AB = Q.ket_2site(1)
        via_op = Q.two_site_operator(1) @ AB.reshape(-1)
        self.assertSimilar(grad.reshape(-1), via_op)

    def test_two_site_operator_matvec(self):
        H = self._H(4)
        state = self._random_canonical(4, center=1)
        Q = QuadraticForm(H, state, start=1)
        L, H12, R = Q.two_site_block(1)
        op = Q.two_site_operator(1)

        b, i, k, g = L.shape[2], H12.shape[2], H12.shape[4], R.shape[2]
        v = np.random.randn(b, i, k, g)

        result_fast = (op @ v.reshape(-1)).reshape(
            L.shape[0], H12.shape[1], H12.shape[3], R.shape[0]
        )
        result_exact = np.einsum("acb,cjilkd,bikg,hdg->ajlh", L, H12, v, R)

        self.assertSimilar(result_fast, result_exact)

    def test_two_site_operator_adjoint(self):
        H = self._H(4)
        state = self._random_canonical(4, center=1, complex=True)
        Q = QuadraticForm(H, state, start=1)
        self._assert_adjoint_consistent(Q.two_site_operator(1))

    def test_gradient_2site(self):
        H = self._H(4)
        state = self._random_canonical(4, center=1, complex=True)
        Q = QuadraticForm(H, state, start=1)

        grad_fast = Q.gradient_2site(1)
        L, R = Q.left_env[1], Q.right_env[2]
        O1, O2 = Q.O[1], Q.O[2]
        A, B = Q.ket[1], Q.ket[2]

        grad_exact = np.einsum("acb,cjie,bif,elkd,fkg,hdg->ajlh", L, O1, A, O2, B, R)
        self.assertSimilar(grad_fast, grad_exact)
