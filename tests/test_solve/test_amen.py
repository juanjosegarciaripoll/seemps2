import numpy as np
from seemps.analysis.derivatives.finite_differences import tridiagonal_mpo
from seemps.operators import MPO
from seemps.solve import amen_solve
from seemps.state import DEFAULT_STRATEGY, MPS, CanonicalMPS, random_mps
from .problems import TestSolveProblems


class TestAMEn(TestSolveProblems):
    def assertSolves(self, A: MPO, x: MPS, b: MPS, tolerance: float):
        """Check that `x` solves `A x = b`, in residual and in solution."""
        matrix, v = A.to_matrix(), b.to_vector()
        self.assertLess(np.linalg.norm(matrix @ x.to_vector() - v), tolerance)
        exact = np.linalg.solve(matrix, v)
        error = np.linalg.norm(x.to_vector() - exact)
        self.assertLess(error, tolerance * np.linalg.cond(matrix))
        # The solution must be in canonical form around its center.
        self.assertAlmostEqual(x.norm(), np.linalg.norm(x.to_vector()))

    def test_basic_problems(self):
        for p in self.DMRG_PROBLEMS:
            with self.subTest(msg=p.name):
                A, b = p.invertible_mpo, p.get_rhs()
                x = amen_solve(A, b, atol=p.tolerance, rtol=0.0, method="gmres")
                self.assertIsInstance(x, CanonicalMPS)
                self.assertSolves(A, x, b, p.tolerance)

    def test_converges_for_positive_definite_operator(self):
        A = tridiagonal_mpo(8, 2.2, -0.5, -0.5, closed=False)
        for is_complex in (False, True):
            with self.subTest(complex=is_complex):
                exact = random_mps([2] * 8, D=3, complex=is_complex, rng=self.rng)
                b = A.apply(exact, simplify=False)
                x = amen_solve(A, b, rtol=1e-8, method="cg")
                self.assertSolves(A, x, b, 1e-8 * b.norm())

    def test_converges_for_nonsymmetric_operator(self):
        A = tridiagonal_mpo(8, 2.2, -0.3, -0.8, closed=False)
        b = A.apply(random_mps([2] * 8, D=3, rng=self.rng), simplify=False)
        x = amen_solve(A, b, rtol=1e-8, method="gmres")
        self.assertSolves(A, x, b, 1e-8 * b.norm())

    def test_converges_from_guess_with_smaller_bond_dimension(self):
        A = tridiagonal_mpo(8, 2.2, -0.5, -0.5, closed=False)
        b = A.apply(random_mps([2] * 8, D=4, rng=self.rng), simplify=False)
        for name, guess in [
            ("zero", b.zero_state()),
            ("product", random_mps([2] * 8, D=1, rng=self.rng)),
        ]:
            with self.subTest(guess=name):
                self.assertEqual(guess.max_bond_dimension(), 1)
                x = amen_solve(A, b, guess, rtol=1e-8)
                self.assertGreater(x.max_bond_dimension(), 1)
                self.assertSolves(A, x, b, 1e-8 * b.norm())

    def test_converges_with_smallest_residual_rank(self):
        A = tridiagonal_mpo(8, 2.2, -0.5, -0.5, closed=False)
        b = A.apply(random_mps([2] * 8, D=4, rng=self.rng), simplify=False)
        x = amen_solve(A, b, b.zero_state(), rtol=1e-8, residual_rank=1)
        self.assertSolves(A, x, b, 1e-8 * b.norm())

    def test_converges_with_mixed_physical_dimensions(self):
        dims = [2, 3, 4, 2]
        local = [self.rng.normal(size=(d, d)) for d in dims]
        A = MPO(
            [(m @ m.T + np.eye(d)).reshape(1, d, d, 1) for m, d in zip(local, dims)]
        )
        b = random_mps(dims, D=3, rng=self.rng)
        x = amen_solve(A, b, b.zero_state(), rtol=1e-8)
        self.assertSolves(A, x, b, 1e-8 * b.norm())

    def test_single_site_and_zero_right_hand_side(self):
        A = MPO([np.diag([2.0, 3.0]).reshape(1, 2, 2, 1)])
        b = MPS([np.array([2.0, 6.0]).reshape(1, 2, 1)])
        self.assertSolves(A, amen_solve(A, b), b, 1e-8)
        A = tridiagonal_mpo(4, 2.2, -0.5, -0.5, closed=False)
        x = amen_solve(A, MPS([np.zeros((1, 2, 1))] * 4))
        self.assertTrue(np.all(x.to_vector() == 0))

    def test_does_not_modify_inputs(self):
        A = tridiagonal_mpo(6, 2.2, -0.5, -0.5, closed=False)
        b = CanonicalMPS(random_mps([2] * 6, D=3, rng=self.rng), center=0)
        guess = CanonicalMPS(random_mps([2] * 6, D=2, rng=self.rng), center=0)
        for g in (None, guess):
            saved = [t.copy() for t in (*A, *b, *guess)]
            amen_solve(A, b, g, rtol=1e-8)
            for t, original in zip((*A, *b, *guess), saved):
                self.assertTrue(np.array_equal(t, original))

    def test_warns_when_it_does_not_converge(self):
        A = tridiagonal_mpo(8, 2.2, -0.5, -0.5, closed=False)
        b = A.apply(random_mps([2] * 8, D=4, rng=self.rng), simplify=False)
        strategy = DEFAULT_STRATEGY.replace(max_bond_dimension=2)
        with self.assertWarns(RuntimeWarning):
            x = amen_solve(A, b, rtol=1e-10, maxiter=2, strategy=strategy)
        self.assertEqual(x.max_bond_dimension(), 2)
