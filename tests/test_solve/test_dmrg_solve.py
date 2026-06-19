import numpy as np
from .problems import TestSolveProblems, make_complex_problem, make_identity_problem
from seemps.solve.dmrg import dmrg_solve
from seemps.state import CanonicalMPS


class TestDMRGSolve(TestSolveProblems):
    def test_basic_problems(self):
        for p in self.DMRG_PROBLEMS:
            with self.subTest(msg=p.name):
                x, r = dmrg_solve(
                    p.invertible_mpo,
                    p.get_rhs(),
                    guess=p.get_rhs(),
                    atol=p.tolerance,
                    rtol=0.0,
                )
                self.assertTrue(r < p.tolerance)
                exact_x = np.linalg.solve(
                    p.invertible_mpo.to_matrix(), p.get_rhs().to_vector()
                )
                self.assertTrue(np.linalg.norm(x.to_vector() - exact_x) < p.tolerance)

    def test_atol_is_respected_with_residuals(self):
        for p in self.DMRG_PROBLEMS:
            with self.subTest(msg=p.name):
                atol = p.tolerance
                x, r = dmrg_solve(
                    p.invertible_mpo,
                    p.get_rhs(),
                    guess=p.get_rhs(),
                    atol=atol,
                    rtol=0.0,
                    compute_residuals=True,
                )
                self.assertIsNotNone(r)
                self.assertLess(r, atol)

    def test_atol_is_respected_without_residuals(self):
        for p in self.DMRG_PROBLEMS:
            with self.subTest(msg=p.name):
                atol = p.tolerance
                x, r = dmrg_solve(
                    p.invertible_mpo,
                    p.get_rhs(),
                    guess=p.get_rhs(),
                    atol=atol,
                    rtol=0.0,
                    compute_residuals=False,
                )
                self.assertIsNone(r)
                residual = float(
                    (p.invertible_mpo @ x - p.get_rhs()).norm()
                )
                self.assertLess(residual, 10 * atol)

    def test_atol_greater_than_rtol(self):
        for p in self.DMRG_PROBLEMS:
            with self.subTest(msg=p.name):
                b_norm = p.get_rhs().norm()
                atol = 10.0 * b_norm
                rtol = 1e-5
                x, r = dmrg_solve(
                    p.invertible_mpo,
                    p.get_rhs(),
                    guess=p.get_rhs(),
                    atol=atol,
                    rtol=rtol,
                    compute_residuals=True,
                )
                self.assertLess(r, atol)

    def test_accepts_canonical_guess_with_interior_center(self):
        p = make_identity_problem(5, self.rng)
        b = p.get_rhs()
        guess = CanonicalMPS(b, center=2)

        x, r = dmrg_solve(
            p.invertible_mpo,
            b,
            guess=guess,
            atol=1e-5,
            rtol=0.0,
        )

        self.assertLessEqual(r, 1e-5)
        self.assertLess(np.linalg.norm(x.to_vector() - b.to_vector()), 1e-5)

    def test_maxiter_one_performs_one_sweep(self):
        p = make_complex_problem(3, self.rng)
        b = p.get_rhs()

        x, r = dmrg_solve(
            p.invertible_mpo,
            b,
            guess=b,
            maxiter=1,
            atol=1e-5,
            rtol=0.0,
        )

        exact_x = np.linalg.solve(p.invertible_mpo.to_matrix(), b.to_vector())
        self.assertLess(r, p.tolerance)
        self.assertLess(np.linalg.norm(x.to_vector() - exact_x), p.tolerance)
