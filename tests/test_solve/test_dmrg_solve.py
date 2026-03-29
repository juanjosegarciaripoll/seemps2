import numpy as np
from .problems import TestSolveProblems
from seemps.solve.dmrg import dmrg_solve
from seemps.analysis.operators import id_mpo
from seemps.state import CanonicalMPS, product_state
from unittest.mock import patch


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

    def test_rejects_invalid_maxiter(self):
        b = product_state([1.0, 0.0], 2)
        with self.assertRaises(Exception):
            dmrg_solve(id_mpo(2), b, maxiter=0)

    def test_rejects_unknown_solver_name(self):
        b = product_state([1.0, 0.0], 2)
        with self.assertRaises(Exception):
            dmrg_solve(id_mpo(2), b, maxiter=1, method="not-a-solver")

    def test_positive_direction_branch_uses_cg_and_default_guess(self):
        b = product_state([1.0, 0.0], 2)
        zero = b.zero_state()
        events = {}

        class FakeQuadraticForm:
            def __init__(self, A, guess, start):
                events["positive_start"] = start
                self.state = zero

            def solve(self, i, tensor2site, atol, rtol, solver):
                events["positive_solver"] = solver
                events["positive_tensor2site"] = tensor2site
                return np.zeros((1, 2, 2, 1)), 1, 0.5

            def update_2site_right(self, AB, i, strategy):
                events["used_right_update"] = True
                self.state = b

        class FakeAntilinearForm:
            def __init__(self, guess, rhs, center):
                events["positive_center"] = center

            def tensor2site(self, direction):
                return ("tensor2site", direction)

            def update_right(self):
                events["updated_right"] = True

        with (
            patch("seemps.solve.dmrg.QuadraticForm", FakeQuadraticForm),
            patch("seemps.solve.dmrg.AntilinearForm", FakeAntilinearForm),
        ):
            x, residual = dmrg_solve(id_mpo(2), b, guess=None, maxiter=2, method="cg")

        self.assertEqual(events["positive_start"], 0)
        self.assertEqual(events["positive_center"], 0)
        self.assertEqual(events["positive_tensor2site"], ("tensor2site", 1))
        self.assertIs(events["positive_solver"], __import__("scipy").sparse.linalg.cg)
        self.assertTrue(events["used_right_update"])
        self.assertTrue(events["updated_right"])
        self.assertAlmostEqual(residual, 0.0)
        self.assertSimilar(x, b)

    def test_negative_direction_branch_uses_bicg(self):
        b = product_state([1.0, 0.0], 2)
        guess = CanonicalMPS(b.zero_state(), center=-1)
        events = {}

        class FakeQuadraticForm:
            def __init__(self, A, guess, start):
                events["negative_start"] = start
                self.state = guess

            def solve(self, i, tensor2site, atol, rtol, solver):
                events["negative_solver"] = solver
                events["negative_tensor2site"] = tensor2site
                return np.zeros((1, 2, 2, 1)), 1, 0.25

            def update_2site_left(self, AB, i, strategy):
                events["used_left_update"] = True
                self.state = CanonicalMPS(b, center=-1)

        class FakeAntilinearForm:
            def __init__(self, guess, rhs, center):
                events["negative_center"] = center

            def tensor2site(self, direction):
                return ("tensor2site", direction)

            def update_left(self):
                events["updated_left"] = True

        with (
            patch("seemps.solve.dmrg.QuadraticForm", FakeQuadraticForm),
            patch("seemps.solve.dmrg.AntilinearForm", FakeAntilinearForm),
        ):
            x, residual = dmrg_solve(id_mpo(2), b, guess=guess, maxiter=2, method="bicg")

        self.assertEqual(events["negative_start"], 0)
        self.assertEqual(events["negative_center"], 0)
        self.assertEqual(events["negative_tensor2site"], ("tensor2site", -1))
        self.assertIs(events["negative_solver"], __import__("scipy").sparse.linalg.bicg)
        self.assertTrue(events["used_left_update"])
        self.assertTrue(events["updated_left"])
        self.assertAlmostEqual(residual, 0.0)
        self.assertSimilar(x, b)
