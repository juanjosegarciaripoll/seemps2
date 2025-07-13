import numpy as np
from .problems import DMRG_PROBLEMS
from .. import tools
from seemps.optimization.dmrg import dmrg_solve


class TestDMRGSolve(tools.TestCase):
    def test_basic_problems(self):
        for p in DMRG_PROBLEMS:
            print(f"Running {p.name}", flush=True)
            with self.subTest(msg=p.name):
                x, r = dmrg_solve(
                    p.invertible_mpo,
                    p.get_rhs(),
                    guess=p.get_rhs(),
                    tol=p.tolerance,
                )
                self.assertTrue(r < p.tolerance)
                exact_x = np.linalg.solve(
                    p.invertible_mpo.to_matrix(), p.get_rhs().to_vector()
                )
                self.assertTrue(np.linalg.norm(x.to_vector() - exact_x) < p.tolerance)
