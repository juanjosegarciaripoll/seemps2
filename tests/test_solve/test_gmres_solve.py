import numpy as np
from .problems import GMRES_PROBLEMS
from .. import tools
from seemps.solve import gmres_solve


class TestGMRES(tools.TestCase):
    def test_basic_problems(self):
        for p in GMRES_PROBLEMS:
            with self.subTest(msg=p.name):
                x, r = gmres_solve(
                    p.invertible_mpo,
                    p.get_rhs(),
                    guess=p.get_rhs(),
                    tolerance=p.tolerance,
                )
                self.assertTrue(r < p.tolerance)
                exact_x = np.linalg.solve(
                    p.invertible_mpo.to_matrix(), p.get_rhs().to_vector()
                )
                self.assertTrue(np.linalg.norm(x.to_vector() - exact_x) < p.tolerance)
