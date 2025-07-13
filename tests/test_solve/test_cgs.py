import numpy as np
from .problems import CGS_PROBLEMS
from .. import tools
from seemps.cgs import cgs


class TestCGS(tools.TestCase):
    def test_basic_problems(self):
        for p in CGS_PROBLEMS:
            print(f"Running {p.name}", flush=True)
            with self.subTest(msg=p.name):
                x, r = cgs(
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
