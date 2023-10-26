import numpy as np
from seemps.state import DEFAULT_STRATEGY, NO_TRUNCATION, random_uniform_mps
from seemps.truncate.simplify import combine

from .tools import *


class TestCombine(TestCase):

    def test_no_truncation(self):
        d = 2
        for n in range(3,9):
            ψ1 = random_uniform_mps(d, n, D=int(2**(n/2)))
            ψ1 = ψ1 * (1/ψ1.norm())
            ψ2 = random_uniform_mps(d, n, D=int(2**(n/2)))
            ψ2 = ψ2 * (1/ψ2.norm())
            a1 = np.random.randn()
            a2 = np.random.randn()
            ψ = a1*ψ1.to_vector() + a2*ψ2.to_vector()
            φ = combine(weights=[a1,a2], states=[ψ1,ψ2], truncation=NO_TRUNCATION)
            self.assertSimilar(ψ, φ.to_vector())

    def test_tolerance(self):
        d = 2
        tolerance = 1e-10
        strategy = DEFAULT_STRATEGY.replace(simplification_tolerance=tolerance)
        for n in range(3,15):
            ψ1 = random_uniform_mps(d, n, D=int(2**(n/2)))
            ψ1 = ψ1 * (1/ψ1.norm())
            ψ2 = random_uniform_mps(d, n, D=int(2**(n/2)))
            ψ2 = ψ2 * (1/ψ2.norm())
            a1 = np.random.randn()
            a2 = np.random.randn()
            ψ = a1*ψ1.to_vector() + a2*ψ2.to_vector()
            φ = combine(weights=[a1,a2], states=[ψ1,ψ2], truncation=strategy)
            err = 2 * abs(
            1.0 - np.vdot(ψ, φ.to_vector()).real / (np.linalg.norm(ψ) * φ.norm()))
            self.assertTrue(err < tolerance)

    def test_max_bond_dimensions(self):
        d = 2
        n = 14
        for D in range(2,15):
            strategy = DEFAULT_STRATEGY.replace(max_bond_dimension=D)
            ψ1 = random_uniform_mps(d, n, D=int(2**(n/2)))
            ψ1 = ψ1 * (1/ψ1.norm())
            ψ2 = random_uniform_mps(d, n, D=int(2**(n/2)))
            ψ2 = ψ2 * (1/ψ2.norm())
            a1 = np.random.randn()
            a2 = np.random.randn()
            φ = combine(weights=[a1,a2], states=[ψ1,ψ2], truncation=strategy)
            max_D_φ = max([max(t.shape) for t in φ])
            self.assertTrue(max_D_φ <= D)