import numpy as np
import dataclasses
from typing import Sequence
from seemps.operators import MPO
from seemps.analysis.mesh import QuantizedInterval
from seemps.analysis.operators import id_mpo
from seemps.analysis.polynomials import mps_from_polynomial
from seemps.state import MPS
from seemps.state.factories import random_mps
from seemps.analysis.derivatives import finite_differences_mpo
from ..tools import SeeMPSTestCase


@dataclasses.dataclass
class MPOInverseProblem:
    name: str
    invertible_mpo: MPO
    right_hand_side: MPS
    tolerance: float = 1e-7

    def get_rhs(self) -> MPS:
        out = self.right_hand_side
        if out is None:
            raise Exception()
        return out


def make_identity_problem(
    n: int, rng: np.random.Generator, rhs: MPS | None = None, label: str | None = None
) -> MPOInverseProblem:
    if label is None:
        label = f"Identity with {n} qubits"
    if rhs is None:
        rhs = random_mps([2] * n, complex=False, rng=rng)
    return MPOInverseProblem(label, id_mpo(n), rhs)


def make_complex_problem(
    n: int, rng: np.random.Generator, rhs: MPS | None = None, label: str | None = None
) -> MPOInverseProblem:
    if label is None:
        label = f"Complex operator and state in {n} qubits"
    if rhs is None:
        rhs = random_mps([2] * n, complex=True, rng=rng)
    return MPOInverseProblem(label, 1j * id_mpo(n), rhs)


def make_Laplacian_problem(
    n: int,
    rhs: np.ndarray | Sequence[float] | None = None,
    label: str | None = None,
) -> MPOInverseProblem:
    if label is None:
        label = f"Laplacian problem with {n} qubits"
    interval = QuantizedInterval(0.0, 1.0, qubits=n)
    mpo = id_mpo(n) + finite_differences_mpo(
        order=2, filter=3, interval=interval, periodic=True
    )
    if rhs is None:
        rhs = np.asarray([0.5, 1.0])  # 0.5 + x
    rhs_mps = mps_from_polynomial(np.asarray(rhs), interval)
    return MPOInverseProblem(label, mpo.join(), rhs_mps)


class TestSolveProblems(SeeMPSTestCase):
    def setUp(self):
        super().setUp()
        self.CGS_PROBLEMS = [
            make_identity_problem(2, self.rng),
            make_identity_problem(3, self.rng),
            make_complex_problem(3, self.rng),
            make_Laplacian_problem(3, [1.0]),
        ]
        self.DMRG_PROBLEMS = self.CGS_PROBLEMS
        self.GMRES_PROBLEMS = self.CGS_PROBLEMS
