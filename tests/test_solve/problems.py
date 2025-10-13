import numpy as np
import dataclasses
from typing import Sequence
from seemps.operators import MPO
from seemps.analysis.mesh import RegularInterval
from seemps.analysis.operators import id_mpo
from seemps.analysis.polynomials import mps_from_polynomial
from seemps.state import MPS
from seemps.state.factories import random_mps
from seemps.analysis.finite_differences import smooth_finite_differences_mpo


@dataclasses.dataclass
class MPOInverseProblem:
    name: str
    invertible_mpo: MPO
    right_hand_side: MPS | None = None
    tolerance: float = 1e-7

    def __post_init__(self):
        if self.right_hand_side is None:
            self.right_hand_side = random_mps(
                self.invertible_mpo.dimensions(), rng=np.random.default_rng(128842)
            )

    def get_rhs(self) -> MPS:
        out = self.right_hand_side
        if out is None:
            raise Exception()
        return out


def make_identity_problem(
    n: int, rhs: MPS | None = None, label: str | None = None
) -> MPOInverseProblem:
    if label is None:
        label = f"Identity with {n} qubits"
    return MPOInverseProblem(label, id_mpo(n), rhs)


def make_complex_problem(
    n: int, rhs: MPS | None = None, label: str | None = None
) -> MPOInverseProblem:
    if label is None:
        label = f"Complex operator and state in {n} qubits"

    if rhs is None:
        rhs = random_mps([2]*n, complex=True, rng=np.random.default_rng(0))

    return MPOInverseProblem(label, 1j * id_mpo(n), rhs)


def make_Laplacian_problem(
    n: int, rhs: np.ndarray | Sequence[float] | None = None, label: str | None = None
) -> MPOInverseProblem:
    if label is None:
        label = f"Laplacian problem with {n} qubits"
    interval = RegularInterval(0.0, 1.0, 2**n)
    x = interval.to_vector()
    dx = x[1] - x[0]
    mpo = id_mpo(n) + smooth_finite_differences_mpo(
        n, order=2, filter=3, periodic=True, dx=dx
    )
    if rhs is None:
        rhs = np.asarray([0.5, 1.0])  # 0.5 + x
    rhs_mps = mps_from_polynomial(np.asarray(rhs), interval)
    return MPOInverseProblem(label, mpo.join(), rhs_mps)


CGS_PROBLEMS = [
    make_identity_problem(2),
    make_identity_problem(3),
    make_complex_problem(3),
    make_Laplacian_problem(3, [1.0]),
]

DMRG_PROBLEMS = CGS_PROBLEMS

GMRES_PROBLEMS = CGS_PROBLEMS
