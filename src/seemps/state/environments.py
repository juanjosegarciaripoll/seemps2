from __future__ import annotations
import numpy as np
from ..typing import Weight, Tensor3, Tensor4, MPOEnvironment
from ..cython.core import (
    _begin_environment,
    _update_left_environment,
    _update_right_environment,
    _end_environment,
    _join_environments,
    scprod,
    vdot,
)


def begin_mpo_environment() -> MPOEnvironment:
    return np.ones((1, 1, 1), dtype=np.float64)


def update_left_mpo_environment(
    rho: MPOEnvironment, A: Tensor3, O: Tensor4, B: Tensor3
) -> MPOEnvironment:
    # output = opt_einsum.contract("acb,ajd,cjie,bif->def", rho, A, O, B)
    # bif,acb->ifac
    aux = np.tensordot(B, rho, (0, 2))
    # ifac,cjie->faje
    aux = np.tensordot(aux, O, ([0, 3], [2, 0]))
    # faje,ajd-> def
    aux = np.tensordot(aux, np.conj(A), ((1, 2), (0, 1))).transpose(2, 1, 0)
    return aux


def update_right_mpo_environment(
    rho: MPOEnvironment, A: Tensor3, O: Tensor4, B: Tensor3
) -> MPOEnvironment:
    # output = opt_einsum.contract("def,ajd,cjie,bif->acb", rho, A, O, B)
    # ajd,def->ajef
    aux = np.tensordot(np.conj(A), rho, (2, 0))
    # ajef,cjie->afci
    aux = np.tensordot(aux, O, ((1, 2), (1, 3)))
    # afci,bif->acb
    aux = np.tensordot(aux, B, ((1, 3), (2, 1)))
    return aux


def end_mpo_environment(ρ: MPOEnvironment) -> Weight:
    """Extract the scalar product from the last environment."""
    return ρ[0, 0, 0]


def join_mpo_environments(left: MPOEnvironment, right: MPOEnvironment) -> Weight:
    return np.dot(left.reshape(-1), right.reshape(-1))


__all__ = [
    "_begin_environment",
    "_update_left_environment",
    "_update_right_environment",
    "_end_environment",
    "_join_environments",
    "scprod",
    "vdot",
    "begin_mpo_environment",
    "update_left_mpo_environment",
    "update_right_mpo_environment",
    "end_mpo_environment",
    "join_mpo_environments",
]
