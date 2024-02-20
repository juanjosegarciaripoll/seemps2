from __future__ import annotations
import numpy as np
from .mps import MPS
from .canonical_mps import CanonicalMPS
from ..typing import Vector


def all_entanglement_entropies(state: MPS) -> Vector:
    cstate = CanonicalMPS(state, center=0)
    L = len(cstate)
    entropies = np.empty(L)
    for i in range(L):
        cstate = CanonicalMPS(cstate, center=i)
        entropies[i] = cstate.entanglement_entropy(i)
    return entropies


def all_Renyi_entropies(state: MPS, alpha: float) -> Vector:
    cstate = CanonicalMPS(state, center=0)
    L = len(cstate)
    entropies = np.empty(L)
    for i in range(L):
        cstate = CanonicalMPS(cstate, center=i)
        entropies[i] = cstate.Renyi_entropy(i, alpha)
    return entropies
