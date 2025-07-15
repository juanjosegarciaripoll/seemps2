import numpy as np
from typing import Literal, TypeAlias
from collections.abc import Sequence
from .mpo import MPO
from ..state import Strategy, DEFAULT_STRATEGY, NO_TRUNCATION


def identity_mpo(dimensions: Sequence[int], strategy: Strategy = NO_TRUNCATION) -> MPO:
    return MPO([np.eye(d).reshape(1, d, d, 1) for d in dimensions], strategy)


IndexSelector: TypeAlias = Sequence[int]

ALL_STATES = -1


def basis_states_projector_mpo(
    selectors: list[IndexSelector],
    dimensions: Sequence[int],
    strategy: Strategy = DEFAULT_STRATEGY,
) -> MPO:
    """Return a projector over a collection of basis states."""
    tensors = []
    D = len(selectors)
    for n, d in enumerate(dimensions):
        A = np.zeros((D, d, d, D))
        tensors.append(A)
        for i, s in enumerate(selectors):
            which = s[n]
            if which == ALL_STATES:
                A[i, :, :, i] = np.eye(d)
            elif not isinstance(which, int) or which < 0 or which >= d:
                raise Exception(
                    f"Invalid state selector {which} into Hilbert space of dimension {d}"
                )
            else:
                A[i, which, which, i] = 1.0
    tensors[0] = np.sum(tensors[0], 0)[np.newaxis, :, :, :]
    tensors[-1] = np.sum(tensors[-1], -1)[:, :, :, np.newaxis]
    return MPO(tensors, strategy)
