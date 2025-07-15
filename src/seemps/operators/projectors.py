import numpy as np
from typing import Literal, TypeAlias
from collections.abc import Sequence
from .mpo import MPO
from ..state import Strategy, DEFAULT_STRATEGY, NO_TRUNCATION


def identity_mpo(dimensions: Sequence[int], strategy: Strategy = NO_TRUNCATION) -> MPO:
    """Return the identity operator for a composite Hilbert space of given dimensions."""
    return MPO([np.eye(d).reshape(1, d, d, 1) for d in dimensions], strategy)


IndexSelector: TypeAlias = Sequence[int]

ALL_STATES = -1


def basis_states_projector_mpo(
    selectors: list[IndexSelector],
    dimensions: Sequence[int],
    strategy: Strategy = DEFAULT_STRATEGY,
) -> MPO:
    r"""Return a projector over a collection of basis states.

    This function constructs a diagonal operator that projects onto the
    states from the basis in which this MPO is represented. It takes as
    first argument a list of `IndexSelector`, a sequence of integers
    representing specific basis states, or `ALL_STATES`, allowing any
    state along that direction.

    Arguments
    ---------
    selectors : list[IndexSelector]
        A list of `list[int]` where the magic value `ALL_STATES`
        indicates that all basis states for the specific component
        are allowed.
    dimensions : list[int]
        List of physical dimensions of the MPO
    strategy : Strategy, default = `DEFAULT_STRATEGY`
        Truncation strategy when applying this MPO.

    Returns
    -------
    projector : MPO
        The diagonal operator in matrix product form, unsimplified.

    Examples
    --------
    Projector onto the :math:`|0,0,0\rangle` state.
    >>> P000 = basis_states_projector_mpo([0,0,0], [2,2,2])
    Composite operator :math:`I\otimes|0\rangle\langle0|\otimes I` state.
    >>> P000 = basis_states_projector_mpo([ALL_STATES,0,ALL_STATES], [2,2,2])
    """
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
