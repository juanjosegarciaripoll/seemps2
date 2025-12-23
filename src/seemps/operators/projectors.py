import numpy as np
from typing import TypeAlias
from collections.abc import Sequence
from .mpo import MPO
from ..state import Strategy, DEFAULT_STRATEGY, NO_TRUNCATION


def identity_mpo(dimensions: Sequence[int], strategy: Strategy = NO_TRUNCATION) -> MPO:
    """Return the identity operator for a composite Hilbert space of given dimensions."""
    return MPO([np.eye(d).reshape(1, d, d, 1) for d in dimensions], strategy)


IndexSelector: TypeAlias = Sequence[int | tuple[int, ...]]
"""Sequence of basis states.

This type is a sequence of objects that select basis states. Admitted
values in the sequence include
- An `int` representing one basis state
- A tuple of one or more `int`, representing multiple basis states
- A `ALL_STATES` object representing all basis states.
"""

ALL_STATES: tuple = tuple()


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

    Parameters
    ----------
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
    ok = True
    for n, d in enumerate(dimensions):
        A = np.zeros((D, d, d, D))
        tensors.append(A)
        for i, s in enumerate(selectors):
            e = np.zeros(d)
            w = s[n]
            if isinstance(w, int):
                if w < 0 or w >= d:
                    ok = False
                else:
                    e[w] = 1.0
            elif not isinstance(w, tuple):
                ok = False
            elif w is ALL_STATES:
                e[:] = 1.0
            else:
                for wi in w:
                    if wi < 0 or wi >= d:
                        ok = False
                        break
                    e[wi] = 1.0
            if not ok:
                raise Exception(f"Invalid basis state selector {w}")
            A[i, :, :, i] = np.diag(e)
    tensors[0] = np.sum(tensors[0], 0)[np.newaxis, :, :, :]
    tensors[-1] = np.sum(tensors[-1], -1)[:, :, :, np.newaxis]
    return MPO(tensors, strategy)
