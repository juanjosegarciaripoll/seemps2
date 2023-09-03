import numpy as np
from ..typing import *
from ..state.core import MAX_BOND_DIMENSION
from ..state import (
    DEFAULT_TOLERANCE,
    Truncation,
    Strategy,
    MPS,
    CanonicalMPS,
)
from ..tools import log
from ..state._contractions import _contract_last_and_first
from ..expectation import (
    begin_environment,
    update_right_environment,
    update_left_environment,
    scprod,
)


class AntilinearForm:
    """Representation of a scalar product :math:`\\langle\\xi|\\psi\\rangle`
    with capabilities for differentiation.

    This class is an object that formally implements
    :math:`\\langle\\xi|\\psi\\rangle` with an argument :math:`\\xi`
    that may change and be updated over time. In particular, given a site `n`
    it can construct the tensor `L` such that the contraction between `L`
    and the `n`-th tensor from MPS :math:`\\xi` is the result of the linear form.

    Parameters
    ----------
    bra, ket: MPS
        MPS states :math:`\\xi` and :math:`\\psi` above.
    center: int, default = 0
        Position at which the `L` tensor is precomputed.
    """

    def __init__(self, bra, ket, center=0):
        assert bra.size == ket.size
        size = bra.size
        ρ = begin_environment()
        R = [ρ] * size
        for i in range(size - 1, center, -1):
            R[i - 1] = ρ = update_right_environment(bra[i], ket[i], ρ)

        ρ = begin_environment()
        L = [ρ] * size
        for i in range(0, center):
            L[i + 1] = ρ = update_left_environment(bra[i], ket[i], ρ)

        self.bra = bra
        self.ket = ket
        self.size = size
        self.R = R
        self.L = L
        self.center = center

    def tensor1site(self) -> Tensor3:
        """Return the tensor representing the AntilinearForm at the
        `self.center` site."""
        center = self.center
        L = self.L[center]
        R = self.R[center]
        C = self.ket[center]
        return np.einsum("li,ijk,kn->ljn", L, C, R)

    def tensor2site(self, direction: int) -> Tensor4:
        """Return the tensor that represents the LinearForm using 'center'
        and another site.

        Parameters
        ----------
        direction : {+1, -1}
            If positive, the tensor acts on `self.center` and `self.center+1`
            Otherwise on `self.center` and `self.center-1`.

        Returns
        -------
        Tensor4
            Four-legged tensor representing the antilinear form.
        """
        if direction > 0:
            i = self.center
            j = i + 1
        else:
            j = self.center
            i = j - 1
        L = self.L[i]
        A = self.ket[i]
        B = self.ket[j]
        R = self.R[j]
        # np.einsum("li,ijk->ljk", L, A)
        LA = _contract_last_and_first(L, A)
        # np.einsum("kmn,no->kmo", B, R)
        BR = np.matmul(B, R)
        # np.einsum("ljk,kmo->ljmo", LA, BR)
        return _contract_last_and_first(LA, BR)

    def update(self, direction: int) -> None:
        """Notify that the `bra` state has been changed, and that we move to
        `self.center + direction`.

        We have updated 'mps' (the bra), which is now centered on a different point.
        We have to recompute the environments.

        Parameters
        ----------
        direction : { +1 , -1 }
        """
        prev = self.center
        if direction > 0:
            nxt = prev + 1
            if nxt < self.size:
                self.L[nxt] = update_left_environment(
                    self.bra[prev], self.ket[prev], self.L[prev]
                )
                self.center = nxt
        else:
            nxt = prev - 1
            if nxt >= 0:
                self.R[nxt] = update_right_environment(
                    self.bra[prev], self.ket[prev], self.R[prev]
                )
                self.center = nxt


# TODO: We have to rationalize all this about directions. The user should
# not really care about it and we can guess the direction from the canonical
# form of either the guess or the state.
def simplify(
    state: MPS,
    maxsweeps: int = 4,
    direction: int = +1,
    tolerance: float = DEFAULT_TOLERANCE,
    normalize: bool = True,
    max_bond_dimension: int = MAX_BOND_DIMENSION,
) -> tuple[MPS, float, int]:
    """Simplify an MPS state transforming it into another one with a smaller bond
    dimension, sweeping until convergence is achieved.

    Parameters
    ----------
    state : MPS
        State to approximate
    direction : { +1, -1 }
        Direction of the first sweep
    maxsweeps : int
        Maximum number of sweeps to run
    tolerance : float
        Relative tolerance when splitting the tensors. Defaults to
        `DEFAULT_TOLERANCE`
    max_bond_dimension : int
        Maximum bond dimension. Defaults to `MAX_BOND_DIMENSION`

    Returns
    -------
    CanonicalMPS
        Approximation :math:`\\xi` to the state.
    float
        Total error in norm-2 squared :math:`\\Vert\\xi-\\psi\\Vert^2`.
    int
        Direction that the next sweep would be.
    """
    size = state.size
    start = 0 if direction > 0 else size - 1

    truncation = Strategy(
        method=Truncation.RELATIVE_NORM_SQUARED_ERROR,
        tolerance=tolerance,
        max_bond_dimension=max_bond_dimension,
        normalize=normalize,
    )
    mps = CanonicalMPS(state, center=start, strategy=truncation)
    if normalize:
        mps.normalize_inplace()
    if max_bond_dimension == 0 and tolerance <= 0:
        return mps, 0.0, -direction

    form = AntilinearForm(mps, state, center=start)
    norm_state_sqr = scprod(state, state).real
    base_error = state.error()
    err = 1.0
    log(
        f"SIMPLIFY state with |state|={norm_state_sqr**0.5} for {maxsweeps} sweeps, with tolerance {tolerance}."
    )
    for sweep in range(maxsweeps):
        if direction > 0:
            for n in range(0, size - 1):
                mps.update_2site_right(form.tensor2site(direction), n, truncation)
                form.update(direction)
            last = size - 1
        else:
            for n in reversed(range(0, size - 1)):
                mps.update_2site_left(form.tensor2site(direction), n, truncation)
                form.update(direction)
            last = 0
        #
        # We estimate the error
        #
        last = mps.center
        B = mps[last]
        norm_mps_sqr = np.vdot(B, B).real
        if normalize:
            mps[last] = B = B / norm_mps_sqr
            norm_mps_sqr = 1.0
        mps_state_scprod = np.vdot(B, form.tensor1site())
        old_err = err
        err = 2 * abs(
            1.0 - mps_state_scprod.real / np.sqrt(norm_mps_sqr * norm_state_sqr)
        )
        log(
            f"sweep={sweep}, rel.err.={err}, old err.={old_err}, |mps|={norm_mps_sqr**0.5}"
        )
        if err < tolerance or err > old_err:
            log("Stopping, as tolerance reached")
            break
        direction = -direction
    mps._error = 0.0
    mps.update_error(base_error)
    mps.update_error(err)
    # TODO: Inconsistency between simplify() and combine(). Only the former
    # returns a direction.
    return mps, err, direction
