from __future__ import annotations

import numpy as np

from ..state import (DEFAULT_TOLERANCE, MAX_BOND_DIMENSION, MPS, CanonicalMPS,
                     MPSSum, Strategy, Truncation)
from ..state.environments import scprod
from ..tools import log
from ..typing import *
from .antilinear import AntilinearForm

# TODO: We have to rationalize all this about directions. The user should
# not really care about it and we can guess the direction from the canonical
# form of either the guess or the state.

SIMPLIFICATION_STRATEGY = Strategy(
        method=Truncation.RELATIVE_NORM_SQUARED_ERROR,
        tolerance=DEFAULT_TOLERANCE,
        max_bond_dimension=MAX_BOND_DIMENSION,
        normalize=True,
        max_sweeps=4
    )

def simplify(
    state: Union[MPS, MPSSum], 
    truncation: Strategy = SIMPLIFICATION_STRATEGY, 
    direction: int = +1
    
) -> MPS:
    """Simplify an MPS state transforming it into another one with a smaller bond
    dimension, sweeping until convergence is achieved.

    Parameters
    ----------
    state : MPS | MPSSum
        State to approximate.
    truncation : Strategy
        Truncation strategy. Defaults to `SIMPLIFICATION_STRATEGY`.
    direction : { +1, -1 }
       Initial direction for the sweeping algorithm.

    Returns
    -------
    CanonicalMPS
        Approximation :math:`\\xi` to the state.
    """
    if isinstance(state, MPSSum):
        return combine(
            state.weights,
            state.states,
            truncation=truncation,
            direction=direction,
        )

    size = state.size
    start = 0 if direction > 0 else size - 1
    normalize= truncation.get_normalize_flag()
    maxsweeps = truncation.get_max_sweeps()
    tolerance = truncation.get_tolerance()
    simplification_tolerance = truncation.get_simplification_tolerance()
    max_bond_dimension = truncation.get_max_bond_dimension()
    mps = CanonicalMPS(state, center=start, strategy=truncation)
    if normalize:
        mps.normalize_inplace()
    if not truncation.get_simplification_method():
        return mps
    if max_bond_dimension == 0 and simplification_tolerance <= 0:
        return mps

    form = AntilinearForm(mps, state, center=start)
    norm_state_sqr = scprod(state, state).real
    base_error = state.error()
    err = 1.0
    log(
        f"SIMPLIFY state with |state|={norm_state_sqr**0.5} for {maxsweeps} sweeps, with tolerance {simplification_tolerance}."
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
        if err < simplification_tolerance or err > old_err:
            log("Stopping, as tolerance reached")
            break
        direction = -direction
    mps._error = 0.0
    mps.update_error(base_error)
    mps.update_error(err)
    # TODO: Inconsistency between simplify() and combine(). Only the former
    # returns a direction.
    return mps


def multi_norm_squared(weights: list[Weight], states: list[MPS]) -> float:
    """Compute the norm-squared of the linear combination of weights and
    states."""
    c: float = 0.0
    for i, wi in enumerate(weights):
        for j in range(i):
            c += 2 * (wi.conjugate() * weights[j] * scprod(states[i], states[j])).real
        c += np.abs(wi) ** 2 * scprod(states[i], states[i]).real
    return c


def guess_combine_state(weights: list[Weight], states: list[MPS]) -> MPS:
    """Make an educated guess that ensures convergence of the :func:`combine`
    algorithm."""

    def combine_tensors(A: Tensor3, sumA: Tensor3) -> Tensor3:
        DL, d, DR = sumA.shape
        a, d, b = A.shape
        if DL < a or DR < b:
            # Extend with zeros to accommodate new contribution
            newA = np.zeros((max(DL, a), d, max(DR, b)), dtype=sumA.dtype)
            newA[:DL, :, :DR] = sumA
            sumA = newA
        dt = type(A[0, 0, 0] + sumA[0, 0, 0])
        if sumA.dtype != dt:
            sumA = sumA.astype(dt)
        else:
            sumA[:a, :, :b] += A
        return sumA

    guess: MPS = weights[0] * states[0]
    for n, state in enumerate(states[1:]):
        for i, (A, sumA) in enumerate(zip(state, guess)):
            guess[i] = combine_tensors(A if i > 0 else A * weights[n], sumA)
    return guess


# TODO: We have to rationalize all this about directions. The user should
# not really care about it and we can guess the direction from the canonical
# form of either the guess or the state.
def combine(
    weights: list[Weight],
    states: list[MPS],
    guess: Optional[MPS] = None, 
    truncation: Strategy = SIMPLIFICATION_STRATEGY, 
    direction: int = +1
) -> MPS:
    """Approximate a linear combination of MPS :math:`\\sum_i w_i \\psi_i` by
    another one with a smaller bond dimension, sweeping until convergence is achieved.

    Parameters
    ----------
    weights : list[Weight]
        Weights of the linear combination :math:`w_i` in list form.
    states : list[MPS]
        List of states :math:`\\psi_i`.
    guess : MPS, optional
        Initial guess for the iterative algorithm.
    truncation : Strategy
        Truncation strategy. Defaults to `SIMPLIFICATION_STRATEGY`.
    direction : {+1, -1}
        Initial direction for the sweeping algorithm.

    Returns
    -------
    CanonicalMPS
        Approximation to the linear combination in canonical form
    """
    if guess is None:
        guess = guess_combine_state(weights, states)
    base_error = sum(
        np.sqrt(np.abs(weights)) * np.sqrt(state.error())
        for weights, state in zip(weights, states)
    )
    normalize= truncation.get_normalize_flag()
    maxsweeps = truncation.get_max_sweeps()
    simplification_tolerance = truncation.get_simplification_tolerance()
    start = 0 if direction > 0 else guess.size - 1
    φ = CanonicalMPS(guess, center=start, strategy=truncation, normalize=normalize)
    if not truncation.get_simplification_method():
        return φ
    err = norm_ψsqr = multi_norm_squared(weights, states)
    if norm_ψsqr < simplification_tolerance:
        return MPS([np.zeros((1, P.shape[1], 1)) for P in φ])
    log(
        f"COMBINE state with |state|={norm_ψsqr**0.5} for {maxsweeps} sweeps with tolerance {simplification_tolerance}.\nWeights: {weights}"
    )

    size = φ.size
    forms = [AntilinearForm(φ, state, center=start) for state in states]
    tensor: Tensor4
    for sweep in range(maxsweeps):
        if direction > 0:
            for n in range(0, size - 1):
                tensor = sum(
                    weights * f.tensor2site(direction)
                    for weights, f in zip(weights, forms)
                )  # type: ignore
                φ.update_2site_right(tensor, n, truncation)
                for f in forms:
                    f.update(direction)
        else:
            for n in reversed(range(0, size - 1)):
                tensor = sum(
                    weights * f.tensor2site(direction)
                    for weights, f in zip(weights, forms)
                )  # type: ignore
                φ.update_2site_left(tensor, n, truncation)
                for f in forms:
                    f.update(direction)
            last = 0
        #
        # We estimate the error
        #
        last = φ.center
        B = φ[last]
        norm_φsqr = np.vdot(B, B).real
        if normalize:
            φ[last] = B / norm_φsqr
            norm_φsqr = 1.0
        C = sum(weights * f.tensor1site() for weights, f in zip(weights, forms))
        scprod_φψ = np.vdot(B, C)
        old_err = err
        err = 2 * abs(1.0 - scprod_φψ.real / np.sqrt(norm_φsqr * norm_ψsqr))
        log(f"sweep={sweep}, rel.err.={err}, old err.={old_err}, |φ|={norm_φsqr**0.5}")
        if err < simplification_tolerance or err > old_err:
            log("Stopping, as tolerance reached")
            break
        direction = -direction
    φ._error = 0.0
    φ.update_error(base_error**2)
    φ.update_error(err)
    return φ
