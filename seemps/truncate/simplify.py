from __future__ import annotations

import numpy as np

from ..state import (
    DEFAULT_TOLERANCE,
    MAX_BOND_DIMENSION,
    MPS,
    CanonicalMPS,
    MPSSum,
    Simplification,
    Strategy,
    Truncation,
)
from ..state.environments import scprod
from ..tools import log, DEBUG
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
    max_sweeps=4,
    simplify=Simplification.VARIATIONAL,
)


def simplify(
    state: Union[MPS, MPSSum],
    strategy: Strategy = SIMPLIFICATION_STRATEGY,
    direction: int = +1,
) -> MPS:
    """Simplify an MPS state transforming it into another one with a smaller bond
    dimension, sweeping until convergence is achieved.

    Parameters
    ----------
    state : MPS | MPSSum
        State to approximate.
    strategy : Strategy
        Truncation strategy. Defaults to `SIMPLIFICATION_STRATEGY`.
    direction : { +1, -1 }
        Initial direction for the sweeping algorithm.

    Returns
    -------
    CanonicalMPS
        Approximation :math:`\\xi` to the state.
    """
    if strategy.get_simplification_method() == Simplification.DO_NOT_SIMPLIFY:
        raise ValueError("Not valid simplification method.")
    if isinstance(state, MPSSum):
        return combine(
            state.weights,
            state.states,
            strategy=strategy,
            direction=direction,
        )
    #
    # Prepare initial guess
    normalize = strategy.get_normalize_flag()
    size = state.size
    start = 0 if direction > 0 else size - 1
    mps = CanonicalMPS(state, center=start, strategy=strategy)

    # If we only do canonical forms, not variational optimization, a second
    # pass on that initial guess suffices
    if strategy.get_simplification_method() == Simplification.CANONICAL_FORM:
        mps = CanonicalMPS(mps, center=-1 - start, strategy=strategy)
        if normalize:
            mps.normalize_inplace()
        return mps

    simplification_tolerance = strategy.get_simplification_tolerance()
    norm_state_sqr = scprod(state, state).real
    form = AntilinearForm(mps, state, center=start)
    err = 2.0
    if DEBUG:
        log(
            f"SIMPLIFY state with |state|={norm_state_sqr**0.5} for "
            f"{strategy.get_max_sweeps()} sweeps, with tolerance {simplification_tolerance}."
        )
    for sweep in range(max(1, strategy.get_max_sweeps())):
        if direction > 0:
            for n in range(0, size - 1):
                mps.update_2site_right(form.tensor2site(direction), n, strategy)
                form.update_right()
            last_tensor = mps[size - 1]
        else:
            for n in reversed(range(0, size - 1)):
                mps.update_2site_left(form.tensor2site(direction), n, strategy)
                form.update_left()
            last_tensor = mps[0]
        #
        # We estimate the error
        #
        norm_mps_sqr = np.vdot(last_tensor, last_tensor).real
        mps_state_scprod = np.vdot(last_tensor, form.tensor1site())
        old_err = err
        err = 2 * abs(
            1.0 - mps_state_scprod.real / np.sqrt(norm_mps_sqr * norm_state_sqr)
        )
        if DEBUG:
            log(
                f"sweep={sweep}, rel.err.={err}, old err.={old_err}, "
                f"|mps|={norm_mps_sqr**0.5}"
            )
        if err < simplification_tolerance or err > old_err:
            log("Stopping, as tolerance reached")
            break
        direction = -direction
    mps.set_error(state.error())
    mps.update_error(err)
    if normalize:
        last_tensor /= norm_mps_sqr
    return mps


def multi_norm_squared(weights: list[Weight], states: list[MPS]) -> float:
    """Compute the norm-squared of the linear combination of weights and
    states."""
    c: float = 0.0
    for i, wi in enumerate(weights):
        for j in range(i):
            c += 2 * (wi.conjugate() * weights[j] * scprod(states[i], states[j])).real
        c += np.abs(wi) ** 2 * scprod(states[i], states[i]).real
    return abs(c)


def crappy_guess_combine_state(weights: list[Weight], states: list[MPS]) -> MPS:
    """Make an educated guess that ensures convergence of the :func:`combine`
    algorithm."""

    def combine_tensors(A: Tensor3, sumA: Tensor3) -> Tensor3:
        DL, d, DR = sumA.shape
        a, d, b = A.shape
        if DL < a or DR < b:
            # Extend with zeros to accommodate new contribution
            newA = np.zeros((max(DL, a), d, max(DR, b)), dtype=sumA.dtype)
            newA[:DL, :, :DR] = sumA
        else:
            newA = sumA.copy()
        dt = type(A[0, 0, 0] + sumA[0, 0, 0])
        if sumA.dtype != dt:
            newA = sumA.astype(dt)
        else:
            newA[:a, :, :b] += A
        return newA

    guess: MPS = weights[0] * states[0]
    for n, state in enumerate(states[1:]):
        for i, (A, sumA) in enumerate(zip(state, guess)):
            guess[i] = combine_tensors(A if i > 0 else A * weights[n], sumA)
    return guess


def guess_combine_state(weights: list, states: list[MPS]) -> MPS:
    """Make an educated guess that ensures convergence of the :func:`combine`
    algorithm."""

    def combine_tensors(A, sumA, idx):
        DL, d, DR = sumA.shape
        a, d, b = A.shape
        if idx == 0:
            new_A = np.zeros((d, DR + b))
            for d_i in range(d):
                new_A[d_i, :] = np.concatenate(
                    (sumA.reshape(d, DR)[d_i, :], A.reshape(d, b)[d_i, :])
                )
            new_A = new_A.reshape(1, d, DR + b)
        elif idx == -1:
            new_A = np.zeros((DL + a, d))
            for d_i in range(d):
                new_A[:, d_i] = np.concatenate(
                    (sumA.reshape(DL, d)[:, d_i], A.reshape(a, d)[:, d_i])
                )
            new_A = new_A.reshape(DL + a, d, 1)
        else:
            new_A = np.zeros((DL + a, d, DR + b))
            for d_i in range(d):
                new_A[:, d_i, :] = scipy.linalg.block_diag(
                    sumA[:, d_i, :], A[:, d_i, :]
                )
        return new_A

    guess = []
    size = states[0].size
    for i in range(size):
        sumA = states[0][i] * weights[0] if i == 0 else states[0][i]
        if i == size - 1:
            i = -1
        for n, state in enumerate(states[1:]):
            A = state[i]
            sumA = combine_tensors(A * weights[n + 1] if i == 0 else A, sumA, i)
        guess.append(sumA)
    return MPS(guess)


# TODO: We have to rationalize all this about directions. The user should
# not really care about it and we can guess the direction from the canonical
# form of either the guess or the state.
def combine(
    weights: list[Weight],
    states: list[MPS],
    guess: Optional[MPS] = None,
    strategy: Strategy = SIMPLIFICATION_STRATEGY,
    direction: int = +1,
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
    strategy : Strategy
        Truncation strategy. Defaults to `SIMPLIFICATION_STRATEGY`.
    direction : {+1, -1}
        Initial direction for the sweeping algorithm.

    Returns
    -------
    CanonicalMPS
        Approximation to the linear combination in canonical form
    """
    #
    # Prepare initial guess
    if guess is None:
        if strategy.get_simplification_method() == Simplification.CANONICAL_FORM:
            mps = guess_combine_state(weights, states)
        elif strategy.get_simplification_method() == Simplification.VARIATIONAL:
            mps = crappy_guess_combine_state(weights, states)
    else:
        mps = guess
    normalize = strategy.get_normalize_flag()
    start = 0 if direction > 0 else mps.size - 1
    mps = CanonicalMPS(mps, center=start, strategy=strategy)

    # If we only do canonical forms, not variational optimization, a second
    # pass on that initial guess suffices
    if strategy.get_simplification_method() == Simplification.CANONICAL_FORM:
        mps = CanonicalMPS(mps, center=-1 - start, strategy=strategy)
        if normalize:
            mps.normalize_inplace()
        return mps

    simplification_tolerance = strategy.get_simplification_tolerance()
    norm_state_sqr = multi_norm_squared(weights, states)
    if DEBUG:
        log(
            f"COMBINE state with |state|={norm_state_sqr**0.5} for strategy.get_max_sweeps()) "
            f"sweeps with tolerance {simplification_tolerance}.\nWeights: {weights}"
        )
    size = mps.size
    forms = [AntilinearForm(mps, state, center=start) for state in states]
    err = 2.0
    for sweep in range(max(1, strategy.get_max_sweeps())):
        if direction > 0:
            for n in range(0, size - 1):
                mps.update_2site_right(
                    sum(w * f.tensor2site(direction) for w, f in zip(weights, forms)),  # type: ignore
                    n,
                    strategy,
                )
                for f in forms:
                    f.update_right()
            last_tensor = mps[size - 1]
        else:
            for n in reversed(range(0, size - 1)):
                mps.update_2site_left(
                    sum(w * f.tensor2site(direction) for w, f in zip(weights, forms)),  # type: ignore
                    n,
                    strategy,
                )
                for f in forms:
                    f.update_left()
            last_tensor = mps[0]
        #
        # We estimate the error
        #
        norm_mps_sqr = np.vdot(last_tensor, last_tensor).real
        mps_state_scprod = np.vdot(
            last_tensor,
            sum(w * f.tensor1site() for w, f in zip(weights, forms)),
        )
        old_err = err
        err = 2 * abs(
            1.0 - mps_state_scprod.real / np.sqrt(norm_mps_sqr * norm_state_sqr)
        )
        if DEBUG:
            log(
                f"sweep={sweep}, rel.err.={err}, old err.={old_err}, "
                f"|mps|={norm_mps_sqr**0.5}"
            )
        if err < simplification_tolerance or err > old_err:
            log("Stopping, as tolerance reached")
            break
        direction = -direction
    base_error = sum(
        np.abs(weights) * np.sqrt(state.error())
        for weights, state in zip(weights, states)
    )
    mps.set_error(base_error**2)
    mps.update_error(err)
    if normalize:
        last_tensor /= norm_mps_sqr
    return mps
