from __future__ import annotations
from math import sqrt
import numpy as np
from ..tools import make_logger
from . import (
    DEFAULT_TOLERANCE,
    MAX_BOND_DIMENSION,
    MPS,
    CanonicalMPS,
    MPSSum,
    Simplification,
    Strategy,
    Truncation,
)
from ..typing import Weight
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
    state: MPS | MPSSum,
    strategy: Strategy = SIMPLIFICATION_STRATEGY,
    direction: int = +1,
    guess: MPS | None = None,
) -> CanonicalMPS:
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
    guess : MPS
        A guess for the new state, to ease the optimization.

    Returns
    -------
    CanonicalMPS
        Approximation :math:`\\xi` to the state.
    """
    if isinstance(state, MPSSum):
        return simplify_mps_sum(state, strategy, direction, guess)

    # Prepare initial guess
    normalize = strategy.get_normalize_flag()
    size = state.size
    start = 0 if direction > 0 else -1
    logger = make_logger(2)

    # If we only do canonical forms, not variational optimization, a second
    # pass on that initial guess suffices
    if strategy.get_simplification_method() == Simplification.CANONICAL_FORM:
        mps = CanonicalMPS(state, center=start, strategy=strategy).recenter(
            -1 - start, strategy
        )
        if logger:
            logger(
                f"SIMPLIFY state with |state|={mps.norm():5e}\nusing two-pass "
                + f"canonical form, with tolerance {strategy.get_tolerance():5e}\n"
                + f"produces error {mps.error():5e}.\nStrategy: {strategy}",
            )
        return mps

    # TODO: DO_NOT_SIMPLIFY should do nothing. However, since the
    # output is expected to be a CanonicalMPS, we must use the
    # strategy to construct it.
    if strategy.get_simplification_method() == Simplification.DO_NOT_SIMPLIFY:
        mps = CanonicalMPS(state, center=-1 - start, strategy=strategy)
        if logger:
            logger(
                f"SIMPLIFY state with |state|={mps.norm():5e}\nusing single-pass "
                + f"canonical form, with tolerance {strategy.get_tolerance():5e}\n"
                + f"produces error {mps.error():5e}.\nStrategy: {strategy}",
            )
        return mps

    mps = CanonicalMPS(
        state if guess is None else guess,
        center=start,
        normalize=False,
        strategy=strategy,
    )

    simplification_tolerance = strategy.get_simplification_tolerance()
    if not (norm_state_sqr := state.norm_squared()):
        return CanonicalMPS(state.zero_state(), is_canonical=True)
    form = AntilinearForm(mps, state, center=start)
    err = 2.0
    if logger:
        logger(
            f"SIMPLIFY state with |state|={norm_state_sqr**0.5} for "
            + f"{strategy.get_max_sweeps()} sweeps, with tolerance {simplification_tolerance}.\nStrategy: {strategy}",
        )
    norm_mps_sqr = 0.0
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
        err = 2 * abs(1.0 - mps_state_scprod.real / sqrt(norm_mps_sqr * norm_state_sqr))
        if logger:
            logger(
                f"sweep={sweep}, rel.err.={err:6g}, old err.={old_err:6g}, |mps|={norm_mps_sqr**0.5:6g}, tol={simplification_tolerance:6g}",
            )
        if err < simplification_tolerance or err > old_err:
            logger("Stopping, as tolerance reached")
            break
        direction = -direction
    total_error_bound = state._error + sqrt(err)
    if normalize and norm_mps_sqr:
        factor = sqrt(norm_mps_sqr)
        last_tensor /= factor  # pyright: ignore[reportPossiblyUnboundVariable]
        total_error_bound /= factor
    mps._error = total_error_bound
    logger.close()
    return mps


# TODO: We have to rationalize all this about directions. The user should
# not really care about it and we can guess the direction from the canonical
# form of either the guess or the state.
def simplify_mps_sum(
    sum_state: MPSSum,
    strategy: Strategy = SIMPLIFICATION_STRATEGY,
    direction: int = +1,
    guess: MPS | None = None,
) -> CanonicalMPS:
    """Approximate a linear combination of MPS :math:`\\sum_i w_i \\psi_i` by
    another one with a smaller bond dimension, sweeping until convergence is achieved.

    Parameters
    ----------
    state : MPSSum
        State to approximate
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
    # Compute norm of output and eliminate zero states
    norm_state_sqr = sum_state.delete_zero_components()
    logger = make_logger(2)
    if not norm_state_sqr:
        if logger:
            logger("COMBINE state with |state|=0. Returning zero state.")
        return CanonicalMPS(sum_state.states[0].zero_state(), is_canonical=True)

    normalize = strategy.get_normalize_flag()
    start = 0 if direction > 0 else -1
    # CANONICAL_FORM implements a simplification based on two passes
    if strategy.get_simplification_method() == Simplification.CANONICAL_FORM:
        mps = CanonicalMPS(sum_state.join(), center=start, strategy=strategy).recenter(
            -1 - start, strategy
        )
        if logger:
            logger(
                f"COMBINE state with |state|={mps.norm():5e}\nusing two-pass "
                + f"canonical form, with tolerance {strategy.get_tolerance():5e}\n"
                + f"produces error {mps.error():5e}.\nStrategy: {strategy}",
            )
            logger.close()
        return mps

    # TODO: DO_NOT_SIMPLIFY should do nothing. However, since the
    # output is expected to be a CanonicalMPS, we must use the
    # strategy to construct it.
    if strategy.get_simplification_method() == Simplification.DO_NOT_SIMPLIFY:
        mps = CanonicalMPS(sum_state.join(), center=-1 - start, strategy=strategy)
        if logger:
            logger(
                f"COMBINE state with |state|={mps.norm():5e}\nusing single-pass "
                + f"canonical form, with tolerance {strategy.get_tolerance():5e}\n"
                + f"produces error {mps.error():5e}.\nStrategy: {strategy}",
            )
            logger.close()
        return mps

    # Prepare initial guess
    mps = CanonicalMPS(
        sum_state.join() if guess is None else guess,
        center=start,
        normalize=False,
        strategy=strategy,
    )
    simplification_tolerance = strategy.get_simplification_tolerance()

    size = mps.size
    weights, states = sum_state.weights, sum_state.states
    forms = [AntilinearForm(mps, si, center=start) for si in states]
    if logger:
        logger(
            f"COMBINE state with |state|={norm_state_sqr**0.5:5e} for {strategy.get_max_sweeps():5e}"
            + f"sweeps with tolerance {simplification_tolerance:5e}.\nStrategy: {strategy}"
            + f"\nWeights: {weights}",
        )

    err = 2.0
    norm_mps_sqr = 0.0
    for sweep in range(max(1, strategy.get_max_sweeps())):
        if direction > 0:
            for n in range(0, size - 1):
                mps.update_2site_right(
                    sum(w * f.tensor2site(direction) for w, f in zip(weights, forms)),  # type: ignore # pyright: ignore[reportArgumentType]
                    n,
                    strategy,
                )
                for f in forms:
                    f.update_right()
            last_tensor = mps[size - 1]
        else:
            for n in reversed(range(0, size - 1)):
                mps.update_2site_left(
                    sum(w * f.tensor2site(direction) for w, f in zip(weights, forms)),  # type: ignore # pyright: ignore[reportArgumentType]
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
        err = 2 * abs(1.0 - mps_state_scprod.real / sqrt(norm_mps_sqr * norm_state_sqr))
        if logger:
            logger(
                f"sweep={sweep}, rel.err.={err:6g}, old err.={old_err:6g}, |mps|={norm_mps_sqr**0.5:6g}, tol={simplification_tolerance:6g}",
            )
        if err < simplification_tolerance or err > old_err:
            logger("Stopping, as tolerance reached")
            break
        direction = -direction
    total_error_bound = sum_state.error() + sqrt(err)
    if normalize and norm_mps_sqr:
        factor = sqrt(norm_mps_sqr)
        last_tensor /= factor  # pyright: ignore[reportPossiblyUnboundVariable]
        total_error_bound /= factor
    mps._error = total_error_bound
    logger.close()
    return mps


def combine(
    weights: list[Weight],
    states: list[MPS],
    strategy: Strategy = SIMPLIFICATION_STRATEGY,
    direction: int = +1,
    guess: MPS | None = None,
) -> CanonicalMPS:
    """Deprecated, use `simplify` instead."""
    return simplify_mps_sum(MPSSum(weights, states))


simplify_mps = simplify

__all__ = ["simplify", "simplify_mps"]
