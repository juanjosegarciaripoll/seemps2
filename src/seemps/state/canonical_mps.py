from __future__ import annotations
import warnings
import numpy as np
from typing import Optional, Sequence, Iterable
from ..typing import Vector, Tensor3, Tensor4, VectorLike, Environment
from . import environments, schmidt
from seemps.state.core import (
    DEFAULT_STRATEGY,
    Strategy,
    _update_canonical_right,
    _update_canonical_left,
    _update_canonical_2site_left,
    _update_canonical_2site_right,
    _canonicalize,
    CanonicalMPS,
)
from .mps import MPS


@classmethod  # type: ignore
def from_vector(
    cls,
    ψ: VectorLike,
    dimensions: Sequence[int],
    strategy: Strategy = DEFAULT_STRATEGY,
    normalize: bool = True,
    center: int = 0,
    **kwdargs,
) -> CanonicalMPS:
    """Create an MPS in canonical form starting from a state vector.

    Parameters
    ----------
    ψ : VectorLike
        Real or complex vector of a wavefunction.
    dimensions : Sequence[int]
        Sequence of integers representing the dimensions of the
        quantum systems that form this state.
    strategy : Strategy, default = DEFAULT_STRATEGY
        Default truncation strategy for algorithms working on this state.
    normalize : bool, default = True
        Whether the state is normalized to compensate truncation errors.
    center : int, default = 0
        Center for the canonical form of this decomposition.

    Returns
    -------
    CanonicalMPS
        A valid matrix-product state approximating this state vector.

    See also
    --------
    :py:meth:`~seemps.state.MPS.from_vector`
    """
    data, error = schmidt._vector2mps(ψ, dimensions, strategy, normalize, center)
    return CanonicalMPS(
        data,
        error=error,
        center=center,
        is_canonical=True,
    )


CanonicalMPS.from_vector = from_vector  # type: ignore

__all__ = ["CanonicalMPS"]
