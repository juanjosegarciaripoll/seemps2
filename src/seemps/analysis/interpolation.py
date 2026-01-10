from __future__ import annotations
import copy
from math import sqrt
from typing import Literal
import numpy as np
from ..register import mpo_weighted_shifts
from ..operators import MPO
from ..qft import qft_mpo
from ..state import DEFAULT_STRATEGY, MPS, CanonicalMPS, Strategy, simplify
from .space import Space, mpo_flip


def twos_complement(L: int, strategy: Strategy = DEFAULT_STRATEGY):
    """Two's complement operation."""
    A0 = np.zeros((1, 2, 2, 2))
    A0[0, 0, 0, 0] = 1.0
    A0[0, 1, 1, 1] = 1.0
    A = np.zeros((2, 2, 2, 2))
    A[0, 0, 0, 0] = 1.0
    A[0, 1, 1, 0] = 1.0
    A[1, 1, 0, 1] = 1.0
    A[1, 0, 1, 1] = 1.0
    Aend = A[:, :, :, [0]] + A[:, :, :, [1]]
    return MPO([A0] + [A] * (L - 2) + [Aend], strategy)


def fourier_interpolation_1D(
    vector: MPS,
    space: Space,
    M0: int,
    Mf: int,
    dim: int,
    strategy: Strategy = DEFAULT_STRATEGY,
):
    """Obtain the Fourier interpolated MPS over the chosen dimension
    with a new number of sites Mf.

    Parameters
    ----------
    vector: MPS
        Discretized multidimensional function MPS.
    space: Space
        Space object of the defined vector.
    MO: int
        Initial number of sites.
    Mf: int
        Final number of sites.
    dim: int
        Dimension to perform the interpolation.
    strategy : Strategy, optional
        Truncation strategy, defaults to DEFAULT_STRATEGY

    Returns
    -------
    result: MPS
        Interpolated MPS.
    new_space: Space
        New space of the interpolated MPS.
    """
    old_sites = space.sites
    U2c = space.extend(mpo_flip(twos_complement(M0)), dim)
    QFT_op = space.extend(qft_mpo(len(old_sites[dim]), sign=+1, strategy=strategy), dim)
    Fvector = U2c @ (QFT_op @ vector)
    #
    # Extend the state with zero qubits
    new_qubits_per_dimension = space.qubits_per_dimension.copy()
    new_qubits_per_dimension[dim] += Mf - M0
    new_space = Space(new_qubits_per_dimension, space.L, space.closed)
    new_sites = new_space.sites
    idx_old_sites = new_sites.copy()
    idx_old_sites[dim] = list(
        np.append(idx_old_sites[dim][: (-(Mf - M0) - 1)], idx_old_sites[dim][-1])
    )
    new_size = Fvector.size + Mf - M0
    Fresult = Fvector.extend(L=new_size, sites=sum(idx_old_sites, []))
    #
    # Undo Fourier transform
    iQFT_op = new_space.extend(
        mpo_flip(qft_mpo(len(new_sites[dim]), sign=-1, strategy=strategy)), dim
    )
    U2c = new_space.extend(mpo_flip(twos_complement(Mf, strategy=strategy)), dim)
    result = iQFT_op @ (U2c @ Fresult)
    result = sqrt(2 ** (Mf - M0)) * result
    factor = sqrt(2 ** (Mf - M0))
    if strategy.get_normalize_flag():
        factor /= result.norm()
    return factor * result, new_space


def fourier_interpolation(
    tensor: MPS,
    space: Space,
    old_sites: list,
    new_sites: list,
    strategy: Strategy = DEFAULT_STRATEGY,
):
    """Fourier interpolation on an MPS.

    Parameters
    ----------
    tensor : MPS
        Discretized multidimensional function MPS.
    space: Space
        Space object of the defined ψmps.
    old_sites : list[int]
        List of integers with the original number of sites for each dimension.
    new_sites : list[int]
        List of integers with the new number of sites for each dimension.
    strategy : Strategy, optional
        Truncation strategy, defaults to DEFAULT_STRATEGY.

    Returns
    -------
    MPS
        Interpolated multidimensional function MPS.
    """
    space = copy.copy(space)
    if not isinstance(tensor, CanonicalMPS):
        tensor = CanonicalMPS(tensor, strategy=strategy)
    for i, sites in enumerate(new_sites):
        tensor, space = fourier_interpolation_1D(
            tensor, space, old_sites[i], sites, dim=i, strategy=strategy
        )
    return tensor


def finite_differences_interpolation_1D(
    vector: MPS,
    space: Space,
    dim: int = 0,
    order: Literal[1] | Literal[2] | Literal[3] = 1,
    strategy: Strategy = DEFAULT_STRATEGY,
):
    """Finite differences interpolation of dimension dim of an MPS representing
    a multidimensional function.

    Parameters
    ----------
    vector : MPS
        MPS representing a multidimensional function.
    space : Space
        Space on which the function is defined.
    dim : int
        Dimension to perform the interpolation.
    order : int
        Interpolation order, 1, 2 or 3 (defaults to 1).
    strategy : Strategy, optional
        Truncation strategy, defaults to DEFAULT_STRATEGY.

    Returns
    -------
    MPS
        Interpolated MPS with one more site for the given dimension.
    """
    # Shift operator for finite difference formulas
    # Formulas obtained from InterpolatingPolynomial[] in Mathematica
    # First order is just a mid-point interpolation
    match order:
        case 1:
            weights = [0.5, 0.5]
            shifts = [0, -1]
        case 2:
            weights = [-1 / 16, 9 / 16, 9 / 16, -1 / 16]
            shifts = [1, 0, -1, -2]
        case 3:
            weights = [-3 / 256, 21 / 256, -35 / 128, 105 / 128, 105 / 256, -7 / 256]
            shifts = [2, 1, 0, -1, -2, -3]
        case _:
            raise Exception("Invalid interpolation order")
    interpolant = mpo_weighted_shifts(
        vector.size, weights, shifts, periodic=space.closed
    )
    interpolated_points = interpolant.apply(vector, strategy=strategy, simplify=True)
    #
    # The new space representation with one more qubit
    new_space = space.enlarge_dimension(dim, 1)
    new_positions = new_space.new_positions_from_old_space(space)
    #
    # We create an MPS by extending the old one to the even sites
    # and placing the interpolating polynomials in an MPS that
    # is only nonzero in the odd sites. We then add. There are better
    # ways for sure.
    odd = vector.extend(
        L=new_space.n_sites,
        sites=new_positions,
        dimensions=2,
        state=np.asarray([1.0, 0.0]),
    )
    even = interpolated_points.extend(
        L=new_space.n_sites,
        sites=new_positions,
        dimensions=2,
        state=np.asarray([0.0, 1.0]),
    )
    return simplify(odd + even, strategy=strategy), new_space


def finite_differences_interpolation(
    tensor: MPS,
    space: Space,
    order: Literal[1] | Literal[2] | Literal[3] = 1,
    strategy: Strategy = DEFAULT_STRATEGY,
):
    """Finite differences interpolation of an MPS representing
    a multidimensional function.

    Parameters
    ----------
    tensor : MPS
        MPS representing a multidimensional function.
    space : Space
        Space on which the function is defined.
    order : int
        Interpolation order, 1, 2 or 3 (defaults to 1).
    strategy : Strategy, optional
        Truncation strategy, defaults to DEFAULT_STRATEGY.

    Returns
    -------
    MPS
        Interpolated MPS with one more site for each dimension.
    """
    space = copy.deepcopy(space)
    if not isinstance(tensor, CanonicalMPS):
        tensor = CanonicalMPS(tensor, strategy=strategy)
    for i, _ in enumerate(space.qubits_per_dimension):
        tensor, space = finite_differences_interpolation_1D(
            tensor, space, dim=i, strategy=strategy, order=order
        )
    return tensor


__all__ = [
    "twos_complement",
    "fourier_interpolation_1D",
    "fourier_interpolation",
    "finite_differences_interpolation_1D",
    "finite_differences_interpolation",
]
