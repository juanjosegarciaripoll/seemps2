from __future__ import annotations
import copy
from math import sqrt
import numpy as np
from ..operators import MPO
from ..qft import qft_mpo
from ..state import DEFAULT_STRATEGY, MPS, CanonicalMPS, MPSSum, Strategy, simplify
from .finite_differences import tridiagonal_mpo
from .space import Space, mpo_flip


def twoscomplement(L: int, strategy: Strategy = DEFAULT_STRATEGY):
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
    ψ0mps: MPS,
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
    ψ0mps: MPS
        Discretized multidimensional function MPS.
    space: Space
        Space object of the defined ψ0mps.
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
    ψfmps: MPS
        Interpolated MPS.
    new_space: Space
        New space of the interpolated MPS.
    """
    old_sites = space.sites
    U2c = space.extend(mpo_flip(twoscomplement(M0)), dim)
    QFT_op = space.extend(qft_mpo(len(old_sites[dim]), sign=+1, strategy=strategy), dim)
    Fψ0mps = U2c @ (QFT_op @ ψ0mps)
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
    new_size = Fψ0mps.size + Mf - M0
    Fψfmps = Fψ0mps.extend(L=new_size, sites=sum(idx_old_sites, []))
    #
    # Undo Fourier transform
    iQFT_op = new_space.extend(
        mpo_flip(qft_mpo(len(new_sites[dim]), sign=-1, strategy=strategy)), dim
    )
    U2c = new_space.extend(mpo_flip(twoscomplement(Mf, strategy=strategy)), dim)
    ψfmps = iQFT_op @ (U2c @ Fψfmps)
    ψfmps = sqrt(2 ** (Mf - M0)) * ψfmps
    factor = sqrt(2 ** (Mf - M0))
    if strategy.get_normalize_flag():
        factor /= ψfmps.norm()
    return factor * ψfmps, new_space


def fourier_interpolation(
    ψmps: MPS,
    space: Space,
    old_sites: list,
    new_sites: list,
    strategy: Strategy = DEFAULT_STRATEGY,
):
    """Fourier interpolation on an MPS.

    Parameters
    ----------
    ψmps : MPS
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
    if not isinstance(ψmps, CanonicalMPS):
        ψmps = CanonicalMPS(ψmps, strategy=strategy)
    for i, sites in enumerate(new_sites):
        ψmps, space = fourier_interpolation_1D(
            ψmps, space, old_sites[i], sites, dim=i, strategy=strategy
        )
    return ψmps


def finite_differences_interpolation_1D(
    ψ0mps: MPS,
    space: Space,
    dim: int = 0,
    strategy: Strategy = DEFAULT_STRATEGY,
    closed: bool = True,
    order: int = 1,
):
    """Finite differences interpolation of dimension dim of an MPS representing
    a multidimensional function.

    Parameters
    ----------
    ψ0mps : MPS
        MPS representing a multidimensional function.
    space : Space
        Space on which the function is defined.
    dim : int
        Dimension to perform the interpolation.
    strategy : Strategy, optional
            Truncation strategy, defaults to DEFAULT_STRATEGY.

    Returns
    -------
    MPS
        Interpolated MPS with one more site for the given dimension.
    """
    if False:
        derivative_mps = (
            space.extend(
                tridiagonal_mpo(
                    len(space.sites[dim]), 0.5, 0, 0.5, closed=closed, strategy=strategy
                ),
                dim,
            )
            @ ψ0mps
        )
        # Extend the state with zero qubits
        new_qubits_per_dimension = space.qubits_per_dimension.copy()
        new_qubits_per_dimension[dim] += 1
        new_space = Space(new_qubits_per_dimension, space.L, space.closed)
        new_sites = new_space.sites
        new_positions = new_sites.copy()
        new_positions[dim] = list(np.array(new_positions[dim][:-(1)]))
        new_size = ψ0mps.size + 1
        derivative_mps = derivative_mps.extend(L=new_size, sites=sum(new_positions, []))
        derivative_mps = (
            new_space.extend(
                tridiagonal_mpo(
                    len(new_space.sites[dim]), 0, 1, 0, closed=closed, strategy=strategy
                ),
                dim,
            )
            @ derivative_mps
        )
        new_ψ0mps = ψ0mps.extend(L=new_size, sites=sum(new_positions, []))
        new_ψ0mps = derivative_mps + new_ψ0mps
        return simplify(new_ψ0mps, strategy=strategy), new_space
    else:
        # Shift operator for finite difference formulas
        Sup = space.extend(
            tridiagonal_mpo(
                len(space.sites[dim]), 0, 0, 1, closed=closed, strategy=strategy
            ),
            dim,
        )
        # Formulas obtained from InterpolatingPolynomial[] in Mathematica
        # First order is just a mid-point interpolation
        if order == 1:
            interpolated_points = simplify(
                MPSSum([0.5, 0.5], [ψ0mps, Sup @ ψ0mps]),
                strategy=strategy,
            )
        elif order == 2:
            f1 = ψ0mps
            f2 = Sup @ f1
            f3 = Sup @ f2
            f0 = Sup.T @ f1
            interpolated_points = simplify(
                MPSSum([-1 / 16, 9 / 16, 9 / 16, -1 / 16], [f0, f1, f2, f3]),
                strategy=strategy,
            )
        elif order == 3:
            Sdo = Sup.T
            f2 = ψ0mps
            f3 = Sup @ f2
            f4 = Sup @ f3
            f5 = Sup @ f4
            f1 = Sdo @ f2
            f0 = Sdo @ f1
            interpolated_points = simplify(
                MPSSum(
                    [-3 / 256, 21 / 256, -35 / 128, 105 / 128, 105 / 256, -7 / 256],
                    [f0, f1, f2, f3, f4, f5],
                ),
                strategy=strategy,
            )

        else:
            raise Exception("Invalid interpolation order")
        #
        # The new space representation with one more qubit
        new_space = space.enlarge_dimension(dim, 1)
        new_positions = new_space.new_positions_from_old_space(space)
        #
        # We create an MPS by extending the old one to the even sites
        # and placing the interpolating polynomials in an MPS that
        # is only nonzero in the odd sites. We then add. There are better
        # ways for sure.
        odd = ψ0mps.extend(
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
    ψmps: MPS, space: Space, strategy: Strategy = DEFAULT_STRATEGY
):
    """Finite differences interpolation of an MPS representing
    a multidimensional function.

    Parameters
    ----------
    ψ0mps : MPS
        MPS representing a multidimensional function.
    space : Space
        Space on which the function is defined.
    strategy : Strategy, optional
            Truncation strategy, defaults to DEFAULT_STRATEGY.

    Returns
    -------
    MPS
        Interpolated MPS with one more site for each dimension.
    """
    space = copy.deepcopy(space)
    if not isinstance(ψmps, CanonicalMPS):
        ψmps = CanonicalMPS(ψmps, strategy=strategy)
    for i, _ in enumerate(space.qubits_per_dimension):
        ψmps, space = finite_differences_interpolation_1D(
            ψmps, space, dim=i, strategy=strategy
        )
    return ψmps
