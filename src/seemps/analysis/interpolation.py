import copy
import numpy as np
from ..qft import qft_mpo
from ..state import DEFAULT_STRATEGY
from ..truncate import simplify
from .finite_differences import mpo_combined
from .space import *


def twoscomplement(L, **kwdargs):
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
    return MPO([A0] + [A] * (L - 2) + [Aend], **kwdargs)


def fourier_interpolation_1D(ψ0mps, space, M0, Mf, dim, strategy=DEFAULT_STRATEGY):
    """Obtain the Fourier interpolated MPS over the chosen dimension
    with a new number of sites Mf.

    Parameters
    ---------
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
    ψfmps = ψfmps * (1 / np.sqrt(ψfmps.norm_squared()))

    return ψfmps, new_space


def fourier_interpolation(ψmps, space, old_sites, new_sites, **kwargs):
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
    **kwargs :
        Arguments accepted by :class:`MPO`

    Returns
    -------
    MPS
        Interpolated multidimensional function MPS.

    """
    space = copy.copy(space)
    for i, sites in enumerate(new_sites):
        ψmps, space = fourier_interpolation_1D(
            ψmps, space, old_sites[i], sites, dim=i, **kwargs
        )
    return ψmps


def finite_differences_interpolation_1D(
    ψ0mps, space, dim=0, strategy=DEFAULT_STRATEGY, closed=False
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
    derivative_mps = (
        space.extend(
            mpo_combined(
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
    idx_old_sites = new_sites.copy()
    idx_old_sites[dim] = list(np.array(idx_old_sites[dim][:-(1)]))
    new_size = ψ0mps.size + 1
    derivative_mps = derivative_mps.extend(L=new_size, sites=sum(idx_old_sites, []))
    derivative_mps = (
        new_space.extend(
            mpo_combined(
                len(new_space.sites[dim]), 0, 1, 0, closed=closed, strategy=strategy
            ),
            dim,
        )
        @ derivative_mps
    )
    new_ψ0mps = ψ0mps.extend(L=new_size, sites=sum(idx_old_sites, []))
    new_ψ0mps = derivative_mps + new_ψ0mps
    return simplify(new_ψ0mps, strategy=strategy), new_space


def finite_differences_interpolation(ψmps, space, **kwargs):
    """Finite differences interpolation of an MPS representing
    a multidimensional function.

    Parameters
    ----------
    ψ0mps : MPS
        MPS representing a multidimensional function.
    space : Space
        Space on which the function is defined.
    **kwargs :
        Other arguments accepted by :class:`MPO`

    Returns
    -------
    MPS
        Interpolated MPS with one more site for each dimension.
    """
    space = copy.deepcopy(space)
    for i, q in enumerate(space.qubits_per_dimension):
        ψmps, space = finite_differences_interpolation_1D(ψmps, space, dim=i, **kwargs)
    return ψmps
