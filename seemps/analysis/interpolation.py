import copy

import numpy as np

from ..qft import qft_mpo
from ..state import CanonicalMPS
from .finite_differences import mpo_combined
from .space import *


def fourier_interpolation_1D(f, M, axis=0):
    """Obtain the Fourier interpolated array over the given
    axis with a new number of points M.

    Parameters
    ----------
    f : numpy.ndarray
        Discretized multidimensional function array.
    M : int
        Final number of points of the interpolated axis.
    axis : int
        Axis to perform the interpolation.

    Returns
    -------
    numpy.ndarray
        M-point interpolated function on given axis.
    """
    d = f.shape
    f = np.fft.ifft(f, norm="ortho", axis=axis)
    dims = list(f.shape)
    dims[axis] = M - d[axis]
    filler = np.zeros(dims, dtype=type(f[0]))
    if axis == 1:
        filler = filler.T
    f = np.insert(f, d[axis] // 2, filler, axis=axis)
    f = np.fft.fft(f, norm="ortho", axis=axis)
    return f * np.sqrt(M / d[axis])


def fourier_interpolation(f, new_dims):
    """Fourier interpolation on an n-dimensional array.

    Parameters
    ----------
    f : numpy.ndarray
        Discretized multidimensional function array.
    new_dims : list[int]
        List of integers with the new dimensions for each axis
        of the array.

    Returns
    -------
    numpy.ndarray
        Interpolated multidimensional function array.
    """
    for i, dim in enumerate(new_dims):
        f = fourier_interpolation_1D(f, dim, axis=i)
    return f


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


def fourier_interpolation_mps_1D(ψ0mps, M0, Mf, space, dim, **kwargs):
    """Obtain the Fourier interpolated MPS over the chosen dimension
    with a new number of sites Mf.

    Parameters
    ---------
    ψ0mps: MPS
        Discretized multidimensional function MPS.
    MO: int
        Initial number of sites.
    Mf: int
        Final number of sites.
    space: Space
        Space object of the defined ψ0mps.
    dim: int
        Dimension to perform the interpolation.

    Returns
    -------
    ψfmps: MPS
        Interpolated MPS.
    new_space: Space
        New space of the interpolated MPS.
    """
    old_sites = space.sites
    U2c = space.extend(mpo_flip(twoscomplement(M0, **kwargs)), dim)
    QFT_op = space.extend(qft_mpo(len(old_sites[dim]), sign=+1, **kwargs), dim)
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
        mpo_flip(qft_mpo(len(new_sites[dim]), sign=-1, **kwargs)), dim
    )
    U2c = new_space.extend(mpo_flip(twoscomplement(Mf, **kwargs)), dim)
    ψfmps = iQFT_op @ (U2c @ Fψfmps)
    ψfmps = ψfmps * (1 / np.sqrt(ψfmps.norm_squared()))

    return ψfmps, new_space


def fourier_interpolation_mps(ψmps, old_sites, new_sites, space, **kwargs):
    """Fourier interpolation on an MPS.

    Parameters
    ----------
    ψmps : MPS
        Discretized multidimensional function MPS.
    old_sites : list[int]
        List of integers with the original number of sites for each dimension.
    new_sites : list[int]
        List of integers with the new number of sites for each dimension.
    space: Space
        Space object of the defined ψmps.
    **kwargs :
        Arguments accepted by :class:`MPO`

    Returns
    -------
    MPS
        Interpolated multidimensional function MPS.

    """
    space = copy.copy(space)
    for i, sites in enumerate(new_sites):
        ψmps, space = fourier_interpolation_mps_1D(
            ψmps, old_sites[i], sites, space, dim=i, **kwargs
        )
    return ψmps


def interpolate_first_axis(f):
    """Finite differences interpolation of the first axis of a multidimensional
    array.

    Parameters
    ----------
    f : numpy.ndarray
        Discretized multidimensional function array.

    Returns
    -------
    numpy.ndarray
        Interpolated function with double of points on given axis.
    """
    f = np.asarray(f)
    dims = f.shape
    new_dims = (dims[0] * 2,) + dims[1:]
    output = np.zeros(new_dims)
    output[::2, :] = f
    output[1::2, :] = (f + np.roll(f, -1, 0)) / 2
    return output


def finite_differences_interpolation_2D(f):
    """Finite differences interpolation of the first axis of a multidimensional
    array.

    Parameters
    ----------
    f : numpy.ndarray
        Interpolated function with double of points.

    Returns
    -------
    numpy.ndarray
        Interpolated function with double of points.
    """
    f = interpolate_first_axis(f)
    f = np.transpose(f, [1, 0])
    f = interpolate_first_axis(f)
    return np.transpose(f, [1, 0])


def finite_differences_interpolation_mps_1D(ψ0mps, space, dim=0, **kwargs):
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
    **kwargs :
        Other arguments accepted by :class:`MPO`

    Returns
    -------
    MPS
        Interpolated MPS with one more site for the given dimension.
    """
    derivative_mps = (
        space.extend(mpo_combined(len(space.sites[dim]), 0.5, 0, 0.5, **kwargs), dim)
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
            mpo_combined(len(new_space.sites[dim]), 0, 1, 0, **kwargs), dim
        )
        @ derivative_mps
    )
    new_ψ0mps = ψ0mps.extend(L=new_size, sites=sum(idx_old_sites, []))
    new_ψ0mps = derivative_mps + new_ψ0mps
    return new_ψ0mps.toMPS(), new_space


def finite_differences_interpolation_mps(ψmps, space, **kwargs):
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
        ψmps, space = finite_differences_interpolation_mps_1D(
            ψmps, space, dim=i, **kwargs
        )
    return ψmps
