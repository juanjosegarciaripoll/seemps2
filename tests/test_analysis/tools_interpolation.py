import numpy as np
from math import sqrt


def gaussian(r):
    """Constructs a Gaussian function defined on the given grid.

    Parameters
    ----------
    r: grid on which the Gaussian is defined

    Returns
    -------
    np.ndarray: Gaussian on grid r.
    """
    n_dims = r.shape[0]
    exponent = sum(r[i, :] ** 2 for i in range(n_dims))
    f = np.exp(-exponent / 2)
    return f / np.linalg.norm(f)


def get_position_regular_grid(dims: list, a: list, dx: list) -> np.ndarray:
    """Construct a regular grid r[d,i1,i2,...,id] to encode the coordinates of
    each index in position space, where d are the number of dimensions of the system
    and ik are the number of elements of each dimension.

    Parameters
    ----------
    dims    -- list with the dimension of each variable.
    a         -- list of initial values of the interval for each dimension
    dx        -- list of the discretization step for each dimension.
    """

    d = len(dims)
    r = np.zeros([d] + dims, dtype=np.float64)
    for n, (an, dn) in enumerate(zip(a, dims)):
        rn = an + dx[n] * np.arange(dn)
        # Broadcast to (1, ..., dk, ..., 1)
        dimensions = [1] * n + [dn] + [1] * (d - n - 1)
        rn = rn.reshape(*dimensions)
        r[n, :] = rn
    return r


def fourier_interpolation_vector_1D(f, M, axis=0):
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
    return f * sqrt(M / d[axis])


def fourier_interpolation_vector(f, new_dims):
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
        f = fourier_interpolation_vector_1D(f, dim, axis=i)
    return f


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


def finite_differences_interpolation_vector_2D(f):
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
