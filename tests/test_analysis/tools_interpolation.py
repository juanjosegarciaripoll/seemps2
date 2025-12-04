from math import sqrt
import numpy as np


def gaussian(x):
    return np.exp(-(x * x))


def gaussian_tensor(r):
    """Constructs a Gaussian function defined on the given grid.

    Parameters
    ----------
    r: grid on which the Gaussian is defined
        In this function r[i,...] are the i-th coordinates of the
        N-dimensional grid.

    Returns
    -------
    np.ndarray: Gaussian on grid r.
    """
    n_dims = r.shape[0]
    exponent = sum(r[i, :] ** 2 for i in range(n_dims))
    f = np.exp(-exponent / 2)
    return f / np.linalg.norm(f)


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
    f = np.fft.fft(f, norm="ortho", axis=axis)
    dims = list(f.shape)
    dims[axis] = M - d[axis]
    filler = np.zeros(dims, dtype=type(f[0]))
    if axis == 1:
        filler = filler.T
    f = np.insert(f, d[axis] // 2, filler, axis=axis)
    f = np.fft.ifft(f, norm="ortho", axis=axis)
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
