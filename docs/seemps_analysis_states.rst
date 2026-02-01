.. _analysis_states:

*********************************************
Predefined States (MPS) and Tensor Operations
*********************************************

The SeeMPS library provides exact MPS representations of common functions that can
be constructed analytically with low bond dimension. These building blocks enable
the efficient composition of more complex functions through tensor operations.

Elementary functions
====================

Several elementary functions have exact MPS representations with bond dimension
that does not grow with the number of qubits:

- **Exponentials**: :math:`\exp(kx)` can be represented exactly with bond dimension 1,
  since the exponential factorizes over the binary digits of :math:`x`.

- **Trigonometric functions**: :math:`\sin(kx)` and :math:`\cos(kx)` are constructed
  as sums of complex exponentials and have bond dimension 2.

- **Heaviside step function**: :math:`\Theta(x - x_0)` has bond dimension 2 and is
  useful for constructing piecewise functions.

Example usage:

.. code-block:: python

    from seemps.analysis.mesh import RegularInterval
    from seemps.analysis.factories import mps_exponential, mps_sin, mps_cos

    # Define a grid with 2^10 = 1024 points on [0, 2π)
    interval = RegularInterval(0, 2 * 3.14159, 2**10)

    # Create MPS representations
    exp_x = mps_exponential(interval, k=-1.0)  # exp(-x)
    sin_x = mps_sin(interval, k=2.0)           # sin(2x)
    cos_x = mps_cos(interval, k=1.0)           # cos(x)

Interval representations
========================

The :func:`~seemps.analysis.factories.mps_interval` function creates an MPS
representing the coordinate variable :math:`x` itself on a given interval.
This is useful as a building block for polynomial constructions:

.. code-block:: python

    from seemps.analysis.mesh import RegularInterval, ChebyshevInterval
    from seemps.analysis.factories import mps_interval

    # Regular (equispaced) grid
    regular = RegularInterval(-1, 1, 2**10)
    x_regular = mps_interval(regular)

    # Chebyshev (non-uniform) grid
    chebyshev = ChebyshevInterval(-1, 1, 2**10)
    x_chebyshev = mps_interval(chebyshev)

Tensor operations
=================

MPS can be combined to represent multivariate functions using tensor products
and sums:

- **Tensor product**: :func:`~seemps.state.mps_tensor_product` constructs
  :math:`f(x) \cdot g(y)` from separate MPS for :math:`f` and :math:`g`.

- **Tensor sum**: :func:`~seemps.state.mps_tensor_sum` constructs
  :math:`f(x) + g(y)` as a sum of separable terms.

These operations preserve the MPS structure and enable efficient representation
of multivariate functions with separable structure.

API reference
=============

.. autosummary::

    ~seemps.analysis.factories.mps_exponential
    ~seemps.analysis.factories.mps_sin
    ~seemps.analysis.factories.mps_cos
    ~seemps.analysis.factories.mps_affine
    ~seemps.analysis.factories.mps_heaviside
    ~seemps.analysis.factories.mps_interval
    ~seemps.state.mps_tensor_product
    ~seemps.state.mps_tensor_sum
