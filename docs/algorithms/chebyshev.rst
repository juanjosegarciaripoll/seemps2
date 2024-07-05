.. _analysis_chebyshev:

***********************
Chebyshev Approximation
***********************

Matrix product states (MPS) and operators (MPO) can be expanded on the basis of 
Chebyshev polynomials, allowing to approximate and compose functions.

In principle, this method can be performed for arbitrary multivariate functions.
However, its cost scales exponentially with the dimension of the function, and
the method converges efficiently only for highly-differentiable functions. 

For these reasons, currently the SeeMPS library contains methods to perform Chebyshev
expansions of univariate functions. The method works for both MPS and MPO initial states,
and converges efficiently for analytical or highly-differentiable functions.

Computation of the expansion coefficients
=========================================
The expansion of an univariate function on the basis of Chebyshev polynomials is of the form

.. math::
    p_d(x)=\sum_{k=0}^d c_k T_k(x),

where

.. math::
    T_{k+1}(x)  =2 x T_k(x)-T_{k-1}(x),\;k\geq 1

are the Chebyshev polynomials of order :math:`c_k`.

The coefficients :math:`c_k` contain the information of the function. These coefficients
can be given by the projection of the function on the basis, or by interpolation on a collection
of nodes. The library presents the method :func:`~seemps.analysis.chebyshev.projection_coefficients` for the former, and
:func:`~seemps.analysis.chebyshev.interpolation_coefficients` for the latter, both on Chebyshev-Gauss or Chebyshev-Lobatto nodes.
These methods only depend on the univariate function, its domain of definition, and optionally the 
chosen interpolation order. This order can be estimated to machine precision using the :func:`~seemps.analysis.chebyshev.estimate_order`
routine.

Expansion in the Chebyshev basis
================================
Once the expansion coefficients are computed for the function, the series can be applied on a
generic initial condition. This condition can be an MPS or MPO. These expansions can be respectively
performed using the methods :func:`~seemps.analysis.chebyshev.cheb2mps` and :func:`~seemps.analysis.chebyshev.cheb2mpo`. 

The initial conditions must have a support on the canonical Chebyshev interval :math:`[-1, 1]`. 
For MPS, this initial support is to be understood as the minimum and maximum values of the corresponding
vector, while for MPO it is to be understood as the smallest and largest eigenvalues. 
If the initial condition has a larger support, it must be rescaled using an affine transformation. 
This is performed by default by the main algorithm assuming that the initial condition is defined on 
the domain of definition of the expansion coefficients.

The standard evaluation of the partial sum is based on constructing the Chebyshev polynomials
:math:`T_k` using the recurrence relation, and then performing the linear combination with :math:`c_k` weights.
This can be performed by setting the flag ``clenshaw`` to ``False`` in :func:`~seemps.analysis.chebyshev.cheb2mps` or :func:`~seemps.analysis.chebyshev.cheb2mpo`.

However, there is a more efficient evaluation routine based on Clenshaw's evaluation method. This
procedure avoids computing the intermediate Chebyshev coefficients, and shows a more robust and efficient
performance. It is set by default through the flag ``clenshaw`` set to ``True``. However, this method is very
susceptible to an overestimation of the interpolation order, showing a degrading performance in that case. 

Constructing the initial condition
==================================
This method requires an initial condition, either MPS or MPO, to perform function composition. This
initial condition must be passed to the argument ``initial_mps`` or ``initial_mpo``. However, the initial
conditions for the case of function "loading", which are given by discretized domains, can be built
automatically by passing an :class:`~seemps.analysis.mesh.Interval` object to the ``domain`` argument.

These discretized domains can be alternatively built by creating an :class:`~seemps.analysis.mesh.Interval` object, such as a :class:`~seemps.analysis.mesh.RegularInterval`
or :class:`~seemps.analysis.mesh.ChebyshevInterval`, and constructing the corresponding MPS with the routine :py:func:`~seemps.analysis.factories.mps_interval`.

Multivariate functions
======================

This method enables the construction of multivariate functions by composing functions on multivariate
initial conditions. These conditions can be constructed by tensorized products or sums on univariate states.
These operations can be performed with the methods :func:`~seemps.analysis.factories.mps_tensor_product` and/or :func:`~seemps.analysis.factories.mps_tensor_sum`.
These initial conditions may have a growing support, so they must be rescaled appropriately to fit in :math:`[-1, 1]`.

An example on how to use these functions is shown in `Chebyshev.ipynb <https://github.com/juanjosegarciaripoll/seemps2/blob/main/examples/Chebyshev.ipynb>`_.

.. autosummary::
    :toctree: generated/

    ~seemps.analysis.chebyshev.projection_coefficients
    ~seemps.analysis.chebyshev.interpolation_coefficients
    ~seemps.analysis.chebyshev.estimate_order
    ~seemps.analysis.chebyshev.cheb2mps
    ~seemps.analysis.chebyshev.cheb2mpo