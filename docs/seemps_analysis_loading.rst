.. _analysis_loading:

****************
Function Loading
****************

The SeeMPS library provides several methods to load univariate and multivariate functions in MPS and MPO structures. In the following, the most important are listed.

Tensorized operations
---------------------
These methods are useful to construct MPS corresponding to domain discretizations, and compose them using tensor products and sums to construct multivariate domains.

.. autosummary::

    ~seemps.analysis.mesh.RegularInterval
    ~seemps.analysis.mesh.ChebyshevInterval
    ~seemps.analysis.factories.mps_interval
    ~seemps.analysis.factories.mps_tensor_product
    ~seemps.analysis.factories.mps_tensor_sum

Tensor cross-interpolation (TT-Cross)
-------------------------------------
These methods are useful to compose MPS or MPO representations of black-box functions using tensor-train cross-interpolation (TT-Cross). See :doc:`algorithms/tt-cross`


Polynomial expansions
---------------------
These methods are useful to compose univariate function on generic initial MPS or MPO and compute MPS approximations of functions. See :doc:`algorithms/polynomials`


Multiscale interpolative constructions
--------------------------------------
These methods are useful to construct polynomial interpolants of univariate functions in MPS using the Lagrange interpolation framework.
See :doc:`algorithms/lagrange`.
