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
    ~seemps.state.mps_tensor_product
    ~seemps.state.mps_tensor_sum

Tensor cross-interpolation (TCI)
--------------------------------
These methods are useful to compose MPS or MPO representations of black-box functions using tensor cross-interpolation (TCI). See :doc:`algorithms/tt-cross`.


Polynomial expansions
---------------------
These methods are useful to compose univariate function on generic initial MPS or MPO and compute MPS approximations of functions. See :doc:`algorithms/polynomials`.


Multiscale interpolative constructions
--------------------------------------
These methods are useful to construct polynomial interpolants of low-dimensional functions in MPS using the Chebyshev-Lagrange interpolation framework.
See :doc:`algorithms/lagrange`.

Sketching methods
-----------------
These methods are useful to construct high-dimensional densities or other black-box non-normalized functions from a collection of samples defining the region of interest. See :doc:`algorithms/sketching`.

Computation-tree methods
------------------------
These methods are useful to construct procedurally defined functions and functions with sharp features, where polynomial expansions and tensor cross interpolation may suffer from slow convergence or Gibbs-type artifacts. See :doc:`algorithms/comptree`.