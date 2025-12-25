.. currentmodule:: seemps.analysis.lagrange


.. _alg_lagrange:

**************************************
Multiscale interpolative constructions
**************************************

The MPS representation of an univariate polynomial interpolant can be efficiently constructed using multiscale interpolative constructions, following Lindsey's method (see Ref. https://arxiv.org/pdf/2311.12554). These methods leverage the Lagrange interpolation framework, specifically utilizing Chebyshev-Lobatto nodes. The SeeMPS library implements the interpolative constructs for the univariate scenario. The extensions to the multivariate case have not been implemented yet.

In the following, we provide an overview of the method. Essentially, the basic construct :func:`~lagrange_basic` implements an universal construction by assembling three distinct tensor cores corresponding to the left-most, bulk and right-most edges. The left-most core is function-dependent, while all the rest only depend on the interpolation order. After assembling, these cores are combined to form a Lagrange interpolant across two Chebyshev-Lobatto grids, one for each half of the domain. This approach severely overestimates the required bond dimension of the interpolant, requiring a large-scale final simplification.

This method can be enhanced by performing rank-revealing optimizations using SVD decomposition. The corresponding method, :func:`~lagrange_rank_revealing`, avoids the need for a large-scale final simplification.

Finally, the interpolative constructions can be developed on top of a local Lagrange interpolation framework, :func:`~lagrange_local_rank_revealing`. Then, the tensor cores become sparse, largely improving the overall efficiency of the algorithm.

.. autosummary::

    ~mps_lagrange_chebyshev_basic
    ~mps_lagrange_chebyshev_rr
    ~mps_lagrange_chebyshev_lrr