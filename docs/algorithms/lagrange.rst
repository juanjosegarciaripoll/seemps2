.. currentmodule:: seemps.analysis.lagrange


.. _alg_lagrange:

**************************************
Multiscale interpolative constructions
**************************************

The MPS representation of polynomial interpolants can be efficiently constructed using *multiscale interpolative constructions*, following Lindsey's idea (see Ref. https://arxiv.org/pdf/2311.12554). On the current formulation, these methods are based on the Lagrange interpolation framework, interpolating the function on Chebyshev-Lobatto nodes. 

The SeeMPS library implements these interpolative constructions for multivariate functions. Multiscale resolution features have yet not been implemented. The method requires sampling a tensor of coefficients from the input function, which becomes computationally intractable for high-dimensional problems. Hence, current implementations provide an efficient tool for encoding smooth, low-dimensional functions in MPS form with high efficiency and accuracy.

Several algorithmic variants follow. Basic construct using :func:`~mps_lagrange_chebyshev_basic` assemble three distinct tensor cores, two of which are placed at the edges of the MPS construction and one is copied and placed in the bulk. Only the left-most core is function-dependent, while all the rest only depend on the interpolation order and can therefore be precomputed and reused. Their combination forms a Lagrange-Chebyshev interpolant across two Chebyshev-Lobatto grids, which presents spectral approximation convergence rates for sufficiently smooth functions. However, this approach tends to overestimate the required bond dimension of the interpolant, requiring a large-scale final truncation.

The method's performance can be enhanced through rank-revealing optimizations using the SVD decomposition, avoiding the need for a large-scale final simplification. This is implemented in the :func:`~mps_lagrange_chebyshev_rr` routine. Moreover, local interpolation can be performed, resulting in highly sparse MPS cores and enhancing performance. This is implemented in :func:`~mps_lagrange_chebyshev_lrr`.

.. autosummary::

    ~mps_lagrange_chebyshev_basic
    ~mps_lagrange_chebyshev_rr
    ~mps_lagrange_chebyshev_lrr