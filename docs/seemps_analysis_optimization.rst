.. _analysis_optimization:

*********************
Function Optimization
*********************

This module provides algorithms for finding minima and maxima of functions
encoded as MPS. Unlike eigensolvers in :mod:`seemps.optimization`, which
find ground states of Hamiltonians, these methods locate extrema of scalar
functions represented in the QTT format.

Finding extrema with ``optima_tt``
==================================

The core algorithm is based on the ``optima_tt`` method :cite:`rakhuba2022`, which
performs a probabilistic search through the exponentially large index space of an MPS.
The key insight is that, for a canonicalized MPS, the tensor at the orthogonality
center contains information about which indices contribute most to the state norm.

The algorithm traverses the MPS tensors sequentially, at each step retaining only
the ``num_indices`` most promising index combinations based on their contribution
to the overall amplitude. By keeping a fixed number of candidates at each step,
the computational cost scales linearly with the number of sites rather than
exponentially with the grid size.

The :func:`~seemps.analysis.optimization.optimize_mps` function wraps this algorithm
to find both the minimum and maximum values:

.. code-block:: python

    from seemps.analysis.optimization import optimize_mps
    from seemps.analysis.mesh import RegularInterval
    from seemps.analysis.factories import mps_sin

    # Create an MPS representing sin(x) on [0, 2π]
    n = 10  # 2^10 = 1024 grid points
    interval = RegularInterval(0, 2 * 3.14159, 2**n)
    mps = mps_sin(interval)

    # Find minimum and maximum
    (i_min, y_min), (i_max, y_max) = optimize_mps(mps, num_indices=100)
    print(f"Minimum: {y_min:.6f} at index {i_min}")
    print(f"Maximum: {y_max:.6f} at index {i_max}")

The ``num_indices`` parameter controls the trade-off between accuracy and
computational cost. Larger values increase the probability of finding the
global optimum but require more computation.

Binary search for monotone functions
====================================

For monotone functions (either increasing or decreasing), the
:func:`~seemps.analysis.optimization.binary_search_mps` function provides an
efficient method to find the index where the function crosses a given threshold.
This is useful for computing inverse functions or finding roots of shifted
functions.

.. code-block:: python

    from seemps.analysis.optimization import binary_search_mps

    # Find index where function crosses threshold 0.5
    # Assumes the MPS represents an increasing function
    index = binary_search_mps(mps, threshold=0.5, increasing=True)

The search environments can be precomputed and cached for efficiency when
performing multiple searches on the same MPS with different thresholds.

API reference
=============

.. autosummary::

    ~seemps.analysis.optimization.optimize_mps
    ~seemps.analysis.optimization.optima_tt
    ~seemps.analysis.optimization.binary_search_mps

See also
========

- :doc:`algorithms/dmrg` - DMRG for finding ground states of Hamiltonians
- :doc:`algorithms/gradient_descent` - Gradient descent optimization for MPS
- `Optimization.ipynb <https://github.com/juanjosegarciaripoll/seemps2/blob/main/examples/Optimization.ipynb>`_ - Example notebook
