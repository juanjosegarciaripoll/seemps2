.. _seemps_analysis:

***********************************
Quantum-inspired numerical analysis
***********************************

SeeMPS extends the MPS/TT formalism to function representation and numerical analysis.
Multidimensional functions discretized on exponentially large grids can be encoded using
quantized tensor trains (QTT), exploiting the fact that smooth functions often have
bounded entanglement when represented in this binary encoding.

Consider a one-dimensional function :math:`f(x)` defined over an interval :math:`[a,b)`
sampled on a uniform grid of :math:`2^n` points. The vector of function values
:math:`v_i = f(x_i)` can be tensorized using :math:`n` binary indices (qubits) and
compressed as an MPS. Provided the bond dimensions remain bounded, the storage cost
scales as :math:`\mathcal{O}(n \chi^2)` compared to :math:`\mathcal{O}(2^n)` for the
full grid representation.

The library provides tools for:

- **Function loading**: Multiple strategies to construct MPS representations of functions,
  including direct analytic constructions, polynomial expansions, and tensor cross-interpolation.

- **Differentiation**: Finite differences, Fourier differentiation, and HDAF methods
  implemented as MPO operators.

- **Integration**: Newton-Cotes and Clenshaw-Curtis quadrature rules that reduce to
  inner products between MPS.

- **Interpolation**: Finite differences and Fourier-based interpolation to estimate
  function values between grid points.

- **Optimization**: Finding minima and maxima of functions encoded as MPS, as well as
  obtaining functions that solve optimization problems, such as eigenvalue equations.

These tools enable efficient solvers for ordinary and partial differential equations
in high dimensions, both for eigenvalue problems and source problems with Dirichlet
or periodic boundary conditions.

.. toctree::
   :maxdepth: 1

   seemps_analysis_states
   seemps_analysis_operators
   seemps_analysis_spaces
   seemps_analysis_loading
   seemps_analysis_differentiation
   seemps_analysis_integration
   seemps_analysis_interpolation
   seemps_analysis_pde
   seemps_analysis_optimization
