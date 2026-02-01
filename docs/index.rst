.. SeeMPS documentation master file, created by
   sphinx-quickstart on Sun Sep  1 18:02:11 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to SeeMPS's documentation!
==================================

SeeMPS is a Python library dedicated to implementing tensor network algorithms
based on the Matrix Product States (MPS) and Quantized Tensor Train (QTT) formalisms.
SeeMPS is implemented as a complete finite-precision linear algebra package where
exponentially large vector spaces are compressed using the MPS/TT formalism, enabling
both low-level operations---such as vector (MPS) addition, linear transformations and
Hadamard products---as well as high-level algorithms---approximation of linear equations,
eigenvalue and eigenstate computations, and exponentially efficient Fourier transforms.

This library can be used for traditional quantum many-body physics applications
and also for quantum-inspired numerical analysis problems, such as solving PDEs,
interpolating and integrating multidimensional functions, sampling multivariate
probability distributions, etc.

Features
--------

**MPS-BLAS**: Low-level linear algebra operations

- Vector representation using MPS/TT with controlled truncation
- Matrix representation using MPO
- Vector addition, scaling, inner products
- Matrix-vector products and Hadamard (element-wise) products
- Tensor products and simplification algorithms

**MPS-LAPACK**: High-level linear algebra algorithms

- Eigenvalue search: Power method, Arnoldi, DMRG
- Linear system solvers: CGS, BiCGS, GMRES, DMRG
- Quantum Fourier Transform as MPO

**Functional analysis**: Quantum-inspired numerical methods

- Function loading: direct constructions, polynomial expansions, tensor cross-interpolation (TCI)
- Differentiation: finite differences, Fourier differentiation, HDAF
- Integration: Newton-Cotes, Clenshaw-Curtis quadratures
- Interpolation: finite differences, Fourier methods
- PDE solvers for eigenvalue and source problems
- Time evolution: explicit Runge-Kutta, implicit Crank-Nicolson/Radau, TDVP

**Quantum many-body physics and computing**

- Hamiltonian construction using interaction graphs
- Ground state search with DMRG
- Time evolution with TEBD
- Parameterized quantum circuits

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   getting_started
   seemps_objects
   algorithms/index
   seemps_register
   seemps_analysis
   seemps_hdf5
   seemps_tools
   contributing
   seemps_examples
   api/reference

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
* :ref:`classes`