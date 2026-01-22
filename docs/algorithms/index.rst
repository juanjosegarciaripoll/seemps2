*******************
Index of algorithms
*******************

This section documents the algorithms implemented in SeeMPS, organized by category.

Tensor operations
=================

Low-level operations for decomposing and manipulating tensors within MPS.

.. toctree::
   :maxdepth: 1

   tensor_schmidt
   tensor_split
   tensor_canonical
   tensor_to_mps
   tensor_update
   mps_simplification

Optimization and eigensolvers
=============================

Algorithms for finding ground states and solving eigenvalue problems.

.. toctree::
   :maxdepth: 1

   gradient_descent
   power_method
   arnoldi
   dmrg

Linear system solvers
=====================

Methods for solving systems of linear equations :math:`A \psi = b`.

.. toctree::
   :maxdepth: 1

   cgs

Time evolution
==============

Methods for evolving quantum states under Hamiltonian dynamics.

.. toctree::
   :maxdepth: 1

   arnoldi
   runge_kutta
   crank_nicolson
   split_step
   tebd_evolution
   tdvp

Fourier transform
=================

The Quantum Fourier Transform for MPS representations.

.. toctree::
   :maxdepth: 1

   qft

Function approximation
======================

Algorithms for constructing MPS representations of functions.

.. toctree::
   :maxdepth: 1

   polynomials
   tt-cross
   lagrange
   sketching
   comptree
