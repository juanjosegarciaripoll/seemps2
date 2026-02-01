.. _seemps_analysis_pde:

*************
PDE solutions
*************

SeeMPS can be used to solve both eigenvalue and source problems for partial
differential equations with Dirichlet zero or periodic boundary conditions.
This page explains how to combine the differentiation operators with the
eigensolvers and linear solvers to address these problems.

Problem types
=============

Eigenvalue problems
-------------------

The first family of problems consists of equations that can be brought into the form:

.. math::
    \left[D(\partial_x) + V(\mathbf{x})\right] f(\mathbf{x}) = E f(\mathbf{x})

where :math:`D(\partial_x)` is a differential operator (e.g., the Laplacian) and
:math:`V(\mathbf{x})` is a potential function. Examples include:

- **Quantum harmonic oscillator**: :math:`\left[-\frac{1}{2}\partial_x^2 + \frac{1}{2}x^2\right]\psi = E\psi`
- **Particle in a box**: :math:`-\frac{1}{2}\partial_x^2 \psi = E\psi` with zero boundary conditions
- **Helmholtz equation**: :math:`(\nabla^2 + k^2) u = 0`

To solve these problems:

1. Construct an MPO for the operator :math:`H = D(\partial_x) + V(\mathbf{x})`
2. Use an eigenvalue solver (:doc:`algorithms/dmrg`, :doc:`algorithms/arnoldi`, or :doc:`algorithms/power_method`)

Source problems
---------------

The second family consists of inhomogeneous PDEs with a source term:

.. math::
    \left[D(\partial_x) + V(\mathbf{x})\right] f(\mathbf{x}) = g(\mathbf{x})

Examples include:

- **Poisson equation**: :math:`\nabla^2 u = \rho`
- **Screened Poisson**: :math:`(\nabla^2 - \lambda^2) u = f`
- **Steady-state heat equation**: :math:`\nabla^2 T = Q`

To solve these problems:

1. Construct an MPO for the operator :math:`H = D(\partial_x) + V(\mathbf{x})`
2. Encode the source term :math:`g(\mathbf{x})` as an MPS
3. Use a linear solver (:doc:`algorithms/cgs`, :doc:`algorithms/gmres`, or :doc:`algorithms/dmrg_solve`)

Constructing the operator MPO
=============================

The operator MPO is constructed by combining differentiation operators with potential
terms. SeeMPS provides three methods for differential operators (see
:doc:`seemps_analysis_differentiation`): finite differences, Fourier methods, and HDAFs.

MPOs support standard arithmetic operations, so operators can be combined directly::

    from seemps.analysis.derivatives import finite_differences_mpo

    # Laplacian
    laplacian = finite_differences_mpo(order=2, interval=interval, periodic=True)

    # Combined Hamiltonian H = I - 0.5 * Laplacian
    H = (identity - 0.5 * laplacian).join()

Example: Eigenvalue problem
===========================

The following example finds the ground state of a quantum harmonic oscillator:

.. code-block:: python

    import numpy as np
    from seemps.state import random_uniform_mps
    from seemps.analysis.mesh import RegularInterval
    from seemps.analysis.factories import mps_interval
    from seemps.analysis.derivatives import finite_differences_mpo
    from seemps.optimization import dmrg

    # Grid setup
    n = 10  # 2^10 = 1024 points
    interval = RegularInterval(-5, 5, 2**n)

    # Kinetic energy: -0.5 * d²/dx²
    T = finite_differences_mpo(order=2, interval=interval, periodic=False)

    # Potential energy: 0.5 * x² (as diagonal MPO)
    # V_mpo = ...

    # Hamiltonian: H = -0.5 * T + V
    # H = (-0.5 * T + V_mpo).join()

    # Initial guess
    guess = random_uniform_mps(2, n, D=10)

    # Solve eigenvalue problem
    # result = dmrg(H, guess, maxiter=20, tolerance=1e-10)

Example: Source problem
=======================

The following example solves a Poisson-like equation:

.. code-block:: python

    import numpy as np
    from seemps.state import product_state
    from seemps.operators.projectors import identity_mpo
    from seemps.analysis.mesh import RegularInterval
    from seemps.analysis.derivatives import finite_differences_mpo
    from seemps.solve import cgs_solve

    # Grid setup
    n = 10
    interval = RegularInterval(-1, 1, 2**n)

    # Operator: I - d²/dx² (shifted Laplacian for positive-definiteness)
    laplacian = finite_differences_mpo(order=2, interval=interval, periodic=True)
    identity = identity_mpo([2] * n)
    A = (identity - laplacian).join()

    # Source term (example: constant)
    b = product_state(np.array([1, 0]), n)

    # Solve
    x, residual = cgs_solve(A, b, tolerance=1e-8)

Boundary conditions
===================

The differential operators in SeeMPS support two types of boundary conditions:

- **Periodic**: Function values wrap around at the boundaries
- **Dirichlet zero**: Function vanishes at the boundaries (open boundary conditions)

The boundary condition is specified via the ``periodic`` parameter in the
differentiation functions. Other boundary conditions (Neumann, Robin, etc.)
are not currently supported directly but can sometimes be handled through
problem reformulation.

Multidimensional problems
=========================

For multidimensional PDEs, the MPS encodes the function on a tensor product
grid. Partial derivatives act on subsets of qubits corresponding to each
spatial dimension. The total operator is constructed as a sum of terms, each
acting on the appropriate qubits.

See also
========

- :doc:`seemps_analysis_differentiation` - Differentiation operators
- :doc:`algorithms/dmrg` - DMRG eigenvalue solver
- :doc:`algorithms/cgs` - Conjugate gradient linear solver
- :doc:`algorithms/gmres` - GMRES linear solver
