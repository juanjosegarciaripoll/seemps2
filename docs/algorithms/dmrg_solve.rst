.. _alg_dmrg_solve:

***************************
DMRG linear solver
***************************

The DMRG algorithm can be adapted to solve linear systems :math:`A \mathbf{x} = \mathbf{b}`
by reformulating the problem as a sequence of local optimizations over pairs of
neighboring tensors. This approach is philosophically equivalent to the vector
correction method :cite:p:`kuhner1999` and leverages the same sweep-based structure
used in DMRG for eigenvalue problems.

Mathematical formulation
========================

Given an MPO :math:`A` and an MPS :math:`\mathbf{b}`, we seek an MPS
:math:`\mathbf{x}` that solves:

.. math::
    A \mathbf{x} = \mathbf{b}

The DMRG approach reformulates this as:

.. math::
    \mathrm{argmin}_{\mathbf{x}} \|A\mathbf{x} - \mathbf{b}\|^2

Local tensor optimization
=========================

Using the quadratic form :math:`\langle \mathbf{x} | A^\dagger A | \mathbf{x} \rangle`
and antilinear form :math:`\langle \mathbf{x} | A^\dagger | \mathbf{b} \rangle`
abstractions (see :doc:`dmrg`), the complete system is projected onto pairs of
neighboring tensors :math:`(n, n+1)`:

.. math::
    U_n \mathbf{a}_n = \mathbf{b}_n

where :math:`U_n` is the effective operator acting on the two-site tensor
:math:`\mathbf{a}_n = A^{[n]} A^{[n+1]}`, and :math:`\mathbf{b}_n` is the
projected right-hand side.

This local system of equations is solved using standard sparse linear algebra
methods from SciPy (conjugate gradient, BiCG, or BiCGSTAB), yielding new estimates
for the tensor pair. The tensors are then factorized via SVD and truncated according
to the specified strategy. By repeating this step many times, back and forth along
the tensor train, the algorithm converges to the optimal solution within the given
constraints of tolerance, truncation limits, and bond dimension.

Local solver options
====================

The ``method`` parameter selects the local solver:

- ``"cg"``: Conjugate gradient (for Hermitian positive-definite effective operators)
- ``"bicg"``: Biconjugate gradient
- ``"bicgstab"``: Biconjugate gradient stabilized (default, most robust)

When to use DMRG-solve
======================

DMRG-solve is best suited for:

- **Problems requiring controlled bond dimension**: The two-site update with SVD
  truncation naturally controls the MPS complexity
- **Ill-conditioned systems**: Local optimization can be more stable than global
  Krylov methods for difficult problems
- **High accuracy requirements**: The variational nature ensures optimal approximation
  within the given bond dimension constraints

Compared to Krylov methods (CGS, BiCGS, GMRES):

- DMRG-solve maintains explicit control over bond dimension at each step
- Each sweep is more expensive (requires solving :math:`L-1` local systems)
- May converge faster for problems where the solution has low bond dimension
- Better suited when the truncation error is the dominant source of error

Example
=======

.. code-block:: python

    import numpy as np
    from seemps.state import random_uniform_mps
    from seemps.operators import MPO
    from seemps.solve import dmrg_solve

    # Create your operator and right-hand side
    n = 10
    # ... construct MPO A and MPS b ...

    # Solve A @ x = b using DMRG
    x, residual = dmrg_solve(
        A, b,
        maxiter=20,       # Maximum number of sweeps
        rtol=1e-6,        # Relative tolerance
        method="bicgstab" # Local solver
    )
    print(f"Final residual: {residual}")

.. autosummary::

    ~seemps.solve.dmrg_solve

See also
========

- :doc:`dmrg` - DMRG for eigenvalue problems
- :doc:`cgs` - Conjugate gradient method (global iteration)
- :doc:`bicgs` - Biconjugate gradient stabilized method (global iteration)
- :doc:`gmres` - Generalized minimal residual method (global iteration)
