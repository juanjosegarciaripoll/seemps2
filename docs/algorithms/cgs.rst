.. _alg_cgs:

*****************************
Conjugate gradient (CGS)
*****************************

The conjugate gradient method solves systems of linear equations :math:`A \psi = b`
where the operator :math:`A` (represented as an MPO) is symmetric or Hermitian and
positive-semidefinite. The algorithm iteratively minimizes the residual
:math:`\|A \psi - b\|` until convergence is achieved.

Mathematical formulation
========================

The conjugate gradient method seeks to minimize the quadratic form:

.. math::
    \mathrm{argmin}_{\mathbf{x}} \|A\mathbf{x} - \mathbf{b}\|^2

Given an MPS :math:`\mathbf{x}_k` at iteration :math:`k`, the algorithm computes a search
direction :math:`\mathbf{p}_k` and finds the optimal step size :math:`\alpha` to minimize
the residual along that direction:

.. math::
    \mathbf{x}_{k+1} = \mathbf{x}_k + \alpha \mathbf{p}_k

The step size is computed as:

.. math::
    \alpha = \frac{\|\mathbf{r}_k\|^2}{\langle \mathbf{p}_k | A | \mathbf{p}_k \rangle}

where :math:`\mathbf{r}_k = \mathbf{b} - A\mathbf{x}_k` is the residual. The search
direction is updated using:

.. math::
    \mathbf{p}_{k+1} = \mathbf{r}_{k+1} + \frac{\|\mathbf{r}_{k+1}\|^2}{\|\mathbf{r}_k\|^2} \mathbf{p}_k

MPS implementation
==================

The power of working with the MPS BLAS is that the CGS implementation is almost
indistinguishable from standard implementations using NumPy, except for the implicit
:func:`~seemps.state.simplify` operations that keep the MPS bond dimension in check.

Each iteration involves:

1. Computing :math:`A \mathbf{p}` (MPO-MPS product)
2. Linear combinations of MPS (addition and scalar multiplication)
3. Simplification to control bond dimension growth

The convergence criterion is :math:`\|\mathbf{r}\| < \epsilon \|\mathbf{b}\|` where
:math:`\epsilon` is the user-specified tolerance.

When to use CGS
===============

CGS is best suited for:

- **Hermitian positive-definite operators**: The method requires :math:`A` to be
  symmetric/Hermitian and positive-semidefinite for guaranteed convergence
- **Well-conditioned systems**: Convergence rate depends on the condition number
- **Moderate accuracy requirements**: For very high precision, consider :doc:`gmres` or :doc:`dmrg_solve`

For non-Hermitian operators, use :doc:`bicgs` or :doc:`gmres` instead.

Example
=======

The following example solves a Poisson-like equation :math:`(I - \partial_x^2) \psi = b`
using the CGS solver with a finite-differences Laplacian:

.. code-block:: python

    import numpy as np
    from seemps.state import product_state
    from seemps.operators.projectors import identity_mpo
    from seemps.analysis.mesh import RegularInterval
    from seemps.analysis.derivatives import finite_differences_mpo
    from seemps.solve import cgs_solve

    # Define the interval and discretization
    n = 10  # qubits -> 2^10 = 1024 grid points
    interval = RegularInterval(-1, 1, 2**n)

    # Create the operator A = I - d²/dx²
    laplacian = finite_differences_mpo(order=2, interval=interval, periodic=True)
    identity = identity_mpo([2] * n)
    A = (identity - laplacian).join()

    # Right-hand side: a simple product state
    b = product_state(np.array([1, 0]), n)

    # Solve A @ x = b
    x, residual = cgs_solve(A, b, tolerance=1e-8)
    print(f"Residual: {residual}")

.. autosummary::

    ~seemps.solve.cgs_solve

See also
========

- :doc:`bicgs` - Biconjugate gradient stabilized method for non-Hermitian systems
- :doc:`gmres` - Generalized minimal residual method using Krylov subspaces
- :doc:`dmrg_solve` - DMRG-based solver with local tensor optimization






