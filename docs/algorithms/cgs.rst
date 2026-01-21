.. _alg_cgs:

*******************************
Conjugate gradient descent
*******************************

The conjugate gradient method solves systems of linear equations :math:`A \psi = b`
where the operator :math:`A` (represented as an MPO) is positive-semidefinite.
The algorithm iteratively minimizes the residual :math:`\|A \psi - b\|` until
convergence is achieved.

Given an MPS :math:`\psi_k` at iteration :math:`k`, the algorithm computes a search
direction :math:`p_k` and finds the optimal step size :math:`\alpha` to minimize
the residual along that direction:

.. math::
    \psi_{k+1} = \psi_k + \alpha p_k

The conjugate gradient method is particularly efficient because it generates
orthogonal search directions, guaranteeing convergence in at most :math:`n`
iterations for an :math:`n`-dimensional problem (though in practice, with MPS
truncation, approximate convergence is typically achieved much sooner).

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
    from seemps.solve import cgs

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
    x, residual = cgs(A, b, tolerance=1e-8)
    print(f"Residual: {residual}")

.. autosummary::

    ~seemps.solve.cgs

See also
========

- :func:`~seemps.solve.bicgs_solve` - Biconjugate gradient stabilized method
- :func:`~seemps.solve.gmres_solve` - Generalized minimal residual method
- :func:`~seemps.solve.dmrg_solve` - DMRG-based solver for linear systems






