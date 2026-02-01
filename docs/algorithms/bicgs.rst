.. _alg_bicgs:

******************************************
Biconjugate gradient stabilized (BiCGSTAB)
******************************************

The biconjugate gradient stabilized method (BiCGSTAB) is an iterative algorithm
for solving systems of linear equations :math:`A \mathbf{x} = \mathbf{b}` where
the operator :math:`A` may be non-symmetric or non-Hermitian. It is a variant of
the biconjugate gradient method with improved numerical stability.

Mathematical formulation
========================

BiCGSTAB seeks to solve:

.. math::
    A \mathbf{x} = \mathbf{b}

where :math:`A` is represented as an MPO and :math:`\mathbf{b}` is an MPS. Unlike
the standard conjugate gradient method, BiCGSTAB does not require :math:`A` to be
symmetric or positive-definite, making it suitable for a broader class of problems.

The algorithm maintains two sequences: the residual :math:`\mathbf{r}_k` and a
shadow residual :math:`\mathbf{r}_0^*` (chosen as the initial residual). Each
iteration involves:

1. Compute :math:`\mathbf{v} = A \mathbf{p}`
2. Calculate :math:`\alpha = \rho / \langle \mathbf{r}_0^*, \mathbf{v} \rangle`
3. Compute intermediate solution :math:`\mathbf{h} = \mathbf{x} + \alpha \mathbf{p}`
4. Compute :math:`\mathbf{s} = \mathbf{r} - \alpha \mathbf{v}`
5. Apply stabilization: :math:`\mathbf{t} = A \mathbf{s}`, then :math:`\omega = \langle \mathbf{t}, \mathbf{s} \rangle / \|\mathbf{t}\|^2`
6. Update solution: :math:`\mathbf{x} = \mathbf{h} + \omega \mathbf{s}`
7. Update residual: :math:`\mathbf{r} = \mathbf{s} - \omega \mathbf{t}`
8. Update search direction with :math:`\beta = (\rho_{new}/\rho) \cdot (\alpha/\omega)`

The stabilization step (computing :math:`\omega`) is what distinguishes BiCGSTAB from
the standard biconjugate gradient method and provides improved convergence behavior.

MPS implementation
==================

In the MPS formulation, each iteration requires:

- Two MPO-MPS products (:math:`A \mathbf{p}` and :math:`A \mathbf{s}`)
- Multiple MPS scalar products
- Linear combinations of MPS
- Simplification steps to control bond dimension

The convergence criterion checks whether :math:`\|\mathbf{r}\| < \max(\text{rtol} \cdot \|\mathbf{b}\|, \text{atol})`.

When to use BiCGSTAB
====================

BiCGSTAB is best suited for:

- **Non-Hermitian operators**: Unlike CGS, BiCGSTAB handles general linear operators
- **Moderate-sized problems**: Each iteration requires two MPO-MPS products
- **Problems where CGS fails**: BiCGSTAB often succeeds when standard CG diverges

For very ill-conditioned systems or when high accuracy is needed, consider
:doc:`gmres` or :doc:`dmrg_solve`.

Example
=======

.. code-block:: python

    import numpy as np
    from seemps.state import random_uniform_mps
    from seemps.operators import MPO
    from seemps.solve import bicgs_solve

    # Create a non-symmetric operator (example: advection-diffusion)
    n = 8
    # ... construct your MPO A ...

    # Right-hand side
    b = random_uniform_mps(2, n, D=4)

    # Solve A @ x = b
    x, residual = bicgs_solve(A, b, rtol=1e-6, maxiter=100)
    print(f"Final residual: {residual}")

.. autosummary::

    ~seemps.solve.bicgs_solve

See also
========

- :doc:`cgs` - Conjugate gradient method for Hermitian positive-definite systems
- :doc:`gmres` - Generalized minimal residual method using Krylov subspaces
- :doc:`dmrg_solve` - DMRG-based solver with local tensor optimization
