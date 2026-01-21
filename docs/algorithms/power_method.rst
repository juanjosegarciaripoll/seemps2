.. _alg_power_method:

************
Power method
************

The power method, or power iteration, is an algorithm for approximating the
largest eigenvalue of an operator and its corresponding eigenvector. Starting
from an initial MPS :math:`\psi_0`, each iteration applies the operator and
normalizes the result:

.. math::
    \psi_{k+1} = \frac{H \psi_k}{\| H \psi_k \|}

Convergence to the dominant eigenvector requires that the largest eigenvalue
is non-degenerate and that the initial state has nonzero overlap with it. The
convergence rate depends on the spectral gap between the first two eigenvalues,
so a larger gap generally yields faster convergence.

Inverse power method
====================

To target the smallest eigenvalue of an operator, the method can be combined
with operator inversion. Instead of applying :math:`H` directly, we solve
:math:`H \psi_{k+1} = \psi_k` at each step. This approach effectively applies
:math:`H^{-1}` to the state, causing the eigenvector with the smallest
eigenvalue to become dominant.

For a shifted operator :math:`(H - \epsilon)`, the inverse power method converges
to the eigenvalue closest to :math:`\epsilon`. In SeeMPS, each iteration of the
inverse power method involves solving a linear system using the conjugate gradient
solver, which adds computational cost compared to the standard power iteration
but enables targeting of the ground state.

.. autosummary::

   ~seemps.optimization.power_method

See also
========

- :doc:`gradient_descent` - Optimization using the energy gradient
- :doc:`arnoldi` - Krylov-based optimization with faster convergence
- :doc:`dmrg` - Local tensor optimization algorithm
