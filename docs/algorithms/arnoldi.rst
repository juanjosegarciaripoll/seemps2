.. _alg_arnoldi:

*******************************
Restarted Arnoldi iteration
*******************************

This is a Krylov method whose update rule expands the function in a Krylov basis
:math:`\mathcal{K}_L = \mathrm{lin}\{\ket{\psi_k}, H\ket{\psi_{k}},\ldots,H^{L-1}\ket{\psi_{k}}\}`,

.. math::
    \psi_{k+1} = \sum_{i=0}^{L-1} c_i H^i \psi_k.

This Arnoldi method estimates the energy functional using two matrices, `A` and `N`---included due to MPS finite-precision---, containing
the matrix elements of `H` and the identity computed in the Krylov basis

.. math::
    E[\chi] = E(\boldsymbol{v}) = \frac{\boldsymbol{v}^\dagger A \boldsymbol{v}}{\boldsymbol{v}^\dagger N \boldsymbol{v}}.

This is the cost function for the optimization, whose critical points

.. math::
    \frac{\delta E}{\delta \boldsymbol{v}^*} = \frac{1}{\boldsymbol{v}^\dagger N \boldsymbol{v}}\left(A \boldsymbol{v} - E(\boldsymbol{v}) N\boldsymbol{v}\right) = 0.

This leads to the generalized eigenvalue equation

.. math::
  A \boldsymbol{v} = \lambda N \boldsymbol{v},

where the minimum eigenvalue :math:`\lambda = E(\boldsymbol{v})` gives the optimal energy for the :math:`k`-th step, and the associated direction
:math:`\mathbf{v}` provides the steepest descent on the plane.

This procedure could be enhanced with additional approaches. We introduce a specific restart method,
which halts the expansion of the Krylov basis when either the maximum desired size :math:`n_v` is attained, or when the condition number of
the scalar product matrix :math:`N` exceeds a certain threshold. This occurs when there is a risk of adding a vector that is linearly dependent
on the previous ones. In such a scenario, we can solve the generalized eigenvalue problem to obtain
the next best estimate in a smaller basis. Another approach to enhance the convergence of eigenvalues is to extrapolate the next vector
based on previous estimates, using the formula :math:`|{\xi_{k+1}}\rangle=(1-\gamma)|{\psi_{k+1}}\rangle+|{\psi_k}\rangle`, with the
memory factor :math:`\gamma=-0.75`.


The rule in :ref:`alg_descent` is a particular case of the Arnoldi iteration with a Krylov basis with  :math:`L=2`.

Ref. :cite:t:`GarciaMolina2024` presents this algorithm and its implementation for global optimization problems. It is also suitable for evolution problems.

Time evolution
==============

The Arnoldi method can also be used for time evolution by computing the action of the
matrix exponential :math:`\exp(-i H \delta t)` on the state. Instead of solving an eigenvalue
problem, the Krylov basis is used to approximate the exponential:

.. math::
    \psi(t + \delta t) = \exp(-i H \delta t) \psi(t) \approx \sum_{i=0}^{L-1} c_i H^i \psi(t)

where the coefficients :math:`c_i` are obtained by projecting the exponential onto the
Krylov subspace. This approach is particularly effective when the Hamiltonian has a
large spectral range, as the Krylov subspace naturally adapts to capture the relevant
dynamics.

The Arnoldi time evolution method preserves unitarity within the Krylov subspace and
can be more efficient than Taylor or Chebyshev expansions for certain problems,
especially when high accuracy is required over short time intervals.

.. autosummary::

    ~seemps.optimization.arnoldi
    ~seemps.evolution.arnoldi

See also
========

- :doc:`gradient_descent` - A special case of Arnoldi with a Krylov basis of size 2
- :doc:`dmrg` - An alternative optimization algorithm based on local tensor updates
- :doc:`runge_kutta` - Explicit time evolution methods
- :doc:`tdvp` - Time-dependent variational principle
- :doc:`crank_nicolson` - Implicit time evolution methods






