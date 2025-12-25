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

where the minimum eigenvalue :math:`\lambda = E(\boldsymbol{v})` gives the optimal energy for the $k$-th step, and the associated direction
:math:`\mathbf{v}` provides the steepest descent on the plane.

This procedure could be enhanced with additional approaches. We introdce a specific restart method,
which halts the expansion of the Krylov basis when either the maximum desired size $n_v$ is attained, or when the condition number of
the scalar product matrix $N$ exceeds a certain threshold. This occurs when there is a risk of adding a vector that is linearly dependent
on the previous ones. In such a scenario, we can solve the generalized eigenvalue problem to obtain
the next best estimate in a smaller basis. Another approach to enhance the convergence of eigenvalues is to extrapolate the next vector
based on previous estimates, using the formula :math:`|{\xi_{k+1}}\rangle=(1-\gamma)|{\psi_{k+1}}\rangle+|{\psi_k}\rangle`, with the
memory factor :math:`\gamma=-0.75`.


The rule in :ref:`alg_descent` is a particular case of the Arnoldi iteration with a Krylov basis with  :math:`L=2`.

Ref. :cite:t:`GarciaMolina2024` presents this algorithm and its implementation for global optimization problems. It is also suitable for evolution problems.

.. autosummary::

    ~seemps.optimization.arnoldi
    ~seemps.evolution.arnoldi






