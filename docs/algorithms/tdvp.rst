.. _mps_tdvp:

************************************************
Time-Dependent Variational Principle (TDVP)
************************************************

The Time-Dependent Variational Principle (TDVP) is an algorithm for time evolution
that operates locally on the MPS tensors, similar to DMRG. Unlike global methods
like Runge-Kutta that update all tensors simultaneously, TDVP sweeps through the
MPS and evolves pairs of neighboring tensors using effective Hamiltonians.

This algorithm was introduced by Refs. :cite:t:`Haegeman2011` and :cite:t:`Haegeman2016`,
who showed that the time-dependent variational principle provides a unifying framework
for both time evolution and optimization methods in the context of matrix product states.

The algorithm works as follows. For a two-site TDVP step at sites :math:`n` and :math:`n+1`,
the combined tensor :math:`A^{[n,n+1]}` is evolved forward in time:

.. math::
    A^{[n,n+1]}(t + \delta t/2) = \exp\left( - i \frac{\delta t}{2} H_{\text{eff}}^{[n,n+1]} \right) A^{[n,n+1]}(t)

The evolved tensor is then decomposed via SVD to restore the canonical MPS form,
:math:`A^{[n, n+1]} \to A^{[n]}A^{[n+1]}`, with singular values truncated according
to the specified tolerance and maximum bond dimension.

To advance the center of orthogonality to the next site without overcounting the
evolution on the shared subspace, the single-site tensor :math:`A^{[n+1]}` is evolved
backward in time:

.. math::
    A^{[n+1]}(t) = \exp\left(i \frac{\delta t}{2} H_{\text{eff}}^{[n+1]} \right) A^{[n+1]}(t+\delta t/2)

The effective Hamiltonians :math:`H_\text{eff}^{[n, n+1]}` and :math:`H_\text{eff}^{[n]}`
are obtained by contracting the MPO with the current left and right environments.
A complete TDVP step consists of a left-to-right sweep followed by a right-to-left
sweep, yielding a symmetric second-order integrator with local error :math:`\mathcal{O}(\delta t^3)`.

Advantages of TDVP
==================

- **Bond dimension adaptation**: Like two-site DMRG, TDVP can dynamically adjust
  the bond dimension during evolution.
- **Energy conservation**: TDVP preserves energy (up to truncation errors) because
  it projects onto the tangent space of the MPS manifold.
- **Works with any MPO**: Unlike TEBD, TDVP works with arbitrary Hamiltonians in
  MPO form, not just nearest-neighbor interactions.

.. autosummary::

   ~seemps.evolution.tdvp

See also
========

- :doc:`arnoldi` - Krylov-based time evolution
- :doc:`tebd_evolution` - Local evolution for nearest-neighbor Hamiltonians
- :doc:`runge_kutta` - Explicit time evolution methods
- :doc:`crank_nicolson` - Implicit time evolution methods
