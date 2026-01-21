.. _mps_implicit:

*******************************
Implicit time evolution methods
*******************************

Implicit methods solve the time evolution by inverting a linear system at each step.
While more computationally expensive per step than explicit methods, they offer
superior stability for stiff problems and can use larger time steps.

Crank-Nicolson method
=====================

The Crank-Nicolson method is a second-order implicit approach that combines the
forward Euler method and its backward counterpart at the :math:`k` and :math:`k+1`
iterations, respectively. The approximation of the state at the :math:`k`-th
iteration is expressed as

.. math::
    \left(\mathbb{I}+\frac{i\Delta t}{2}H\right)\psi_{k+1}=\left(\mathbb{I}-\frac{i\Delta t}{2}H\right)\psi_{k}.

Each step requires solving a linear system, which in SeeMPS is done using the
conjugate gradient method.

.. autosummary::

   ~seemps.evolution.crank_nicolson

Radau IIA method
================

For stiff problems requiring high-order accuracy, SeeMPS provides the Radau IIA
integrator with either 3 or 5 stages (order 5 and 9, respectively).

The algorithm requires solving for a vector of stage derivatives :math:`K` in

.. math::
    \left[\mathbb{I}_s \otimes \mathbb{I} + \frac{\delta t}{\hbar} (A \otimes \bar{H})\right] K = -\frac{1}{\hbar} \mathbf{1} \otimes (\bar{H} \mathbf{v}(t)),

where :math:`A` is the Butcher matrix of size :math:`s \times s`, :math:`s` is the
number of stages, and :math:`\mathbf{1}` is a vector of ones. This system is solved
for :math:`K`, and the state update is obtained by contracting the stage index with
the Runge-Kutta weights :math:`b`:

.. math::
    \mathbf{v}(t+\delta t) = \mathbf{v}(t) + \delta t \sum_{j=1}^s b_j K_j.

The linear system is solved using DMRG-based matrix inversion.

.. autosummary::

   ~seemps.evolution.radau

See also
========

- :doc:`arnoldi` - Krylov-based time evolution
- :doc:`runge_kutta` - Explicit time evolution methods
- :doc:`tebd_evolution` - Local evolution for nearest-neighbor Hamiltonians
- :doc:`tdvp` - Time-dependent variational principle