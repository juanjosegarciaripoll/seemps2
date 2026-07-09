.. _mps_runge_kutta:

*******************
Runge-Kutta methods
*******************

Runge-Kutta methods integrate an MPS-valued ordinary differential equation

.. math::
    \frac{d\psi}{dt} = F(t, \psi).

For :func:`~seemps.evolution.runge_kutta` and
:func:`~seemps.evolution.runge_kutta_fehlberg`, the right-hand side can be
provided either as an :class:`~seemps.operators.MPO` or as a Python callable.
If an MPO ``L`` is provided, the solver uses ``F(t, state) = L @ state``. If a
callable is provided, it must have the signature ``F(t, state)`` and return the
MPS derivative at that point. This allows the same solvers to handle
time-independent MPOs, time-dependent operators, and problem-specific PDE
right-hand sides.

The order of the method determines the local truncation error and the number of
operator applications required per step. Thus, it is important to consider the
trade-off between cost and accuracy when choosing the most suitable method for
each application.

The SeeMPS library considers four methods.

1. Euler method
----------------

This is an explicit, first-order Taylor approximation of the evolution, with a
simple update with a fixed time step.

.. math::
    \psi_{k+1} = \psi_k + \Delta t F(t_k, \psi_k).

2. Improved Euler or Heun method
---------------------------------

This is a second-order, fixed-step explicit method that uses two evaluations of
the right-hand side and two linear combinations of states.

.. math::
    v_1 &= F(t_k, \psi_k), \\
    v_2 &= F(t_k + \Delta t, \psi_k + \Delta t v_1), \\
    \psi_{k+1} &= \psi_k + \frac{\Delta t}{2}(v_1 + v_2).

3. Fourth-order Runge-Kutta method
-----------------------------------

This algorithm uses four evaluations of the right-hand side and four linear
combinations of states.

.. math::
    v_1 &= F(t_k, \psi_k), \\
    v_2 &= F(t_k + \Delta t/2, \psi_k + \Delta t v_1/2), \\
    v_3 &= F(t_k + \Delta t/2, \psi_k + \Delta t v_2/2), \\
    v_4 &= F(t_k + \Delta t, \psi_k + \Delta t v_3), \\
    \psi_{k+1} &= \psi_k + \frac{\Delta t}{6}(v_1 + 2v_2 + 2v_3 + v_4).

4. Runge-Kutta-Fehlberg method
-------------------------------
The Runge-Kutta-Fehlberg algorithm is an adaptive step-size solver that combines
two embedded Runge-Kutta formulas. This combination dynamically adjusts the step
size to keep the integration error within a specified tolerance. Each attempted
step evaluates the right-hand side six times, and the solver may repeat a step
if the proposed step size is not suitable.

Examples
========

Using a time-independent MPO:

.. code-block:: python

    final = runge_kutta(L, time=1.0, state=initial, steps=100)

Using a time-dependent or problem-specific operator:

.. code-block:: python

    def rhs(t, state):
        return diffusion_mpo(t) @ state + source(t, state)

    final = runge_kutta_fehlberg(rhs, time=(0.0, 1.0), state=initial)

.. autosummary::

    ~seemps.evolution.euler.euler
    ~seemps.evolution.euler.euler2
    ~seemps.evolution.runge_kutta.runge_kutta
    ~seemps.evolution.runge_kutta.runge_kutta_fehlberg

See also
========

- :doc:`arnoldi` - Krylov-based time evolution
- :doc:`crank_nicolson` - Implicit time evolution methods
- :doc:`tebd_evolution` - Local evolution for nearest-neighbor Hamiltonians
- :doc:`tdvp` - Time-dependent variational principle
