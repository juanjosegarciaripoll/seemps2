.. _alg_split_step:

**********************
Split-step HDAF method
**********************

For PDEs encoded using quantum-inspired techniques (see :ref:`seemps_analysis`)
that can be decomposed into a sum of non-commuting terms, such as
:math:`H = -\frac{1}{2}\partial_x^2 + V(x)`, the time evolution operator can be
efficiently approximated using operator splitting techniques.

Strang splitting
================

A standard choice is the Strang splitting, which yields a symplectic integrator
equivalent to the Störmer-Verlet scheme:

.. math::
    e^{-i \delta t H} = e^{-i\frac{\delta t}{2}V(x)}
    e^{i\frac{\delta t}{2} \partial_x^2}
    e^{-i\frac{\delta t}{2}V(x)} + O(\delta t^3)

This decomposition separates the evolution into:

1. A diagonal potential propagator :math:`e^{-i\frac{\delta t}{2}V(x)}`
2. A kinetic propagator :math:`e^{i\frac{\delta t}{2}\partial_x^2}`

Implementation
==============

Potential propagator
--------------------

The diagonal potential operator :math:`e^{-i\frac{\delta t}{2}V(x)}` is approximated
using tensor cross-interpolation (TCI) techniques. Given a potential function
:math:`V(x)`, the propagator MPS is constructed by sampling the exponential at
the grid points.

Kinetic propagator
------------------

The kinetic propagator :math:`e^{i\frac{\delta t}{2}\partial_x^2}` is non-diagonal
in the coordinate representation. While standard implementations typically require
a transformation to momentum space using the quantum Fourier transform, SeeMPS adopts
an alternative approach based on Hermite Distributed Approximating Functionals (HDAFs).

This enables approximation of the kinetic propagator directly in the coordinate basis,
where it is represented as a banded MPO constructed as a linear combination of discrete
shift operators, with controllable accuracy determined by the HDAF truncation order
(see :doc:`../seemps_analysis_differentiation`).

Evolution algorithm
-------------------

Each time step proceeds as follows:

1. Apply half-step potential: :math:`\psi \to e^{-i\frac{\delta t}{2}V} \psi` (element-wise MPS product)
2. Simplify the resulting MPS
3. Apply full-step kinetic: :math:`\psi \to e^{i\delta t \partial_x^2/2} \psi` (MPO-MPS product)
4. Simplify the resulting MPS
5. Apply half-step potential: :math:`\psi \to e^{-i\frac{\delta t}{2}V} \psi`
6. Simplify the resulting MPS

This symmetric splitting ensures second-order accuracy in the time step.

Usage
=====

The split-step method supports both real time (:math:`e^{-iHt}`) and imaginary time
(:math:`e^{-Ht}`) evolution. Imaginary time evolution can be used for ground state
preparation, as it exponentially suppresses excited state contributions.

.. autosummary::

   ~seemps.evolution.hdaf.split_step

See also
========

- :doc:`../seemps_analysis_differentiation` - HDAF differentiation methods
- :doc:`runge_kutta` - Explicit Runge-Kutta time evolution
- :doc:`crank_nicolson` - Implicit time evolution methods
- :doc:`tdvp` - Time-dependent variational principle
