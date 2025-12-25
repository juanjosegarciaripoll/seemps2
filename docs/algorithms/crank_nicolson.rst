.. _mps_crank_nicolson:

*********************
Crank-Nicolson method
*********************

The Crank-Nicolson method is a second-order implicit approach that combines the forward Euler method and its backward counterpart at the `k` and `k+1` iterations, respectively. The approximation of the state at the k-th iteration is expressed as

.. math::
    \left(\mathbb{I}+\frac{i\Delta t}{2}H\right)\psi_{k+1}=\left(\mathbb{I}-\frac{i\Delta t}{2}H\right)\psi_{k}.

Techniques for matrix inversion can be employed to solve the system of equations in its matrix-vector form.
Other methods, such as the conjugate gradient descent :func:`seemps.cgs.cgs` , can also be adapted for implementation within an MPO-MPS framework.

.. autosummary::

   ~seemps.evolution.crank_nicolson