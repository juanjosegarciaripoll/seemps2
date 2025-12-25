.. _alg_descent:

****************
Gradient descent
****************

This is a very simple iterative algorithm to solve the problem of minimizing
the energy associated to a Hamiltonian :math:`H`, over the space of matrix-product
states :math:`\psi`. In other words, given the definition

.. math::
    E[\psi] := \frac{\langle{\psi|H|\psi}\rangle}{\langle{\psi|\psi}\rangle},

we want to find :math:`\mathrm{argmin}_\psi E[\psi]`. The gradient of this
functional is simply

.. math::

    \nabla E[\psi] = (H - E[\psi])|\psi\rangle := \tilde{H}|\psi\rangle.


The algorithm proceeds in discrete steps, where given a state :math:`\psi_k`, it
finds the next state that minimizes the energy along the gradient direction:

.. math::
    |\psi_{k+1}\rangle \propto |\psi_k\rangle - a \tilde{H}|\psi_k\rangle,

where

.. math::
    a = \mathrm{argmin} E[(1- a \tilde{H})\psi_k]

The optimum of this descent is given by

.. math::
    a = \frac{E_3 - \sqrt{E_3^2 + 4 \Delta H^6}}{2 \Delta H^2}

with the definitions

.. math::
    \Delta H^2 = \langle\psi|(H-E[\psi])^2|\psi\rangle,\;
    E_3 = \langle\psi|(H-E[\psi])^3|\psi\rangle.

This formulation, used in Ref. :cite:t:`GarciaMolina2024`, is implemented by the function :func:`~seemps.optimization.gradient_descent`.

.. autosummary::

   ~seemps.optimization.gradient_descent
