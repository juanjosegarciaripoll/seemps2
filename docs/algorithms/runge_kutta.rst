.. _mps_runge_kutta:

*******************
Runge-Kutta methods
*******************

Runge-Kutta methods use a Taylor expansion of the state to approximate

.. math::
    \psi_{k+1} = \psi_k + \sum_{p}\frac{1}{p!}(-\Delta \beta H)^p\frac{\partial^{(p)} \psi}{\partial \beta^{(p)}}.

:math:`\Delta \beta` can either be real or imaginary time, and hence these
methods are suitable for both evolution and optimization problems. When using
MPS, these methods perform a global optimization of the MPS at each step,
i.e., they update all tensors simultaneously.

The order of the expansion `p` determines the truncation error of the method, which
is :math:`O(\Delta \beta ^{p+1})`, and also the cost of the method, since
a higher order implies more operations. Thus, it is important to consider
trade-off in cost and accuracy to choose the most suitable method for each application.

The SeeMPS library considers four methods.

1. Euler method
----------------

This is an explicit, first-order Taylor approximation of the evolution, with an error :math:`\mathcal{O}(\Delta\beta^2)`
and a simple update with a fixed time-step :math:`\beta_k = k \Delta\beta`.

.. math::
    \psi_0 &= \psi(\beta_0), \\
    \psi_{k+1} &= \psi_k - \Delta\beta H \psi_k, \quad \text{for } k=0,1,\dots,N-1.

2. Improved Euler or Heun method
---------------------------------

This is a second-order, fixed-step explicit method that uses two matrix-vector multiplications and two linear combinations of
vectors to achieve an error :math:`\mathcal{O}(\Delta\beta^3)`.

.. math::
    \psi_{k+1} = \psi_k - \frac{\Delta\beta}{2} \left[v_1 + H(\psi_k - \Delta\beta v_1)\right], \\
    \text{with } v_1 = H \psi_k.

3. Fourth-order Runge-Kutta method
-----------------------------------

This algorithm achieves an error :math:`\mathcal{O}(\Delta\beta^5)` using four matrix-vector multiplications and four linear combinations of vectors.

.. math::
    \psi_{k+1} &= \psi_k + \frac{\Delta\beta}{6}(v_1 + 2v_2 + 2v_3 + v_4), \\
    v_1 &= -H \psi_k, \\
    v_2 &= -H\left(\psi_k + \frac{\Delta\beta}{2}v_1\right), \\
    v_3 &= -H\left(\psi_k + \frac{\Delta\beta}{2}v_2\right), \\
    v_4 &= -H\left(\psi_k + \Delta\beta v_3\right).

4. Runge-Kutta-Fehlberg method
-------------------------------
The Runge-Kutta-Fehlberg algorithm is an adaptative step-size solver that combines a fifth-order accurate integrator
:math:`O(\Delta\beta^5)` with a sixth-order error estimator  :math:`O(\Delta\beta^6)`. This combination dynamically adjusts the step size  :math:`\Delta\beta`
to maintain the integration error within a specified tolerance. The method requires an initial estimate of the step size, which can be obtained from a simpler
method. Each iteration involves six matrix-vector multiplications and six linear combinations, and it may repeat the evolution steps if the proposed step size
is deemed unsuitable.

.. autosummary::

    ~seemps.evolution.euler.euler
    ~seemps.evolution.euler.euler2
    ~seemps.evolution.runge_kutta.runge_kutta
    ~seemps.evolution.runge_kutta.runge_kutta_fehlberg
