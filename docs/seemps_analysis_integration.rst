.. currentmodule:: seemps.analysis.integrals

.. _analysis_integration:

************************************
Function Integration
************************************

Functions encoded in MPS can be efficiently integrated, by contracting those quantum representations with representations of weights that implement some quadrature formula. For instance, a simple Riemann approximation results from the addition of all values of the functions, weighted by the interval size, which are equivalent to the contraction between the representation of :math:`f(x)` and the identity function :math:`g(x)=1`

.. math::
    \int f(x)\mathrm{d}x \simeq \sum_i f(x_i) \Delta{x} = \langle g | f\rangle.

In the following table we find both functions that construct the states associated to various quadratures---i.e. `mps_*` functions---and a function that implements the integral using any of those rules :py:func:`integrate_mps`

.. autosummary::
    :toctree: generated/

    ~integrate_mps
    ~mps_midpoint
    ~mps_trapezoidal
    ~mps_simpson
    ~mps_fifth_order
    ~mps_fejer
