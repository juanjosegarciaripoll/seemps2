.. currentmodule:: seemps.analysis

.. _analysis_derivatives:

************************************
Function Differentiation
************************************

The approximation of differential operators is key to efficiently solve partial
differential equations.

Finite Differences
==================

The second-order finite difference method approximates the first and second derivatives of `f(x)` as

.. math::
  \frac{\partial f(x)}{\partial x} = \frac{f(x + \Delta x) - f(x - \Delta x)}{2 \Delta x} + O(\Delta x^2),

.. math::
  \frac{\partial^2 f(x)}{\partial x^2} = \frac{f(x + \Delta x) - 2 f(x) + f(x - \Delta x)}{ \Delta x^2} + O(\Delta x^2).

This is translated into the quantum register representation using the displacement operator

.. math::
    \hat{\Sigma}^+|s\rangle = \left\{
    \begin{array}{ll}
      |s+1\rangle, & s < 2^{n} \\
      0 & \mbox{else.}
    \end{array}
  \right. \quad \hat{\Sigma}^- = \left(\hat{\Sigma}^+\right)^\dagger,

leading to

.. math::
    \ket{\partial_{x}f^{(n)}} \simeq \frac{1}{2\Delta{x}}\left(\hat{\Sigma}^+-\hat{\Sigma}^-\right)\ket{f^{(n)}},

.. math::
    \ket{\partial^2_{x}f^{(n)}} \simeq \frac{1}{\Delta{x}^2}\left(\hat{\Sigma}^+-2\mathbb{I}+\hat{\Sigma}^-\right)\ket{f^{(n)}}.

This approximation is improved with a smoother formula to avoid noise resilience following `Holoborodko <http://www.holoborodko.com/pavel/numerical-methods/numerical-derivative/smooth-low-noise-differentiators>`_.

An example on how to use these functions is shown in `Differentiation.ipynb <https://github.com/juanjosegarciaripoll/seemps2/blob/main/examples/Differentiation.ipynb>`_.

.. autosummary::

    ~finite_differences.smooth_finite_differences_mpo

Hermite Distributed Approximating Functionals (HDAFs)
=====================================================

HDAFs allow to approximate operators that are functions of derivatives. In particular, for a derivative of order :math:`k`,

.. math::
   |\partial_x^k f\rangle \approx  \left[\Delta x \delta_{M}^{(k)}(0; \sigma) \mathbb{I} + \sum_{i=1}^{W} \Delta x \delta_{M}^{(k)}(i\Delta x; \sigma) \left(\hat{\Sigma}^{+i} + (-1)^{k}\hat{\Sigma}^{-i}\right)\right] |f\rangle,

where :math:`W` is a suitable index where the summand vanishes,

.. math::
   \delta_{M}^{(k)}(x; \sigma) = \left(\frac{-1}{\sqrt{2}\sigma}\right)^{k} \frac{\exp\left(\frac{-x^{2}}{2\sigma^{2}}\right)}{\sqrt{2\pi}\sigma} \sum_{m=0}^{M/2}\left(-\frac{1}{4}\right)^{m} \frac{H_{2m + k}\left(\frac{x}{\sqrt{2}\sigma}\right)}{m!},

and :math:`H_n(x)` is the :math:`n`-th Hermite polynomial on :math:`x`.

.. autosummary::

   ~hdaf.hdaf_mpo


Fourier approximation
=====================
Ref. :cite:t:`GarciaRipoll2021` shows that the quantum Fourier transform :math:`\mathcal{F}` can be used to construct
as differential operator as

.. math::
  D(-i\nabla) := \mathcal{F}^{-1} \sum_{\lbrace s \rbrace} D(p_s)\ket{s}\!\bra{s} \mathcal{F}.

This can be obtained combining the :func:`operators.p_to_n_mpo` function with SeeMPS's QFT :func:`seemps.qft.qft`. However, the QFT is not yet optimally implemented for this task, since the bond dimension scales linearly with the number of sites.