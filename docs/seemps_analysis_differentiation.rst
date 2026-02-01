.. _analysis_derivatives:

************************************
Function Differentiation
************************************

The approximation of differential operators is key to efficiently solve partial
differential equations. We provide three mechanisms to do it, which are described below.

.. autosummary::

    ~seemps.analysis.derivatives.finite_differences_mpo
    ~seemps.analysis.derivatives.fourier_derivative_mpo
    ~seemps.analysis.derivatives.hdaf_derivative_mpo


Finite Differences
==================

The second-order finite difference method approximates the first and second derivatives of :math:`f(x)` as

.. math::
  \frac{\partial f(x)}{\partial x} = \frac{f(x + \Delta x) - f(x - \Delta x)}{2 \Delta x} + O(\Delta x^2),

.. math::
  \frac{\partial^2 f(x)}{\partial x^2} = \frac{f(x + \Delta x) - 2 f(x) + f(x - \Delta x)}{ \Delta x^2} + O(\Delta x^2).

MPO representation
------------------

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

These MPOs have a bond dimension equal to the number of displacements (:math:`\chi=3` for the
second derivative), and an error that decays algebraically with the grid step
:math:`O((\Delta x)^2)` and exponentially in the number of qubits :math:`O(2^{-2n})`.

Higher-order stencils
---------------------

This approximation is improved with smoother formulas to avoid noise artifacts,
following `Holoborodko <http://www.holoborodko.com/pavel/numerical-methods/numerical-derivative/smooth-low-noise-differentiators>`_.
Higher-order derivatives and higher-order stencils can be constructed using the same formalism,
with the `order` parameter controlling the accuracy of the approximation.

An example on how to use these functions is shown in `Differentiation.ipynb <https://github.com/juanjosegarciaripoll/seemps2/blob/main/examples/Differentiation.ipynb>`_.


Hermite Distributed Approximating Functionals (HDAFs)
=====================================================

HDAFs are well-tempered approximations of the Dirac-delta distribution constructed
with Hermite polynomials :cite:p:`hoffman1991`. They allow approximating operators
that are functions of derivatives, making them particularly useful for computing
differential operators and time evolution propagators.

Mathematical formulation
------------------------

Given a function :math:`f(x)`, the HDAF approximation reconstructs it through convolution
with a smoothed delta function:

.. math::
   f(x) \simeq \int \delta_M(x - x';\sigma)f(x')\mathrm{d}x'

where the HDAF kernel is:

.. math::
   \delta_M(x;\sigma) = \frac{\exp\left(\frac{-x^2}{2\sigma^2}\right)}{\sqrt{2\pi}\sigma}
   \sum_{m=0}^{M/2}\left(-\frac{1}{4}\right)^m\frac{1}{m!}
   H_{2m}\left(\frac{x}{\sqrt{2}\sigma}\right)

and :math:`H_n(x)` is the :math:`n`-th Hermite polynomial.

Derivative approximation
------------------------

For a derivative of order :math:`k`, the HDAF approximation becomes:

.. math::
   |\partial_x^k f\rangle \approx  \left[\Delta x \delta_{M}^{(k)}(0; \sigma) \mathbb{I} + \sum_{i=1}^{W} \Delta x \delta_{M}^{(k)}(i\Delta x; \sigma) \left(\hat{\Sigma}^{+i} + (-1)^{k}\hat{\Sigma}^{-i}\right)\right] |f\rangle,

where :math:`W` is a suitable cutoff index where the summand vanishes, and

.. math::
   \delta_{M}^{(k)}(x; \sigma) = \left(\frac{-1}{\sqrt{2}\sigma}\right)^{k} \frac{\exp\left(\frac{-x^{2}}{2\sigma^{2}}\right)}{\sqrt{2\pi}\sigma} \sum_{m=0}^{M/2}\left(-\frac{1}{4}\right)^{m} \frac{H_{2m + k}\left(\frac{x}{\sqrt{2}\sigma}\right)}{m!}

The vectors of HDAF coefficients decay rapidly, so this expansion can be truncated
to finite order, yielding an efficient linear combination of displacement operators.

Kinetic propagator
------------------

Beyond differentiation, HDAFs can also approximate the kinetic propagator
:math:`\exp(-it\partial_x^2/2)` directly in the coordinate basis. This is particularly
useful for split-step time evolution methods (see :func:`~seemps.evolution.hdaf.split_step`),
where the kinetic propagator is represented as a banded MPO constructed from
discrete shift operators, with controllable accuracy determined by the truncation order :math:`M`.

.. autosummary::

   ~seemps.analysis.hdaf.hdaf_mpo


Fourier approximation
=====================

Spectral differentiation based on Fourier techniques achieves significantly higher
accuracy than finite differences and allows computing any analytical function
:math:`G(\partial_x)` of the differential operator.

Mathematical formulation
------------------------

Ref. :cite:t:`GarciaRipoll2021` shows that the quantum Fourier transform :math:`\mathcal{F}`
can be used to construct a differential operator as:

.. math::
  G(\partial_x) \to \mathcal{F}^{-1} G(i\mathbf{p}) \mathcal{F}

where :math:`\mathbf{p}` labels the MPS encodings of the vector of frequencies in Fourier space.

Convergence properties
----------------------

For analytic and periodic functions, spectral differentiation achieves exponential
convergence :math:`O(e^{-rN})` with the number of grid points :math:`N=2^n`. This implies
a super-exponential convergence with the number of qubits: :math:`O(e^{-r2^n})`.

However, when the function is not periodic, or the interval size does not match the period,
the Fourier expansion may lead to localized Gibbs oscillations that decay only algebraically
with the number of qubits: :math:`O(N^{-\alpha}) \sim O(2^{-n\alpha})`.

Implementation notes
--------------------

While the momentum operator :math:`\hat{p}` admits an MPO representation with bond dimension
:math:`\chi=2`, a naive MPO construction of the QFT leads to a bond dimension that grows
linearly with the number of sites. Efficient implementations exploiting bit-reversal
symmetry significantly reduce this cost :cite:p:`chen2023`.

This can be obtained combining the :func:`~seemps.register.p_to_n_mpo` function with
SeeMPS's QFT (see :doc:`algorithms/qft`). The resulting :class:`~seemps.operators.MPOList`
object may need to be simplified using :func:`~seemps.operators.simplify_mpo` for
efficient application.

See also
========

- :doc:`algorithms/qft` - Quantum Fourier Transform implementation
- :doc:`seemps_analysis_interpolation` - Fourier-based interpolation methods
