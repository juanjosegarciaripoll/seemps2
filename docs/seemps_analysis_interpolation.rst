.. _analysis_interpolation:

************************************
Function Interpolation
************************************
Interpolation allows to extend the number of points on which a function is
defined by estimating the new data points from the original ones. The SeeMPS
library provides two interpolation algorithms suitable for an MPS representation.

Finite difference interpolation
===============================
Finite differences is a widespread algorithm for approximation of derivatives. This technique
is also useful for interpolation when combined with the Taylor expansion. When given a representation of
a function :math:`f\left(x_s^{(n)}\right)` on an `n`-qubit system with a discretization step of :math:`\Delta x_s^{(n)}`,
the second-order finite difference interpolant calculates the new points of the `(n+1)`-qubit grid as

.. math::
  f(x + \Delta x/2) \approx f(x) + \frac{f(x + \Delta x)-f(x)}{2}.

In the quantum register representation---equivalently MPS representation--- the function displacements
are performed by the displacement operators

.. math::
    \hat{\Sigma}^+|s\rangle = \left\{
    \begin{array}{ll}
      |s+1\rangle, & s < 2^{n} \\
      0 & \mbox{else.}
    \end{array}
  \right. \quad \hat{\Sigma}^- = \left(\hat{\Sigma}^+\right)^\dagger.

.. autosummary::

    ~seemps.analysis.interpolation.finite_differences_interpolation

Ref. :cite:t:`GarciaMolina2024` presents a more detailed explanation and an use case.

Fourier interpolation
=====================

If the function is bandwidth-limited and for a sufficiently small spacing `\Delta{x}^{(n)}\leqslant 2\pi/L_p`
according to the Nyquist-Shannon theorem, it is possible to use Fourier interpolation to reconstruct the continuous function with up to
doubly-exponentially decaying error in the number of qubits as

.. math::
  f(x) \propto \sum_{s=0}^{2^n-1} e^{-ip_s x} \langle{s|\text{QFT} |f^{(n)}}\rangle.

Its quantum register implementation has three steps: (i) computation of the QFT of the originally encoded function :math:`|{\tilde{f}^{(n)}}\rangle`,
(ii) addition of `m`` auxiliary qubits to enlarge the momentum space with qubit reordering `U_\text{sym}` to map the original discretization with $2^n$ to the
intervals `s \in [0,2^{n-1}) \cup [2^{n+m}-2^{n-1},2^{n+m})`, and (iii)  Fourier transform back to recover the state with $n+m$ qubits. The complete algorithm reads

.. math::
  |f^{(n+m)}\rangle = \text{QFT}^{-1}U_{\text{sym}} \left(|{0}\rangle^{\otimes m} \otimes \text{QFT}|{f^{(n)}}\rangle\right)=: U_\text{int}^{n,m}|{f^{(n)}}\rangle.

.. autosummary::

    ~seemps.analysis.interpolation.fourier_interpolation

Ref. :cite:t:`GarciaMolina2022` presents a more detailed explanation and an use case.

An example on how to use these functions is shown in `Interpolation.ipynb <https://github.com/juanjosegarciaripoll/seemps2/blob/main/examples/Interpolation.ipynb>`_.