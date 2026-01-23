.. _analysis_polynomials:

*********************
Polynomial expansions
*********************

Consider a function :math:`f(x)` expressed as a finite polynomial expansion

.. math::
    f(x) = \sum_{k=0}^d c_k p_k(x)

Smooth functions can often be accurately approximated by truncated expansions over
a complete polynomial basis :math:`\{P_k(x)\}`. SeeMPS provides tools for both direct
encoding of functions and function composition using polynomial expansions.

SeeMPS provides two *distinct* but related sets of polynomial approximation tools:

* **Exact construction of MPS for 1D polynomials**, when the coefficients are known explicitly.

* **Polynomial expansions of operators or states**, evaluating :math:`f(A)` where :math:`A` is either an MPS or an MPO.

Exact polynomial MPS constructions
==================================

The function :func:`~seemps.analysis.polynomials.mps_from_polynomial` constructs the
*exact* MPS corresponding to a one-dimensional polynomial expressed in the monomial
basis :math:`p_k(x)=x^k` over some equispaced discretization :math:`[a, b]`, represented
by the class :class:`~seemps.analysis.mesh.RegularInterval`.

Any degree-:math:`d` polynomial can be constructed using an MPS with bond dimension
:math:`\chi \leq d+1` (in practice, often smaller if compression routines are used).
This function can either take the coefficients :math:`\{p_i\}` or convert a NumPy
representation of a polynomial to MPS form.

Polynomial expansions of MPS and MPOs
=====================================

This implements the remaining functionality using the :meth:`~seemps.analysis.expansion.PolynomialExpansion.to_mps`
and :meth:`~seemps.analysis.expansion.PolynomialExpansion.to_mpo` methods. These apply
the Clenshaw evaluation method, a numerically stable technique to evaluate polynomials
in situations of finite precision.

Function composition
--------------------

Given an MPS :math:`\mathbf{v}` encoding a function :math:`g(x)`, we can compute the
encoding :math:`\mathbf{w}` of the composed function through the expansion:

.. math::
    (f \circ g)(x) = f(g(x)) \to \mathbf{w} = \sum_k c_k P_k(\mathbf{v})

The same technique applies straightforwardly to operator-valued functions, allowing
the evaluation of :math:`f(O)` for an operator :math:`O` represented as an MPO.

Available expansion classes
---------------------------

SeeMPS currently provides the following expansion classes:

* :class:`~seemps.analysis.expansion.PowerExpansion`: A polynomial expansion in the
  monomial basis :math:`p_k(x)=x^k`. In this scenario, Clenshaw's evaluation formula
  reduces to Horner's method, an efficient and robust technique for evaluating
  polynomial expansions. The user must explicitly provide the coefficients :math:`\{c_k\}`.

* :class:`~seemps.analysis.expansion.ChebyshevExpansion`: An expansion in the orthogonal
  basis of Chebyshev_ polynomials. The use of Chebyshev polynomials yields an approximation
  framework analogous to Matlab's "ChebFun" package, but formulated entirely within the
  MPS/MPO formalism.

* :class:`~seemps.analysis.expansion.LegendreExpansion`: An expansion in the orthogonal
  basis of Legendre_ polynomials.

.. _Chebyshev: https://en.wikipedia.org/wiki/Chebyshev_polynomials
.. _Legendre: https://en.wikipedia.org/wiki/Legendre_polynomials

Three-term recurrence
---------------------

Orthogonal polynomial expansions rely on three-term recurrence relations:

.. math::
    P_{k+1}(x) = (\alpha_k x + \beta_k) P_k(x) - \gamma_k P_{k-1}(x)

which are evaluated using numerically stable Clenshaw formulas. To completely determine the basis, two additional
coefficients determining the linear term and fixing affine translations are required:

.. math::
    P_{1}(x) = \sigma x + \mu

The user provides the target function :math:`f` together with an initial MPS or MPO encoding the argument.

This expansion framework is easily extensible to any classical orthogonal polynomial family by
subclassing :class:`~seemps.analysis.expansion.PolynomialExpansion`, requiring only
the three-term recurrence relation, the affine fixing coefficients, and the orthogonality domain of the basis.

Coefficient computation
-----------------------

All expansion objects can be constructed by providing the coefficients :math:`[c_0,c_1,...]`
explicitly. Alternatively, the coefficients can be computed by projecting the target
function onto the orthogonal basis through projection methods, which estimate a
finite-order expansion of a scalar function using numerical quadratures.

Limitations
-----------

The applicability of this technique is limited by the regularity of the target function and the properties of the basis.
Generally, highly differentiable functions present favorable convergence rates, while functions
with discontinuities or sharp features---such as Heaviside functions---are poorly
approximated by polynomial expansions, requiring prohibitively large expansion orders.

Example
-------

An example demonstrating the use of these functions for the case of Chebyshev polynomials is shown in
`Chebyshev.ipynb <https://github.com/juanjosegarciaripoll/seemps2/blob/main/examples/Chebyshev.ipynb>`_.

The following example encodes a multivariate function :math:`f(x, y) = e^{x + y}` using
a Chebyshev expansion::

    >>> import numpy as np
    >>> from seemps.state import mps_tensor_sum
    >>> from seemps.analysis.mesh import RegularInterval
    >>> from seemps.analysis.factories import mps_interval
    >>> from seemps.analysis.expansion import ChebyshevExpansion
    >>>
    >>> interval = RegularInterval(-1, 1, 10)
    >>> mps_x = mps_interval(interval)
    >>> mps_xy = mps_tensor_sum([mps_x] * 2)
    >>> f = lambda x: np.exp(x)
    >>> expansion = ChebyshevExpansion.project(f, (-1, 1))
    >>> mps_f = expansion.to_mps(argument=mps_xy)

.. autosummary::

    ~seemps.analysis.expansion.PolynomialExpansion
    ~seemps.analysis.expansion.PowerExpansion
    ~seemps.analysis.expansion.ChebyshevExpansion
    ~seemps.analysis.expansion.LegendreExpansion
    ~seemps.analysis.expansion.ChebyshevExpansion.project
    ~seemps.analysis.polynomials.mps_from_polynomial

See also
========

- :doc:`tt-cross` - Tensor cross-interpolation for function encoding
- :doc:`lagrange` - Multiscale interpolative constructions