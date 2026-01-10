.. _analysis_polynomials:

*********************
Polynomial expansions
*********************

Consider a function :math:`f(x)` expressed as a finite polynomial expansion

.. math::
    f(x) = \sum_{k=0}^d c_k p_k(x)

SeeMPS provides two *distinct* but related sets of polynomial approximation tools:

* **Exact construction of MPS for 1D polynomials**, when the coefficients are known explicitly.

* **Polynomial expansions of operators or states**, evaluating :math:`f(A)` where :math:`A` is either an MPS or an MPO.

Exact polynomial MPS constructions
----------------------------------

The function :func:`seemps.analysis.polynomials.mps_from_polynomial` constructs the *exact* MPS corresponding to a one-dimensional polynomial expressed in the monomial basis :math:`p_k(x)=x^k` over some equispaced discretization :math:`[a, b]`, represented by the class :class:`seemps.analysis.mesh.RegularInterval`.

Polynomial expansions of MPS and MPOs
-------------------------------------

This implements the remaining functionality using the :meth:`seemps.analysis.expansion.PolynomialExpansion.to_mps` and :meth:`seemps.analysis.expansion.PolynomialExpansion.to_mpo` methods. These apply the Clenshaw evaluation method, a numerically stable technique to evaluate polynomials in situations of finite precision.

SeeMPS currently provides the following expansion classes:

* :class:`seemps.analysis.expansion.PowerExpansion`, a polynomial expansion in the monomial basis :math:`p_k(x)=x^k`. In this scenario, Clenshaw's evaluation formula reduces to Horner's method, an efficient and robust technique for evaluating polynomial expansions.

* :class:`seemps.analysis.expansion.ChebyshevExpansion`, an expansion in the orthogonal basis of Chebyshev_ polynomials.

* :class:`seemps.analysis.expansion.LegendreExpansion`, an expansion in the orthogonal basis of Legendre_ polynomials.

.. _Chebyshev: https://en.wikipedia.org/wiki/Chebyshev_polynomials
.. _Legendre: https://en.wikipedia.org/wiki/Legendre_polynomials

This expansion framework is easily extensible to any orthogonal polynomial family by subclassing :class:`seemps.analysis.expansion.OrthogonalExpansion`, requiring only the three-term recurrence relation and the canonical domain of definition.

All expansion objects can be constructed by providing the coefficients :math:`[c_0,c_1,...]` explicitly. Alternatively, the coefficients can be computed by projecting the target function onto the orthogonal basis through projection methods, which estimate a finite-order expansion of a scalar function over a given domain :math:`[a, b]` using numerical quadratures.

An example demonstrating the use of these functions for the case of Chebyshev polynomials is shown in
`Chebyshev.ipynb <https://github.com/juanjosegarciaripoll/seemps2/blob/main/examples/Chebyshev.ipynb>`_.

.. autosummary::

    ~seemps.analysis.expansion.PolynomialExpansion
    ~seemps.analysis.expansion.ChebyshevExpansion
    ~seemps.analysis.expansion.LegendreExpansion
    ~seemps.analysis.expansion.ChebyshevExpansion.project
    ~seemps.analysis.polynomials.mps_from_polynomial