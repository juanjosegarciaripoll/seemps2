.. _analysis_polynomials:

************************
Polynomial Approximation
************************

Given a function :math:`f(x)` expressed in a basis of polynomials

.. math::
    f(x) = \sum_{k=0}^d c_k p_k(x)

SeeMPS provides routines to

* Reconstruct :math:`f(x)` as an MPS, over some interval.
* Evaluate :math:`f(A)` where :math:`A` is either an MPS or an MPO.

Both applications are implemented using the :meth:`seemps.analysis.expansion.PolynomialExpansion.to_mps`
and :meth:`seemps.analysis.expansion.PolynomialExpansion.to_mpo`, and a clever applications
of Clenshaw's evaluation method for polynomials (a numerically stable technique to compute
polynomials in situations of finite precision).

SeeMPS provides four polynomial expansions:

* :func:`seemps.analysis.mps_from_polynomial` constructs an unoptimize MPS from is expansion in monomials :math:`p_k(x)=x^k`.

* :class:`seemps.analysis.expansion.PowerExpansion` is another expansion in monomials :math:`p_k(x)=x^k` using now Clenshaw's formula.

* :class:`seemps.analysis.expansion.ChebyshevExpansion` is an expansion in the orthogonal basis of Chebyshev_ polynomials.

* :class:`seemps.analysis.expansion.LegendreExpansion` is an expansion in orthogonal Legendre_ polynomials.

.. _Chebyshev: https://en.wikipedia.org/wiki/Chebyshev_polynomials
.. _Legendre: https://en.wikipedia.org/wiki/Legendre_polynomials

All expansion objects can be constructed by providing explicitly the coefficients `[c_0,c_1,...]`
and the domain of definition `[a,b]`. However, the orthogonal expansions also have additional
`project()` methods that, given a scalar function and a definition domain, estimate a finite-order
expansion using numerical integration techniques.

An example on how to use these functions is shown in
`Chebyshev.ipynb <https://github.com/juanjosegarciaripoll/seemps2/blob/main/examples/Chebyshev.ipynb>`_.

.. autosummary::

    ~seemps.analysis.expansion.PolynomialExpansion
    ~seemps.analysis.expansion.ChebyshevExpansion
    ~seemps.analysis.expansion.LegendreExpansion
    ~seemps.analysis.expansion.ChebyshevExpansion.project
    ~seemps.analysis.expansion.LegendreExpansion.project
    ~seemps.analysis.polynomials.mps_from_polynomial