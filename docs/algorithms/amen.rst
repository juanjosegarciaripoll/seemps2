.. currentmodule:: seemps.solve.amen

Alternating minimal energy (AMEn)
=================================

:func:`amen_solve` solves an MPO linear system ``A x = b`` and returns a
``CanonicalMPS``. The one-site method uses an ALS approximation of
the residual to enrich the solution ranks, following
`Dolgov and Savostyanov (2014) <https://doi.org/10.1137/140953289>`_.

For example::

    from seemps.solve import amen_solve

    x = amen_solve(A, b, guess, rtol=1e-8, maxiter=20)

The initial ``guess`` is optional and defaults to ``b``. Inputs are not
modified. The local solver ``method`` accepts the same options as
:func:`~seemps.solve.dmrg_solve`: use the default ``"cg"`` for Hermitian
positive definite operators, or ``"gmres"`` for nonsymmetric systems.

``residual_rank`` sets the rank of the auxiliary residual approximation
(default 8); each bond is enriched with up to that many residual directions.
The standard ``strategy`` controls SVD truncation and the maximum bond
dimension, for example ``strategy.replace(max_bond_dimension=128)``.
Tighter residual targets may require a tighter strategy tolerance.

Local accuracy is derived from ``atol`` and ``rtol``. When the residual
approximation falls below the tolerance, the true residual is computed, and
only it can stop the sweeps. SeeMPS logging reports the rank, the residual
approximation and the last true residual after each half-sweep.

.. autofunction:: amen_solve
