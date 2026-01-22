.. _analysis_operators:

**************************
Predefined Operators (MPO)
**************************

The SeeMPS library provides exact MPO representations of operators commonly used
in numerical analysis. These include coordinate operators, momentum operators,
and elementary functions that act multiplicatively on function representations.

Coordinate and momentum operators
=================================

The position operator :math:`\hat{x}` and momentum operator :math:`\hat{p}` are
fundamental building blocks for constructing differential operators and potentials:

- :func:`~seemps.analysis.operators.x_to_n_mpo` creates :math:`\hat{x}^n`, which
  acts by multiplying a function by :math:`x^n`.

- :func:`~seemps.analysis.operators.p_to_n_mpo` creates :math:`\hat{p}^n`, where
  :math:`\hat{p}` is the momentum operator conjugate to :math:`x` via the Fourier
  transform.

These operators have low bond dimension that grows linearly with the power :math:`n`.

Multiplicative operators
========================

Several elementary functions are provided as diagonal MPOs that multiply the
input function pointwise:

- :func:`~seemps.analysis.operators.exponential_mpo`: Multiplication by :math:`\exp(cx)`
- :func:`~seemps.analysis.operators.cos_mpo`: Multiplication by :math:`\cos(x)`
- :func:`~seemps.analysis.operators.sin_mpo`: Multiplication by :math:`\sin(x)`

These are useful for constructing potential energy terms or weight functions.

Example:

.. code-block:: python

    from seemps.analysis.operators import exponential_mpo, x_to_n_mpo

    n_qubits = 10
    a, dx = 0.0, 0.01  # Grid parameters

    # Create operators
    x_squared = x_to_n_mpo(n_qubits, a, dx, n=2)  # x² operator
    exp_minus_x = exponential_mpo(n_qubits, a, dx, c=-1.0)  # exp(-x)

Utility operators
=================

- :func:`~seemps.analysis.operators.id_mpo`: Identity operator, useful as a
  baseline for constructing linear combinations.

- :func:`~seemps.analysis.operators.mpo_cumsum`: Computes the cumulative sum
  (discrete integration) of an MPS.

- :func:`~seemps.analysis.operators.mpo_affine`: Applies an affine transformation
  mapping coordinates from one interval to another.

API reference
=============

.. autosummary::

    ~seemps.analysis.operators.id_mpo
    ~seemps.analysis.operators.x_to_n_mpo
    ~seemps.analysis.operators.p_to_n_mpo
    ~seemps.analysis.operators.exponential_mpo
    ~seemps.analysis.operators.cos_mpo
    ~seemps.analysis.operators.sin_mpo
    ~seemps.analysis.operators.mpo_affine
    ~seemps.analysis.operators.mpo_cumsum
