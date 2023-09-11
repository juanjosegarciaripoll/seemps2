.. currentmodule:: seemps

.. _mps_class:

************************************
Matrix-product states (:class:`MPS`)
************************************

The matrix-product state is an efficient representation of composite quantum
systems that reconstructs the total wavefunction from the contraction of
three-legged tensors.

If :math:`\Psi` is a quantum state with `N` components, a MPS representation
would look as follows

.. math::
    |\Psi\rangle = \sum_{\vec{i},\vec{\alpha}}
    \prod_{n=1}^N A_{\alpha_n, i_n, \alpha_{n+1}}
    |i_1,i_2,\ldots, i_N\rangle

Here, the :math:`i_n` are indices labelling the quantum states of the respective
subsystems, while the :math:`\alpha_n` are integer indices connecting
(correlating) neigboring quantum systems. The former are usually labeled the
"physical" indices, while the latter are declared "virtual" indices.

In SeeMPS, matrix-product states are represented by the class :class:`MPS`. The
instances of this class keep a sequence (list) of these three-legged tensors,
as well as other information, such as accumulated errors in this representation.
Such matrix-product states can be mutated, they can be added, rescaled,
renormalized, or they can be subject to quantum operations, such as gates
or the computation of observables.

Creation
========

Matrix product states can be created by supplying the tensors that form the
state, and in the case of small wavefunctions, they can be reconstructed directly
from the state vector of the composite quantum state. In addition to this, we
offer some functions to create convenience states

.. autosummary::
    :toctree: generated/

    ~seemps.state.MPS
    ~seemps.state.MPS.from_vector
    ~seemps.state.AKLT
    ~seemps.state.GHZ
    ~seemps.state.graph
    ~seemps.state.product_state
    ~seemps.state.random_mps
    ~seemps.state.W

.. _mps_operations:

Simple operations
=================

Matrix product states can be combined with each other and transformed:

- An MPS `a` can be rescaled by a number `n * a`
- An MPS can be conjugated `a.conj()`
- Two MPS `a` and `b` can be added, producing an :class:`~seemps.state.MPSSum`
  (see :doc:`seemps_objects_sum`).
- The wavefunctions of two states can be multiplied element-wise `a * b`
  in an unphysical transformation.

.. _mps_expectation:

Observables
===========

It is possible to extract expectation values from matrix-product states in a
very efficient way, thanks to their simple tensor structure. The following
functions provide access to single- and two-body expectation values in a
convenient way.

.. autosummary::
    :toctree: generated/

    ~seemps.expectation.scprod
    ~seemps.expectation.MPS.norm
    ~seemps.expectation.MPS.norm_squared
    ~seemps.expectation.expectation1
    ~seemps.expectation.expectation2
    ~seemps.expectation.all_expectation1
