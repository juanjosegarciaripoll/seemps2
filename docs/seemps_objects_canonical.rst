.. currentmodule:: seemps.state

.. _mps_classes:

********************************************************
Canonical form MPS (:class:`CanonicalMPS`)
********************************************************

There is a particular form of matrix-product states where most of the tensors
are isometries that, when contracted from the left or the right, they give the
identity. In SeeMPS we call those states being in "canonical form" and have
a special class :class:`CanonicalMPS` for them.

Assume we have a state :math:`\psi` in canonical form with respect to the
`n`-th site. This state can be written as

.. math::
    |\psi\rangle = A_{\alpha_n,i_n,\alpha_{n+1}}\ket{\alpha_n}\ket{i_n}\ket{\alpha_{n+1}}

where the quantum states :math:`\ket{\alpha_{n}}` and :math:`\ket{\alpha_{n+1}}`
enclose all subsystems to the left (`j < i`) or to the right (`j > i`).
Because the state is in canonical form, these "environment" states satisfy
orthogonalitiy relations

.. math::
    \braket{\alpha_{n}'|\alpha_{n}} = \delta_{\alpha_{n}'\alpha_{n}},
    \mbox{ and }
    \braket{\alpha_{n+1}'|\alpha_{n+1}} = \delta_{\alpha_{n+1}'\alpha_{n+1}}

This simplifies many tasks, such as the computation of expectation values:

.. math::
    \braket{O_n} =
    \sum_{i,j,\alpha,\beta} (O_n)_{ji} A^*_{\alpha j \beta} A_{\alpha i \beta}

.. _canonical_mps_creation:

Creation
========

Many of the algorithms described in other parts of this documentation
(e.g., :doc:`the algorithms section <index>`) produce MPS in
canonical form. Alternatively, we can create a :class:`CanonicalMPS`
object from a :class:`MPS` by a process known as orthogonalization,
specifying both the center for the canonical form, as well as the truncation
errors and bond dimensions that we will allow.

.. autosummary::

    CanonicalMPS
    CanonicalMPS.from_vector
