**************
Canonical form
**************

The :doc:`sequential Schmidt decomposition <tensor_to_mps>` creates a particular
type of matrix-product state, called a "canonical form". In the example cited
above, the final MPS is in canonical form with respect to the first site. This
means that we can write

.. math::
    |\psi\rangle = A^{(1)}_{1i_1\alpha_1}|i_1\rangle|\alpha_1\rangle

Here, :math:`i_1` labels the physical state of the first subsystem and
:math:`\alpha_1` labels the possible states of the collective of remaining
(N-1) quantum subsystems:

.. math::
    |\alpha_1\rangle = \sum_{\alpha,i} A^{(2)}_{\alpha_2i_2\alpha_3}\cdot A^{(N)}_{\alpha_Ni_N1}|\alpha_1,\ldots\alpha_N\rangle


The use of the Schmidt decomposition guarantees that the collective states of
the remaining systems---the so called "environment"---forms an orthonormal basis.

.. math::
    \langle \alpha_N|\alpha_N'\rangle = \delta_{\alpha_N\alpha_N'}

This is extremely convenient for many types of calculations. For instance, to
compute the expected value of an observable :math:`O` acting on the first site,
we can ignore the environment and use

.. math::
    \langle O\rangle = \sum_{i,j,\alpha_1} O_{ij} A^{(1) *}_{1i\alpha}  A^{(1) *}_{1j\alpha}

Interestingly, the canonical form can be constructed with respect to any site,
not just the first or the last one. In SeeMPS, the :py:class:`CanonicalMPS`
class is responsible for creating and maintaining these canonical forms, as
explained :doc:`in this manual <../seemps_objects_canonical>`. It is also
possible to move the center of a canonical form using the
:func:`~seemps.state.CanonicalMPS.recenter` method.

Consider the following example. We initially create a random matrix-product
state with arbitrary tensors and transform it into canonical form with respect
to the first site. This state is then destructively transformed into canonical
form with respect to the last site. In both cases we can use a single tensor
from the state to compute the norm, because the other tensors create an
orthogonormal environment basis:

    >>> import numpy as np
    >>> from seemps import random_uniform_mps, CanonicalMPS
    >>> state = CanonicalMPS(random_uniform_mps(2, 10), center=0)
    >>> print(f"State norm: {np.linalg.norm(state[0])}")
    >>> state.recenter(-1)
    >>> print(f"State norm: {np.linalg.norm(state[-1])}")