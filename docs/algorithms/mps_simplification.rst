.. _mps_truncate:

******************
MPS simplification
******************

The first and most fundamental algorith, on top of which all other algorithms
can be rigorously constructed, is the simplification. This is the search for
a matrix-product state :math:`\xi` that approximates another matrix-product
state :math:`\psi`, with the goal to make it simpler: i.e., typically reduce
the size of the bond dimensions.

Mathematically, we are solving the minimization of the norm-2 distance:

.. math::
   \mathrm{argmin}_{\xi \in \mathcal{MPS}_{D'}} \Vert{\xi-\psi}\Vert^2

There are two variants of the algorithm. The first one
:func:`~seemps.state.simplify` approximates just a single state. The second
one approximates a linear combination of states and weights :math:`\psi_i` and
:math:`w_i`, as in

.. math::
   \mathrm{argmin}_{\xi \in \mathcal{MPS}_{D'}} \Vert{\xi- \sum w_i \psi_i}\Vert^2

This second algorithm is the one used to convert :class:`seemps.state.MPSSum`
objects into ordinary :class:`seemps.state.MPS` states (see
:doc:`MPS combination <../seemps_objects_sum>`). Both are implemented using the
same front-end function.

It is possible to extend this algorithm to MPOs by recasting them as an MPS.

.. autosummary::

   ~seemps.state.simplify
   ~seemps.operators.simplify_mpo
   ~seemps.state.Strategy
   ~seemps.state.Simplification
   ~seemps.state.Truncation