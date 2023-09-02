.. _seemps_algorithms:

**************
MPS algorithms
**************

.. _mps_truncate:

MPS simplification
------------------

The first and most fundamental algorith, on top of which all other algorithms
can be rigorously constructed, is the simplification. This is the search for
a matrix-product state :math:`\xi` that approximates another matrix-product
state :math:`\psi`, with the goal to make it simpler: i.e., typically reduce
the size of the bond dimensions.

Mathematically, we are solving the minimization of the norm-2 distance:

.. math::
   \mathrm{argmin}_{\xi \in \mathcal{MPS}_{D'}} \Vert{\xi-\psi}\Vert^2

There are two variants of the algorithm. The first one
:func:`~seemps.truncate.simplify` approximates just a single state. The second
one approximates a linear combination of states and weights :math:`\psi_i` and
:math:`w_i`, as in

.. math::
   \mathrm{argmin}_{\xi \in \mathcal{MPS}_{D'}} \Vert{\xi- \sum w_i \psi_i}\Vert^2

This second algorithm is the one used to convert :class:`seemps.state.MPSSum`
objects into ordinary :class:`seemps.state.MPS` states (see
:doc:`MPS combination <seemps_objects_sum>`) 

.. autosummary::
   :toctree: generated/

   ~seemps.truncate.simplify
   ~seemps.truncate.combine
   ~seemps.state.MPSSum.toMPS

.. _mps_tebd:

TEBD Time evolution
-------------------

The second but better known algorithm is the time-evolving block decimation
method to approximate the evolution of a quantum state. This algorithm solves
the Schrödinger equation

.. math::
   i \partial_t |\psi\rangle = H|\psi\rangle

with a Hamiltonian that consists entirely of nearest-neighbor interactions

.. math::
   H = \sum_{i=1}^{N-1} h_{i,i+1}

The algorithm proceeds by applying local gates made of small time-steps
:math:`\exp(-i \delta{t} h_{i,i+1})`, adapting the matrix-product state so that
the bond dimension does not grow too much.

.. autosummary::
   :toctree: generated/

   ~seemps.evolution.Trotter2ndOrder
   ~seemps.evolution.Trotter3rdOrder

The following is an example evolving a matrix-product state with 20 qubits
under a spin-1/2 Heisenberg Hamiltonian::

   >>> import seemps
   >>> mps = seemps.random_mps(2, 20)
   >>> H = seemps.hamiltonians.HeisenbergHamiltonian(20)
   >>> dt = 0.1
   >>> U = seemps.Trotter2ndOrder(H, dt)
   >>> strategy = seemps.DEFAULT_STRATEGY.replace(tolerance = 1e-8)
   >>> t = 0.0
   >>> for steps in range(0, 50):
   ...   mps = U.apply_inplace(mps)
   ...   t += dt
   >>> mps
   <seemps.state.canonical_mps.CanonicalMPS object at 0x000002166AFFC1D0>
