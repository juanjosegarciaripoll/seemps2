.. _mps_tebd:

*******************
TEBD Time evolution
*******************

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

   ~seemps.evolution.trotter.Trotter2ndOrder
   ~seemps.evolution.trotter.Trotter3rdOrder

The following is an example evolving a matrix-product state with 20 qubits
under a spin-1/2 Heisenberg Hamiltonian::

   >>> import seemps
   >>> mps = seemps.random_uniform_mps(2, 20)
   >>> H = seemps.hamiltonians.HeisenbergHamiltonian(20)
   >>> dt = 0.1
   >>> U = seemps.evolution.trotter.Trotter2ndOrder(H, dt)
   >>> strategy = seemps.state.DEFAULT_STRATEGY.replace(tolerance = 1e-8)
   >>> t = 0.0
   >>> for steps in range(0, 50):
   ...   mps = U.apply_inplace(mps)
   ...   t += dt
   >>> mps
   <seemps.state.canonical_mps.CanonicalMPS object at 0x000002166AFFC1D0>
