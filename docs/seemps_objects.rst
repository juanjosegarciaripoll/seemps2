.. _seemps_objects:

***************
Quantum objects
***************

SeeMPS is built around tensor network representations of vectors and linear operators
in exponentially large vector spaces. The core data structures are:

- :class:`~seemps.state.MPS`: Matrix Product States represent vectors using a chain of
  three-legged tensors. They can encode quantum states of composite systems or, more
  generally, any high-dimensional vector with subexponential memory when correlations
  are bounded.

- :class:`~seemps.state.CanonicalMPS`: A specialized MPS in canonical form, where tensors
  satisfy orthonormality conditions that simplify many operations and improve numerical
  stability.

- :class:`~seemps.state.MPSSum`: A lazy representation of weighted sums of MPS, useful
  for intermediate computations before simplification.

- :class:`~seemps.operators.MPO`: Matrix Product Operators represent linear transformations
  using a chain of four-legged tensors. They can encode Hamiltonians, evolution operators,
  or any linear map between MPS.

These structures form the foundation for all algorithms in SeeMPS, from basic linear
algebra operations to advanced eigensolvers and PDE integrators.

.. toctree::
   :maxdepth: 1

   seemps_objects_mps
   seemps_objects_canonical
   seemps_objects_sum
   seemps_objects_mpo
   seemps_objects_hamiltonians