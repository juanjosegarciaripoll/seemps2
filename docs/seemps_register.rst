.. _seemps_register:

*****************
Quantum registers
*****************

The MPS is a convenient representation to store, manipulate and interrogate a
quantum register of qubits. SeeMPS can be used to classically simulate weakly
entangled computations, as MPS provide an efficient representation for quantum
states with limited entanglement.

SeeMPS's architecture is well suited for quantum circuit emulation:

- A single :class:`~seemps.operators.MPO` can represent a layer of quantum gates encoded with low bond dimension
- Several layers can be chained into an :class:`~seemps.operators.MPOList` to represent a quantum circuit
- Standard contraction and simplification routines study how circuits act on an MPS quantum register

.. _mps_register_circuit:

Parameterized quantum circuits
------------------------------

The module :py:mod:`seemps.register.circuits` provides a framework for variational
quantum algorithms. The main class is :class:`~seemps.register.circuit.ParameterizedLayeredCircuit`,
which is a collection of unitary operations with and without parameters.

Available layer types
~~~~~~~~~~~~~~~~~~~~~

.. autosummary::

   ~seemps.register.circuit.LocalRotationsLayer
   ~seemps.register.circuit.TwoQubitGatesLayer
   ~seemps.register.circuit.HamiltonianEvolutionLayer
   ~seemps.register.circuit.ParameterFreeMPO

- :class:`~seemps.register.circuit.LocalRotationsLayer`: Layers with local rotation gates on each qubit
- :class:`~seemps.register.circuit.TwoQubitGatesLayer`: Layers with entangling gates in a 1D nearest-neighbor architecture
- :class:`~seemps.register.circuit.HamiltonianEvolutionLayer`: Multi-qubit gate implementing evolution with a generic MPO Hamiltonian
- :class:`~seemps.register.circuit.ParameterFreeMPO`: Generic parameter-free layers encoded as MPO

Circuit composition
~~~~~~~~~~~~~~~~~~~

The :class:`~seemps.register.circuit.ParameterizedLayeredCircuit` is a
composite circuit that can be built from parameterized layers
as well as :class:`~seemps.operators.MPO` and :class:`~seemps.operators.MPOList`
transformations without parameters.

.. autosummary::

   ~seemps.register.circuit.ParameterizedLayeredCircuit

For example, the following two codes create equivalent operators, ``U3`` and ``U4``:

   >>> from seemps.register.circuits import *
   >>> U1 = LocalRotationsLayer(register_size=2, operator='Sz')
   >>> U2 = TwoQubitGatesLayer(register_size=2)
   >>> U3 = ParameterizedLayeredCircuit(register_size=2, layers=[U1, U2])
   >>> U4 = VQECircuit(register_size=2, layers=2)

Parameterized application
~~~~~~~~~~~~~~~~~~~~~~~~~

The circuits are "parameterized" because the rotation depends on angles that
are provided at construction time or when the unitaries are applied. For
instance, the following two codes produce the same state:

   >>> p = [0.13, -0.25]
   >>> U1 = LocalRotationsLayer(register_size=2, operator='Sz', same_angle=False, default_parameters=p)
   >>> state = U1.apply(state)

Or the alternative:

   >>> p = [0.13, -0.25]
   >>> U1 = LocalRotationsLayer(register_size=2, operator='Sz', same_angle=False)
   >>> state = U1.apply(state, p)

Pre-built variational circuits
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

SeeMPS provides pre-built circuits for common variational algorithms:

.. autosummary::

   ~seemps.register.circuit.VQECircuit
   ~seemps.register.circuit.IsingQAOACircuit

- :class:`~seemps.register.circuit.VQECircuit`: Hardware-efficient 1D VQE ansatz
- :class:`~seemps.register.circuit.IsingQAOACircuit`: Generic QAOA variational circuit for Ising models

These can be used for variational quantum eigensolver implementations, quantum
machine learning models, and analysis of variational states created by QAOA circuits.

.. _mps_qft:

Fourier transforms
------------------

SeeMPS also provides matrix-product operators and functions that implement
the quantum Fourier transform. In some cases the functions and MPO's act
over the whole quantum register (:func:`qft`, :func:`qft_mpo`,...) and in
other cases you can specify a subset of quantum systems (:func:`qft_nd_mpo`, etc).

.. autosummary::

   ~seemps.qft.qft
   ~seemps.qft.iqft
   ~seemps.qft.qft_mpo
   ~seemps.qft.iqft_mpo
   ~seemps.qft.qft_flip
   ~seemps.qft.qft_nd_mpo
   ~seemps.qft.iqft_nd_mpo

.. _mps_register_transformations:

Other transformations
---------------------

.. autosummary::

   ~seemps.register.twoscomplement
   ~seemps.register.qubo_mpo
   ~seemps.register.qubo_exponential_mpo
