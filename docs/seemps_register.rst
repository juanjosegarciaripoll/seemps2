.. _seemps_register:

*****************
Quantum registers
*****************

The MPS is a convenient representation to store, manipulate and interrogate a
quantum register of qubits. We now list some algorithms imported from the world
of quantum computing, which have been reimplemented using MPO's and MPS's.

.. _mps_register_circuit:

Quantum circuits
----------------

The module :py:mod:`seemps.register.circuits` provides a series of classes that
implement local and two-qubit transformations onto an MPS that encodes a quantum
register. These include the following ones:

.. autosummary::

   ~seemps.register.circuit.VQECircuit
   ~seemps.register.circuit.ParameterizedLayeredCircuit
   ~seemps.register.circuit.LocalRotationsLayer
   ~seemps.register.circuit.TwoQubitGatesLayer

The :class:`~seemps.register.circuit.ParameterizedLayeredCircuit` is a
composite circuit that can be built from :class:`~seemps.register.circuit.LocalRotationsLayer`
or :class:`~seemps.register.circuit.TwoQubitGatesLayer` circuits.

For example, the following two codes create equivalent operators, `U3` and `U4`:

   >>> from seemps.register.circuits import *
   >>> U1 = LocalRotationsLayer(register_size=2, operator='Sz')
   >>> U2 = TwoQubitGatesLayer(register_size=2)
   >>> U3 = ParameterizedLayeredCircuit(register_size=2, layers=[U1, U2])
   >>> U4 = VQECircuit(register_size=2, layers=2)

The circuits are "parameterized" because the rotation depends on angles that
are provided at construction time or when the unitaries are applied. For
instance, the following two codes produce the same state:

   >>> p = [0.13, -0.25]
   >>> U1 = LocalRotationsLayer(register_size=2, operator='Sz', same_angle=False, default_parameters=p)
   >>> state = U1.apply(state)

Or the alternative

   >>> p = [0.13, -0.25]
   >>> U1 = LocalRotationsLayer(register_size=2, operator='Sz', same_angle=False)
   >>> state = U1.apply(state, p)

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
