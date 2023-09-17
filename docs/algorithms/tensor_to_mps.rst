.. _vector_to_mps:

***********************************
Creating an MPS from a state vector
***********************************

The Schmidt decomposition can be used sequentially, to split a multilegged
tensor representing a quantum state, such as :math:`\psi_{ijk}`, or
:math:`\psi_{i_1i_2i_3\cdots i_N}` into a collection of three-legged tensors
(See :doc:`matrix-product state objects <../seemps_objects_mps>`).

The rationale starts by understanding that :doc:`Schmidt decomposition <tensor_schmidt>` of the
bipartite system as the separation of a two-legged tensor tensor into
two smaller tensors. For instance, we can write

.. math::
    \psi_{ij} = B_{ik} A_{kj1}

identifying the matrices :doc:`from the SVD form <tensor_schmidt>`

.. math::
    B_{ik} = U_{ik} s_k, \; A_{kj1} = V_{kj}

The `k` index that bonds both tensors is called the "virtual dimension",
as opposed to the original indices `i` and `j`, which we associated to physical
subsystems.

This procedure can be extended to larger tensors wiht more legs. Let us take
an N-dimensional quantum system, whose state is represented by an N-leg tensor
and reinterpret it as a bipartite system with `N-1` and 1 component, respectively.

.. math::
    \psi_{i_1i_2\ldots i_{N-1}i_N} = \Psi_{I i_N}

Here, the index :math:`I` groups all the possible configurations from the previous
(N-1) indices and has a potentially large dimension, :math:`d_1\cdots d_{N-1}`.
But let us ignore this fact and use the Schmidt decomposition to write

.. math::
    \Psi_{I i_N} = B_{Ik} A^{(N)}_{ki_N1}
    = \Psi^{(N-1)}_{i_1\ldots i_{N-1}k} A_{k{i_N}1}

We can repeat this process iteratively, now separating

.. math::
    \Psi^{(N-1)}_{i_1\cdots i_{N-2}i_{N-1}k} =
    \Psi^{(N-2)}_{i_1\cdots i_{N-2}j} A^{(N-1)}_{ji_{N-1}k}

If repeated N times, the outcome is a decomposition of the whole tensor into
a contraction of 3-legged tensors:

.. math::
    \psi_{i_1i_2\ldots i_{N-1}i_N} = A^{(1)}_{1i_1\alpha_2}
    A^{(2)}_{\alpha_2i_2\alpha_3}\cdots A^{(N)}_{\alpha_Ni_N1}

In SeeMPS, this iterative procedure is performed in very different parts of the
code. For the user, this algorithm is offered through the class methods
:func:`~seemps.state.MPS.from_tensor`, which transforms an N-legged tensor
into a matrix-product state with N components.

Take for instance the following example, which creates an unnormalized random
state of 8 qubits, and then uses this decomposition to create a
matrix-product state::

    >>> from seemps import MPS
    >>> import numpy as np
    >>> state = np.random.randn(2,2,2,2,2,2,2,2)
    >>> mps = MPS.from_vector(state)

Alternatively, the same effect can be obtained from the wavefunction of
a composite quantum system represented as a vector. Naturally, because we are
inputting a vector, this function takes as an argument a list with the
dimensions of the quantum subsystems.

    >>> from seemps import MPS
    >>> import numpy as np
    >>> state = np.random.randn(2**8)
    >>> mps = MPS.from_vector(state, [2]*8)
