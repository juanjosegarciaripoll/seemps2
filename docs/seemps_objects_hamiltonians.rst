.. _hamiltonian_classes:

************
Hamiltonians
************

In addition to states, we provide some convenience classes to represent quantum
Hamiltonians acting on composite quantum systems. These Hamiltonians can be
constant, time-dependent or translationally invariant. They can be converted
to matrices, tensors or matrix-product operators.

Nearest-neighbor Hamiltonians
=============================

The basic class is :class:`~seemps.hamiltonians.NNHamiltonian`,
an abstract object representing a sum of nearest-neighbor operators :math:`H = \sum_{i=0}^{N-2} h_{i,i+1}`
where each :math:`h_{i,i+1}` acts on a different, consecutive pair of quantum
objects. This class is extended by different convenience classes that simplify
the construction of such models, or provide specific, well-known ones:

.. autosummary::

    ~seemps.hamiltonians.NNHamiltonian
    ~seemps.hamiltonians.ConstantNNHamiltonian
    ~seemps.hamiltonians.ConstantTIHamiltonian
    ~seemps.hamiltonians.HeisenbergHamiltonian

As example of use, we provide the :class:`~seemps.hamiltonians.HeisenbergHamiltonian` class,
which creates the model :math:`\sum_i \vec{S}_i\cdot\vec{S}_{i+1}` more or less
like this::

    >>> SdotS = 0.25 * (sp.kron(σx, σx) + sp.kron(σy, σy) + sp.kron(σz, σz))
    >>> ConstantTIHamiltonian(size, SdotS)

These classes are required for algorithms such as TEBD and Trotter evolution
(see :doc:`algorithms/tebd_evolution`) that need access to the individual
nearest-neighbor interaction terms.

General Hamiltonian construction
================================

The more advanced class is :class:`~seemps.hamiltonians.InteractionGraph`, an
object that can record all types of interactions in a quantum system and produce
both the :class:`~seemps.operators.MPO` and sparse matrix representation for it.
Unlike typical DMRG libraries, all SeeMPS algorithms work with matrix product
operators, so this class provides a convenient way to construct MPOs from
physical models.

.. autosummary::

    ~seemps.hamiltonians.InteractionGraph

The class works by collecting interacting terms in a database. This database is
then used to create an artificial MPS that represents all the interactions and
local terms. The MPS is brought into the simplest form possible with a compression
stage, and it is then used to recreate the final MPO. This technique allows the
algorithm to discover that nearest neighbor interactions :math:`\sum_i J_i\sigma^z_i\sigma^z_{i+1}`
can be written with bond dimension two, and other interesting simplifications.

Supported interaction types
---------------------------

:class:`~seemps.hamiltonians.InteractionGraph` supports three types of terms:

1. **Local terms**: :math:`\sum_i h_i O_i`
2. **Nearest-neighbor interactions**: :math:`\sum_i J_i O_i O_{i+1}`
3. **Long-range interactions**: :math:`\sum_{ij} J_{ij} O_i O_j`

Example: Heisenberg model
-------------------------

The Heisenberg model :math:`H = \sum_i \vec{S}_i \cdot \vec{S}_{i+1}` can be created as::

    >>> ig = InteractionGraph(dimensions=[2] * 10)
    >>> for O in [σx, σy, σz]:
    ...     ig.add_nearest_neighbor_interaction(O, O)
    >>> mpo = ig.to_mpo()

Example: Long-range Ising model
-------------------------------

Any kind of long-range interactions can be created::

    >>> L = 10
    >>> long_range_Ising = InteractionGraph(dimensions=[2] * L)
    >>> J = np.random.normal(size=(L, L))
    >>> long_range_Ising.add_long_range_interaction(J, σx, σx)
    >>> mpo = long_range_Ising.to_mpo()

Example: Transverse-field Ising model
-------------------------------------

Combining local and interaction terms::

    >>> L = 10
    >>> h = 0.5  # transverse field strength
    >>> J = 1.0  # coupling strength
    >>> tfim = InteractionGraph(dimensions=[2] * L)
    >>> tfim.add_identical_local_terms(h * σx)  # transverse field
    >>> tfim.add_nearest_neighbor_interaction(σz, σz, weights=J)
    >>> mpo = tfim.to_mpo()

Hermiticity preservation
------------------------

By applying the simplification stage onto the internal MPS representation and
not the MPO directly, the algorithm ensures that Hermiticity is preserved,
even if the final operator is within machine precision of the desired Hamiltonian.

See also
========

- :doc:`algorithms/dmrg` - Ground state search using MPO Hamiltonians
- :doc:`algorithms/tebd_evolution` - Time evolution for nearest-neighbor Hamiltonians
- :doc:`seemps_register` - Quantum circuits using Hamiltonian evolution layers

