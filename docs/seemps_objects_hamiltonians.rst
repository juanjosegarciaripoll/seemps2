.. _hamiltonian_classes:

************
Hamiltonians
************

In addition to states, we provide some convenience classes to represent quantum
Hamiltonians acting on composite quantum systems. These Hamiltonians can be
constant, time-dependent or translationally invariant. They can be converted
to matrices, tensors or matrix-product operators.

The basic class is the :class:`NNHamiltonian`, an abstract object representing
a sum of nearest-neighbor operators :math:`H = \sum_{i=0}^{N-2} h_{i,i+1}`
where each :math:`h_{i,i+1}` acts on a different, consecutive pair of quantum
objects. This class is extended by different convenience classes that simplify
the construction of such models, or provide specific, well-known ones:

.. autosummary::

    ~seemps.hamiltonians.NNHamiltonian
    ~seemps.hamiltonians.ConstantNNHamiltonian
    ~seemps.hamiltonians.ConstantTIHamiltonian
    ~seemps.hamiltonians.HeisenbergHamiltonian

As example of use, we can inspect the :class:`~seemps.hamiltonians.HeisenbergHamiltonian` class,
which creates the model :math:`\sum_i \vec{S}_i\cdot\vec{S}_{i+1}` more or less
like this::

    >>> SdotS = 0.25 * (sp.kron(σx, σx) + sp.kron(σy, σy) + sp.kron(σz, σz))
    >>> ConstantTIHamiltonian(size, SdotS)


The more advanced class is :class:`~seemps.hamiltonains.InteractionGraph`, an
object that can record all types of interactions in a quantum system and produce
both the :class:`MPO` and sparse matrix representation for it. This class is
preferred over the previous ones, except in some algorithms, such as TEBD and
Trotter evolution, that require handling the internals of the physical model.

As example of use, the same Heisenberg model would be created using::

    >>> ig = InteractionGraph(dimensions=[2] * 10)
    >>> for O in [σx, σy, σz]:
    ...     ig.add_nearest_neighbor_interaction(O, O)
    >>> mpo = ig.to_mpo()

However, we can create also any kind of long-range interactions using this
technique::

    >>> L = 10
    >>> long_range_Ising = InteractionGraph(dimensions=[2] * L)
    >>> J = np.random.normal(size=(L, L))
    >>> long_range_Ising.add_long_range_interaction(J, σx, σx)
    >>> mpo = long_range_Ising.to_mpo()

