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
