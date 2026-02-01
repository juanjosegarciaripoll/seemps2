************************************
Density-Matrix Renormalization Group
************************************

The DMRG algorithm was an iterative procedure to approximately diagonalize
Hamiltonians, which was shown to produce matrix-product states. These days, a
slightly more modern version of the algorithm is used, which directly operates
in the MPS formalism.

The goal of this algorithm, as we are going to use it, is to compute the
minimum of an energy functional:

.. math::
    \mathrm{argmin}_\psi E[\psi] :=
    \mathrm{argmin}_\psi \frac{\langle{\psi|H|\psi}\rangle}{\langle{\psi|\psi}\rangle},

where the state :math:`\psi` is a matrix product state. The algorithm works by
reinterpreting this functional as a function over the individual tensors.

In a simple version of the algorithm, we could define :math:`E(X^{(k)})` as
the energy that results from substituting the k-th tensor by :math:`X^{(k)}`.
As the picture below shows, the numerator and denominator are quadratic forms
of the only tensor involved:

.. image:: ../pictures/dmrg-tensor-substitution-one-site.drawio.svg
    :align: center

We can therefore write something like

.. math::
    E_{2,3}[\vec{X}] = \frac{\vec{X}^T H_\text{eff} \vec{X}}{\vec{X}^T\vec{X}},

where :math:`H_\text{eff}` is an effective quadratic form that results from
contracting all MPS and MPO tensors except :math:`X` (which we have reshaped
as a vector :math:`\vec{X}`). This functional can be
minimized by solving the eigenvalue equation

.. math::
    H_\text{eff} \vec{X} = E_\text{min} \vec{X}.

In that algorithm, we would find the best tensor for the site `k=1`,
substitute that tensor into the state; compute the best tensor for the site `k=2`,
and so on and so forth.

A more clever version works by operating on pairs of tensors. The idea is that
we re-create the same MPS by joining two tensors from neighboring sites. For
instance, the figure below sketches how we would look at the expectation value
of an MPO Hamiltonian :math:`H` over a state :math:`\psi` which is in canonical
form with respect to the second site, and where we have replaced
the second and third tensor by a combined one :math:`X`.

.. image:: ../pictures/dmrg-tensor-substitution.drawio.svg
    :width: 400

We would apply the same principles as before, solving an eigenvalue problem for
the two-site tensor :math:`X`, but now we add a second step in which this tensor
is optimally split into two smaller tensors, as
:doc:`explained in this manual <tensor_split>`. Note that, while in
the single-site algorithm the size of the tensor :math:`X` is fixed, in this
algorithm the splitting of the two-site tensor will produce objects whose size
will dynamically adapt the size of the virtual dimension---i.e. the amount of
entanglement and correlations---to the needs of the problem-

The DMRG in SeeMPS implements this variant of the algorithm, sweeping over
pairs of sites---e.g., (1,2), then (2,3), then (3,4), and so on---back and forth
until the energy and the state converge.

Example
=======

The following example computes the ground state of a transverse-field Ising model:

.. math::
    H = \sum_{i=1}^{N-1} \sigma^z_i \sigma^z_{i+1} + h \sum_i \sigma^x_i

.. code-block:: python

    import numpy as np
    from seemps.hamiltonians import ConstantTIHamiltonian
    from seemps.optimization import dmrg
    from seemps.state import product_state

    # Define the Hamiltonian
    N = 20
    h = 0.5
    sx = np.array([[0, 1], [1, 0]])
    sz = np.array([[1, 0], [0, -1]])
    H = ConstantTIHamiltonian(
        size=N,
        interaction=np.kron(sz, sz),
        local_term=h * sx
    )

    # Initial guess: product state with all spins in +x direction
    guess = product_state(np.ones(2) / np.sqrt(2), N)

    # Run DMRG
    result = dmrg(H, guess=guess)
    print(f"Ground state energy: {result.energy}")

.. autosummary::

    ~seemps.optimization.dmrg

See also
========

- :doc:`gradient_descent` - A simpler optimization algorithm for finding ground states
- :doc:`arnoldi` - Krylov-based optimization using an expanded basis
- :func:`~seemps.solve.dmrg_solve` - DMRG-based solver for systems of linear equations
