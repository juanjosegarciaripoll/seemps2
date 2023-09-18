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
    \mathrm{argmin}_\psi \frac{\langle{\psi|H|\psi}}{\langle{\psi|\psi}},

where the state :math:`\psi` is a matrix product state. The algorithm works by
reinterpreting this functional as a function over the individual tensors.

In a simple version of the algorithm, we could define :math:`E(A^{(k)})` as
the energy that results from substituting the k-th tensor by :math:`A^{(k)}`.
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

When looking at this form, we realize that, once we contract all tensors, this
expectation value is actually a quadratic form over the four-leg tensor :math:`X`.
We can therefore write something like

.. math::
    E_{2,3}[\vec{X}] = \frac{\vec{X}^T H_\text{eff} \vec{X}}{\vec{X}^T\vec{X}},

where :math:`H_\text{eff}` is an effective quadratic form that results from
contracting all MPS and MPO tensors except :math:`X` (which we have reshaped
as a vector :math:`\vec{X}`). This functional can be
minimized by solving the eigenvalue equation

.. math::
    H_\text{eff} \vec{X} = E_\text{min} \vec{X}.

The DMRG algorithm we implement would do this diagonalization for sites (2,3),
(3,4), (4,5) and so on and so forth, sweeping back and forth along the
composite quantum system, until the energy converges.

.. autosummary::
    :toctree: ../generated

    ~seemps.optimization.dmrg
