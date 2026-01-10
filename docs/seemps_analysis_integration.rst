.. currentmodule:: seemps.analysis.integration

.. _analysis_integration:

********************
Function Integration
********************

Functions encoded in MPS/TT form can be efficiently integrated by contracting them with MPS encodings of weights that implement some quadrature formula. For instance, a simple Riemann approximation results from the addition of all values of the functions, weighted by the interval size, which are equivalent to the contraction between the representation of :math:`f(x)` and the identity function :math:`g(x)=1`

.. math::
    \int f(x)\mathrm{d}x \simeq \sum_i f(x_i) \Delta{x} = \langle g | f\rangle.

In this scenario, the quadrature corresponding to the state :math:`\langle g |` is given by the midpoint quadrature rule. More sophisticated quadrature rules result in more efficient convergence rates---i.e. requiring less nodes or tensor cores to compute an accurate estimation of the true integral.

In the following table we outline implemented routines that construct the states associated to various quadratures---i.e. ``mps_*`` functions---and a routine that implements the integral using any of those rules :func:`integrate_mps`. These routines are valid for input MPS with binary physical dimension, following the quantics representation, and divide in two families:

Newton-Côtes quadratures
------------------------
These are useful to integrate equispaced discretizations, each of increasing order. Compatible with discretizations stemming from :class:`~seemps.analysis.mesh.RegularInterval` objects. The larger the order, the better the convergence rate. However, large-order quadratures impose restrictions on the amounts of qubits they support:

- :func:`mps_simpson` requires a number of qubits divisible by 2.
- :func:`mps_fifth_order` requires a number of qubits divisible by 4.

.. autosummary::

    mps_trapezoidal
    mps_simpson38
    mps_fifth_order

Clenshaw-Curtis quadratures
---------------------------
These are useful to integrate irregular discretizations on either the Chebyshev zeros (Chebyshev-Gauss nodes) or the Chebyshev extrema (Chebyshev-Lobatto nodes). These have an exponentially better rate of convergence than the Newton-Côtes ones. Compatible with discretizations stemming from :class:`~seemps.analysis.mesh.ChebyshevInterval` objects.

.. autosummary::

    mps_fejer
    mps_clenshaw_curtis

Integration
-----------
The standard method for integration consists in first constructing the multivariate quadrature rule using the previous routines, together with :class:`~seemps.analysis.factories.mps_tensor_product` and :class:`~seemps.analysis.factories.mps_tensor_sum` tensorized operations. Then, this quadrature is to be contracted with the desired MPS target using the scalar product routine :class:`~seemps.state.scprod`. However, for ease of use, a helper routine :class:`~seemps.analysis.integration.integrate_mps` is given that automatically computes the best possible quadrature rule associated to a :class:`~seemps.analysis.mesh.Mesh` object, and contracts with the target MPS to compute the integral:

.. autosummary::

    integrate_mps

Both this helper routine and the MPS quadrature rules are only valid for standard function representations in MPS with binary quantization. For more general structures, integration using tensor cross-interpolation (TCI) is preferred, using the :func:`~seemps.analysis.cross.cross_interpolation` routine. Helper methods exist for the general case, requiring an input :class:`~seemps.analysis.mesh.Mesh` object containing the quadrature rules. The function :func:`~seemps.analysis.integration.quadrature_mesh_to_mps` transforms this mesh into a quadrature MPS which may be contracted with the input function using the inner product :func:`~seemps.state.scprod`.

.. autosummary::
    
    ~mesh_to_quadrature_mesh
    ~quadrature_mesh_to_mps

An example on how to use these functions is shown in `Integration.ipynb <https://github.com/juanjosegarciaripoll/seemps2/blob/main/examples/Integration.ipynb>`_.