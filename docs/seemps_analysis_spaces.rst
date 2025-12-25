.. _analysis_spaces:

*****************************************
Uniform Grids and Affine Transformations
*****************************************

Numerical analysis problems are defined on a discretized space. Given a one-dimensional
problem with :math:`x \in [a,b), L=|b-a|`, the discretized grid is defined for the points

.. math::
    x_s = a + s\Delta x, \quad \Delta x = \frac{L}{N},

where :math:`s=0,\dots,N-1`.

Ref. :cite:t:`GarciaMolina2022` presents a set of quantum Fourier analysis techniques
to efficiently represent functions and operators in an `n`-qubit quantum register.
By setting `N=2^n`, each of the points `x_s` of the grid maps to a quantum state `s`
of the quantum register. This leads to a binary encoding of the coordinates

.. math::
    x_s^{(n)} = x_{(s_0s_1\dots s_{N-1})}^{(n)} = a + \Delta{x}^{(n)} \sum_{k=0}^{n-1}s_k 2^{n-k-1}.

Since MPS are representations of quantum states, they are also suitable for this quantum
register representation, by mapping the qubits to the physical indices of the MPS.

It is also possible to use a symmetric definition of the position space interval, by setting `\Delta x = L/(N-1)`.

Multidimensional functions
--------------------------

Let us consider a `d` dimensional function :math:`f(\mathbf{x})=f(x_1, x_2,\dots,x_d)` with a definition interval :math:`x_i \in [a_i,b_i)`
of size :math:`L_{x_i}=|b_i-a_i|$` for each dimension :math:`i=(1,\dots,d)`. An `n`-qubit quantum register
encodes this function, using :math:`n_i`` qubits to represent each coordinate over :math:`2^{n_i}` points as

.. math::
	x_{i,s_i}^{(n_i)}=a_i+s_i \Delta x_i^{(n_i)}.

The integer :math:`s_i\in\lbrace0,1, \dots, 2^{n_i}-1\rbrace` labels the grid coordinate :math:`x_{i,s_i}^{(n_i)}` of the
`i`-th dimension. Then, a set of labels :math:`\mathbf{s}=(s_1,s_2,\dots,s_d)` represents the `d`-dimensional state for the coordinate
:math:`\mathbf{x}_\mathbf{s}$`. For a multidimensional function the set of labels :math:`\mathbf{s}=(s_{1}, s_{2}, \dots, s_{d})`
results from grouping the states of the :math:`n=\sum n_i` qubits in some order. They may be labeled
sequentially, :math:`s_1:=s_{11},s_2:=s_{12},\ldots`, or by significance, :math:`s_1:=s_{11},\,s_2:=s_{21},\ldots`.

The :class::`Space` creates an object to define the problem's coordinates for a multidimensional problem.

.. autosummary::

    ~seemps.analysis.space.Space
    ~seemps.analysis.space.mpo_flip

Implicit representations
------------------------
Alternatively, the multivariate spaces can be represented implicitly using the :class:`~seemps.analysis.mesh.Interval` and :class:`~seemps.analysis.mesh.Mesh` classes.

Essentially, the :class:`~seemps.analysis.mesh.Interval` class represents an univariate discretization implicitly, and can be indexed similarly as an explicit array. Then, the :class:`~seemps.analysis.mesh.Mesh` class represents a multivariate space as a collection of :class:`~seemps.analysis.mesh.Interval` objects. These objects can be indexed using multidimensional indices similarly as explicit multivariate arrays, without explicitly containing them and avoiding an exponential memory overhead.

Currently, there are three types of :class:`~seemps.analysis.mesh.Interval` implemented:

- :class:`~seemps.analysis.mesh.RegularInterval`: An interval representing a regular discretization.
- :class:`~seemps.analysis.mesh.ChebyshevInterval`: An interval representing an irregular discretization on the Chebyshev zeros or extrema.
- :class:`~seemps.analysis.mesh.IntegerInterval`: An interval representing a regular discretization with integers.

.. autosummary::

    ~seemps.analysis.mesh.Mesh
    ~seemps.analysis.mesh.Interval
    ~seemps.analysis.mesh.RegularInterval
    ~seemps.analysis.mesh.ChebyshevInterval
    ~seemps.analysis.mesh.IntegerInterval