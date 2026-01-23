# SeeMPS

## Introduction

SeeMPS is a state of the art library for quantum-inspired algorithms
based on matrix product states (MPS), also known as tensor trains (TT)
or quantized tensor trains (QTT).

SeeMPS provides a complete linear and nonlinear algebra package that
operates with vectors represented as MPS/TT and with matrices represented
as matrix product operators (MPO). More precisely, the library includes:
- Vector and matrix operations
- Essential tasks, such as MPS and MPO compression.
- Solvers of linear equations `A x = b` with MPO (`A`) and MPS (`x`, `b`).
- Eigensolvers

This linear algebra package is used to implement both quantum and quantum-inspired
solutions. The first domain includes:
- Algorithms to construct Hamiltonians for many-body quantum systems
- Search of ground and excited states
- Simulation of time evolution
- Emulation of quantum registers and variational quantum circuits

The quantum-inspired subpackage includes:
- Tools to represent n-dimensional functions as MPS, including:
  + Exact representations
  + Polynomial expansions
  + Tensor cross-interpolation algorithms
- Algorithms for differentiation, integration and interpolatoin
- Algorithms to solve static and time-dependent PDEs in MPS/TT representation

These methods have shown exponential improvements over classical versions of the
same tools thanks to the memory and time compression offered by the MPS formalism.

## Intended audience

The library is a performant tool that is suitable for the design and evaluation of
algorithms in the quantum many-body systems, quantum computing and quantum inspired
numerical analysis. However, the library also allows users to quickly develop
applications in all of these domains, using the tens of algorithms that are already
provided.

The library as it stands has been used in some heavy-duty simulations
involving tens and hundreds of qubits, and, in particular, its current iteration
arises from the following works on quantum-inspired algorithms for numerical analysis:

- *Quantum-inspired algorithms for multivariate analysis: from interpolation to partial differential equations*,
  Juan José García-Ripoll, Quantum 5, 431 (2021), https://doi.org/10.22331/q-2021-04-15-431

- *Global optimization of MPS in quantum-inspired numerical analysis*,
  Paula García-Molina, Luca Tagliacozzo, Juan José García-Ripoll,
  https://arxiv.org/abs/2303.09430

- *Chebyshev approximation and composition of functions in matrix product states for quantum-inspired numerical analysis*,
  Juan José Rodríguez-Aldavero, Paula García-Molina, Luca Tagliacozzo, Juan José García-Ripoll
  https://arxiv.org/abs/2407.09609
  
- *Pseudospectral method for solving PDEs using matrix product states*,
  Jorge Gidi, Paula García-Molina, Luca Tagliacozzo, Juan José García-Ripoll, Journal of Computational Physics, 539 (2025), https://doi.org/10.1016/j.jcp.2025.114228

## Usage

The library is developed in a mixture of Python 3 and Cython, with the support
of Numpy, Scipy and h5py. The library is provided in binary format for a selection
of platforms and can be installed using
```
pip intall seemps
```
Other installation and development instructions are provided in
[the documentation](https://seemps.readthedocs.io), together with extensive
information about the algorithms. Examples are also provided in a
[separate folder](https://github.com/juanjosegarciaripoll/seemps2/tree/main/examples)
as well as in the documentation.

Bugs and feature requests can be reported using [GitHub's issues](https://github.com/juanjosegarciaripoll/seemps2/issues).
We also accept contributions as pull requests via GitHub.

# Authors and contributors

* Juan José García Ripoll (Institute of Fundamental Physics, IFF-CSIC, Spain)
* Paula García Molina (Institute of Fundamental Physics,  IFF-CSIC, Spain)
* Juan José Rodríguez Aldavero (Institute of Fundamental Physics, IFF-CSIC, Spain)
* Jorge Gidi (Universidad de Concepción, Chile)

