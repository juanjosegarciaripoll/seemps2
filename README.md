# SeeMPS

## Introduction

SEEMPS is the second iteration of the SElf-Explaining Matrix-Product-State
library.

The original library, still available [here](https://github.com/juanjosegarciaripoll/seemps)
was a collection of Jupyter notebooks with a well documented implementation
of matrix-product state algorithms.

The [current iteration](https://github.com/juanjosegarciaripoll/seemps2) aims to
be more useable and have better and more standard documentation, while
preserving the same degree of accessibility of the algorithms.

## Intended audience

The library is thought out as introduction to the world of Matrix Product States
and DMRG-inspired algorithms. Its main goal is not performance, but rapid
prototyping and testing of ideas, providing a good playground before dwelling in
more advanced (C++, Julia) versions of the algorithms.

This said, the library as it stands has been used in some heavy-duty simulations
involving tens and hundreds of qubits, and, in particular, its current iteration
arises from two works on quantum-inspired algorithms for numerical analysis:

- *Quantum-inspired algorithms for multivariate analysis: from interpolation to partial differential equations*,
  Juan José García-Ripoll, Quantum 5, 431 (2021), https://doi.org/10.22331/q-2021-04-15-431

- *Global optimization of MPS in quantum-inspired numerical analysis*,
  Paula García-Molina, Luca Tagliacozzo, Juan José García-Ripoll,
  https://arxiv.org/abs/2303.09430

## Usage

The library is developed in a mixture of Python 3 and Cython, with the support
of Numpy, Scipy and h5py. Installation instructions are provided in
[the documentation](https://juanjosegarciaripoll.github.io/seemps2).

Authors:

* Juan José García Ripoll (Institute of Fundamental Physics)
* Paula García Molina (Institute of Fundamental Physics)
* Juan José Rodríguez Aldavero (Institute of Fundamental Physics)

Contributors:

* Jorge Gidi

## TODOs
- Update documentation.
- Pull request with local developments (computational tree, Hadamard sketching (HaTT), etc.) and breaking changes (TT-cross, orthogonal polynomials, integration, etc.).
- Add problem-specific tools: quantile estimation with binary search (for Value at Risk), MPO cumulative sum.
- Revise test suite.
- Revise RK45.
- Revise Radau.
