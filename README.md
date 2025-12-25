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
arises from the following works on quantum-inspired algorithms for numerical analysis:

- *Quantum-inspired algorithms for multivariate analysis: from interpolation to partial differential equations*,
  Juan José García-Ripoll, Quantum 5, 431 (2021), https://doi.org/10.22331/q-2021-04-15-431

- *Global optimization of MPS in quantum-inspired numerical analysis*,
  Paula García-Molina, Luca Tagliacozzo, Juan José García-Ripoll,
  https://arxiv.org/abs/2303.09430

- *Chebyshev approximation and composition of functions in matrix product states for quantum-inspired numerical analysis*,
  Juan José Rodríguez-Aldavero, Paula García-Molina, Luca Tagliacozzo, Juan José García-Ripoll
  https://arxiv.org/abs/2407.09609

## Usage

The library is developed in a mixture of Python 3 and Cython, with the support
of Numpy, Scipy and h5py. Installation instructions are provided in
[the documentation](https://seemps.readthedocs.io).

Authors:

* Juan José García Ripoll (Institute of Fundamental Physics)
* Paula García Molina (Institute of Fundamental Physics)
* Juan José Rodríguez Aldavero (Institute of Fundamental Physics)

Contributors:

* Jorge Gidi

## Development

### Environment
For optimal development the following is expected:
- uv from Astral is installed
- In Linux, if you wish to use a local version of Python, you might need
  to install the `python-devel` package or equivalent one. This also
  installs a C and C++ compilers.
- In Windows, you need to install a Visual Studio C++ (Community Edition)
  compiler to build SeeMPS.
- A copy of Visual Code with the Python extensions installed plus some
  additional recommended extensions:
  + [Coverage Gutters](https://marketplace.visualstudio.com/items?itemName=ryanluker.vscode-coverage-gutters)
  + [Mypy Type Checker](https://marketplace.visualstudio.com/items?itemName=ms-python.mypy-type-checker)
  + [Ruff support](https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff)

The environment is bootstrapped using
```
uv sync --dev
```
This installs both the SeeMPS library and libraries that it depends on,
plus additional tools that are used for development:
- ruff, for code linting
- mypy and basedpyright, for type checking
- coverage, for code coverage

On top of this, please use
```
uv run scripts/make.py --install-hooks
```
to ensure type checkers and other tests are run before committing changes
with git.

### Testing
The library contains a rather complete set of unittests under the `tests/`
folder. The tests can be run using the standard `unittest` module, as in
```
uv run python -m unittest -v
```

The code coverage of the test suite exceeds 88%. To analyze test coverage
you can open a terminal and run
```
uv run coverage run -m unittest -v && uv run coverage report
```
Alternatively, you can use
```
uv run coverage lcov
```
to create a coverage file that is interpreted by the "Coverage Gutters"
Visual Code extension. There is a task (right-button option in the explorer)
with the name "Run Tests with Coverage" that both runs the tests and
automatically creates the reports using
```
uv run coverage run -m unittest -v && uv run coverage lcov
```

## TODOs
- Update documentation.
- Many functions are declared to accept Interval, when they actually can only
  use RegularInterval or ChebyshevInterval
