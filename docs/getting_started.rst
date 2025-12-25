.. _getting_started:

***************
Getting started
***************

Installation
------------
The SeeMPS library can be installed directly using pip::

  pip install seemps

Note that, since we do not provide binaries for all Python versions and all platforms, you might at some point need to have a C++ compiler available, if `pip` or `uv` finds that it has to compile SeeMPS from sources (see :ref:`the development environment instructions <environment>` for more details).

More generally, we recommend that you use `Astral's uv <https://docs.astral.sh/uv>` to manage any Python project you create. In that case, you simply need to list SeeMPS as a requirement for your project::

  uv add seemps

and whenever you recreate the environment using `uv sync`, the library will be pulled and installed.

First usage
-----------
If you have never used SeeMPS, we recommend you simply clone the repository and try the examples there. For this, you have to first install Astral uv (see :ref:`the steps to do it <environment>`) and directly run the examples from the `notebook directories <seemps_examples>`_.

If you are using Visual Studio Code, as recommended, you would simply do the following from the command line::

  git clone https://github.com/juanjosegarciaripoll/seemps2
  cd seemps2
  uv sync
  uv pip install ipykernel
  code examples/DMRG.ipynb

These steps:

1. Clone the repository from GitHub.

2. Enter the right folder

3. Install the library and its dependencies in a local environment (a hidden folder with name `.venv`).

4. Install Jupyter's kernel to be able to run notebooks.

5. Open some notebook in Visual Studio Code.