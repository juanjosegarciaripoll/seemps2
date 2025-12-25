.. _contributing:

************
Contributing
************

SeeMPS is an open source project that is distributed under a very liberal license. SeeMPS's source code is freely available at `GitHub <https://github.com/juanjosegarciaripoll/seemps2>`_ and may be immediately cloned using::

  git clone https://github.com/juanjosegarciaripoll/seemps2.git

Errors should be reported using `GitHub's issues <https://github.com/juanjosegarciaripoll/seemps2/issues>`_ with a careful description of the steps to reproduce whatever bug you found and, ideally, steps to fix it.

Contributed code must be supplied using pull request. This means you should first fork the project in `GitHub <https://github.com/juanjosegarciaripoll/seemps2>`_, work on your copy, commit whatever improvements you wish to submit and use the "Pull request" button to send those changes back to the main project. Note that contributors accept to distribute any such changes under the same license.

To contribute changes you must:

1. Use the recommended :ref:`development environment <environment>`.

2. Build and test all your changes using the :ref:`defined steps <testing>`.

3. Make sure you don't break the documentation either, by :ref:`building it too <documentation>`.

4. Ensure all functions you develop are documented with at least a docstring in Numpy format, and strict argument type declarations.


.. _environment:

Development requirements
------------------------

SeeMPS is developed in Python and Cython, with the help of some recommended tools:

- The source control tool `git`

- `Astral's uv <hhttps://docs.astral.sh/uv>`_ project manager.

- A C/C++ compiler to process sections of SeeMPS that are compiled natively with Cython.

- A copy of Visual Studio Code with Python extensions and some other recommended extensions:

  + [Coverage Gutters](https://marketplace.visualstudio.com/items?itemName=ryanluker.vscode-coverage-gutters)

  + [Mypy Type Checker](https://marketplace.visualstudio.com/items?itemName=ms-python.mypy-type-checker)

  + [Ruff support](https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff)

Linux
^^^^^

Some sections of SeeMPS are written in Cython and need to be processed using some sort of C++ compiler. This is done to speed up certain critical sections, such as tensor contractions that are too cumbersome to implement purely in Python.

In Linux, package managers typically offer a `python-devel` package that installs both Python's C libraries and a C and C++ compiler, typically GCC. For instance, in Debian I would normally do::

  sudo apt-get install python-devel

The `uv` project manager can be downloaded from `Astral's webpage <https://docs.astral.sh/uv/getting-started/installation/>`_ or using `pip`, as explained there.


Windows
^^^^^^^

In Windows, you should install some version of Visual Studio C++ (Community Edition), `uv`, `git` and Visual Studio code. The simplest way to do this is using the `winget` utility from the command line::

  winget install Microsoft.VisualStudio.2022.Community Microsoft.VisualStudioCode astral-sh.uv git.git

Make sure to keep your software up to date by running periodically the following command line instructions::

  winget upgrade --all

.. _testing:

Testing
-------

SeeMPS has a common script, called `make.py <https://github.com/juanjosegarciaripoll/seemps2/blob/main/scripts/make.py>`_, which takes care of many important tasks:

- Running the test suite.

- Checking types with `Mypy <https://mypy-lang.org/>`_ and `basedpyright <https://docs.basedpyright.com/latest/>`.

- Building the documentation using Sphinx.

- Linting the software using `Astral's ruff <https://docs.astral.sh/ruff/>`_.

While editing, we recommend that you activate type checking in Visual Studio Code using standard pyright. This detects quite many errors. Contributions are rejected if there are typing errors.

Once you have completed a change, it is recommended to run the following set of steps, which are all the checks that we run in GitHub prior to a pull request::

    uv run scripts/make.py --test
    uv run scripts/make.py --mypy
    uv run scripts/make.py --basedpyright
    uv run scripts/make.py --ruff

These are, in order, the unit tests that verify all functions, two type checkers (Mypy and basedpyright), and the linter.

.. _documentation:

Documentation
-------------

The documentation is automatically built for every release by `readthedocs <https://seemps.readthedocs.io>`_. However, it is recommended thatyou build this documentation locally, when developing new code::

    uv sync --group doc
    uv run scripts/make.py --doc

This will ensure that the documentation is consistent and you have not removed some function that is referenced there. As a bonus, it will provide you with a local copy of alldocs under the `_site` folder, which you can open using, for instance::

    firefox _site/index.html

