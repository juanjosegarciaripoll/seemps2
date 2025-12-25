# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config
import seemps
import importlib
import inspect
import os
from pathlib import Path

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))

# We need this for autodoc to find the modules it will document
import sys

sys.path.append("../")
import seemps.version

# -- Project information -----------------------------------------------------

project = "SeeMPS"
copyright = "2019, Juan Jose Garcia-Ripoll"
author = "Juan Jose Garcia-Ripoll"

# The full version, including alpha/beta/rc tags
release = seemps.version.number


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "numpydoc",  # Numpy documentation strings
    "sphinx.ext.autodoc",  # For using strings from classes/functions
    "sphinx.ext.mathjax",  # For using equations
    "sphinx.ext.doctest",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",  # Link to other project's doc.
    "sphinxcontrib.bibtex",  # Bibliography files
]
bibtex_bibfiles = ["refs.bib"]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", ".ipynb_checkpoints"]

# This is needed to fix readthedocs
master_doc = "index"

numpydoc_xref_param_type = True
numpydoc_show_class_members = False  # https://stackoverflow.com/a/34604043/5201771
numpydoc_attributes_as_param_list = False

autoclass_content = "class"
# TODO: Fix type hints. They should show somewhere.
# autodoc_typehints = "description"
autodoc_typehints_format = "short"
autodoc_preserve_defaults = True
autodoc_type_aliases = {
    "DenseOperator": "~seemps.typing.DenseOperator",
    "Operator": "~seemps.typing.Operator",
    "Vector": "~seemps.typing.Vector",
    "VectorLike": "~seemps.typing.VectorLike",
    "python:list": "list",
    "Weight": "Weight",
    "Strategy": "~seemps.state.Strategy",
}
autodoc_default_options = {
    "no-value": True,
    "exclude-members": "__init__",
    "inherited-members": False,
    "show-inheritance": True,
    "special-members": False,
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://www.numpy.org/devdocs", None),
    "scipy": ("https://scipy.github.io/devdocs", None),
}

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "pydata_sphinx_theme"


all_modules = [
    "seemps.analysis",
    "seemps.analysis.cross",
    "seemps.analysis.factories",
    "seemps.analysis.finite_differences",
    "seemps.analysis.hdaf",
    "seemps.analysis.integration",
    "seemps.analysis.interpolation",
    "seemps.analysis.lagrange",
    "seemps.analysis.mesh",
    "seemps.analysis.operators",
    "seemps.analysis.optimization",
    "seemps.analysis.space",
    "seemps.evolution",
    "seemps.expectation",
    "seemps.hdf5",
    "seemps.hamiltonians",
    "seemps.state",
    "seemps.register",
    "seemps.register.circuit",
    "seemps.typing",
    "seemps.operators",
    "seemps.optimization",
    "seemps.qft",
    "seemps.solve",
]


def generate_toplevel():
    path = Path(__file__).parent / "api" / "reference.rst"
    os.makedirs(path.parent, exist_ok=True)
    with open(path, "w") as file:
        print(
            r""".. _API:
API Reference
=============
.. toctree::
   :maxdepth: 1

   class/_definitions
   function/_definitions
   type/_definitions

""",
            file=file,
        )


def generate_hidden_toctrees():
    for folder in ["class", "function", "type"]:
        path = Path(__file__).parent / "api" / folder / "_definitions.rst"
        os.makedirs(path.parent, exist_ok=True)
        plural = folder + ("es" if folder[-1] == "s" else "s")
        title = f"All {plural}"
        underline = "=" * len(title)
        with open(path, "w") as file:
            print(
                rf""".. _{plural}:

{title}
{underline}

.. toctree::
   :glob:
   :maxdepth: 1

   *""",
                file=file,
            )


def generate_class(module_name, name):
    object_name = module_name + "." + name
    path = Path(__file__).parent / "api" / "class" / (object_name + ".rst")
    underscore = "=" * len(object_name)
    with open(path, "w") as file:
        print(f"Creating {path}")
        print(
            f"""{object_name}
{underscore}

.. currentmodule:: {module_name}

.. autoclass:: {name}
    :show-inheritance:
    :members:
""",
            file=file,
        )


def generate_function(module_name, name):
    object_name = module_name + "." + name
    path = Path(__file__).parent / "api" / "function" / (object_name + ".rst")
    underscore = "=" * len(object_name)
    with open(path, "w") as file:
        print(f"Creating {path}")
        print(
            f"""{object_name}
{underscore}

.. currentmodule:: {module_name}

.. autofunction:: {name}
""",
            file=file,
        )


def generate_type(module_name, name):
    object_name = module_name + "." + name
    path = Path(__file__).parent / "api" / "type" / (object_name + ".rst")
    underscore = "=" * len(object_name)
    with open(path, "w") as file:
        print(f"Creating {path}")
        print(
            f"""{object_name}
{underscore}

.. currentmodule:: {module_name}

.. autodata:: {name}
""",
            file=file,
        )


def generate_files_for_module(module_name: str, m):
    symbols = m.__dict__
    if "__all__" in symbols:
        for name in symbols:
            o = symbols[name]
            if inspect.isclass(o):
                generate_class(module_name, name)
            elif inspect.isfunction(o):
                generate_function(module_name, name)
            elif name in autodoc_type_aliases:
                generate_type(module_name, name)


def generate_files():
    generate_toplevel()
    generate_hidden_toctrees()
    for module_name in all_modules:
        m = importlib.import_module(module_name)
        generate_files_for_module(module_name, m)


def setup(app):
    generate_files()


if __name__ == "__main__":
    generate_files()
