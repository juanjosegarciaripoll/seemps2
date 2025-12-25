# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config

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
    "sphinx_design",
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
    "Operator": "Operator",
    "Vector": "Vector",
    "VectorLike": "VectorLike",
    "python:list": "list",
    "Weight": "Weight",
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
