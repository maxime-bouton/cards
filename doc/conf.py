# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html


# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.insert(0, os.path.abspath("../../src"))


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "cards"
copyright = "2025, M. Bouton, S. Despierres, P.-A. Thouvenin and P. Chainais"
author = "M. Bouton, S. Despierres, P.-A. Thouvenin and P. Chainais"

# The full version, including alpha/beta/rc tags
version = "0.1.0"
release = "0.1.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",  # Core library for html generation from docstrings
    # "sphinx.ext.autosummary",  # Create neat summary tables
    "autoapi.extension",
    "sphinx.ext.coverage",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    # "sphinx_rtd_theme",
    "sphinxcontrib.bibtex",
    "sphinxcontrib.apa",
    "sphinx.ext.mathjax",  # Render math via JavaScript
    "sphinx.ext.inheritance_diagram",  # Include inheritance diagrams
    "sphinx_copybutton",  # Add copy button to code sections from the doc
    "sphinx_design",  # Add tabs, https://sphinx-design.readthedocs.io/en/latest/tabs.html
]
# autosummary_generate = True  # Turn on sphinx.ext.autosummary

autoapi_dirs = ["../src"]
# autoapi_root = "../autoapi"
autoapi_options = [
    "members",
    "undoc-members",
    "private-members",
    "show-inheritance",
    "show-module-summary",
    "imported-members",
]
autodoc_typehints = "description"
autoapi_keep_files = True
autoapi_generate_api_docs = True
# autoapi_add_toctree_entry = False
# autoapi_ignore = ["path1", "path2"]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Extension configuration -------------------------------------------------

# sphinxcontrib-bibtex https://sphinxcontrib-bibtex.readthedocs.io/en/latest/usage.html
# sphinxcontrib-apa https://pypi.org/project/sphinxcontrib-apa/
bibtex_bibfiles = ["strings_all_ref.bib", "biblio.bib"]
bibtex_encoding = "utf-8-sig"
bibtex_default_style = "alpha"
bibtex_reference_style = "author_year"
# bibtex_reference_style = "alpha"  # alpha, plain , unsrt, and unsrtalpha

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = "furo"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = []  # ['_static']

# html_extra_path = ["../coverage_html_report"]
