# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

# -- Project information -----------------------------------------------------

project = "cards"
copyright = "2025, M. Bouton, S. Despierres, P.-A. Thouvenin and P. Chainais"
author = "M. Bouton, S. Despierres, P.-A. Thouvenin and P. Chainais"
version = release = "0.1.0"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.coverage",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.inheritance_diagram",
    "sphinxcontrib.bibtex",
    "sphinxcontrib.apa",
    "sphinx_copybutton",
    "sphinx_design",
]

templates_path = ["_templates"]
exclude_patterns = []

# -- Autodoc & autosummary ---------------------------------------------------

autodoc_typehints = "description"
autosummary_generate = True
autosummary_generate_overwrite = False

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "private-members": True,
    "show-inheritance": True,
}

autodoc_mock_imports = [
    "cupy",
    "h5py",
    "imageio",
    "matplotlib",
    "mpi4py",
    "numba",
    "numpy",
    "scipy",
    "skimage",
    "torch",
    "torchvision",
    "tqdm",
    "ucx",
]

# -- Napoleon (docstring style) ----------------------------------------------

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_attr_annotations = True

# -- sphinxcontrib-bibtex ----------------------------------------------------

bibtex_bibfiles = ["strings_all_ref.bib", "biblio.bib"]
bibtex_encoding = "utf-8-sig"
bibtex_default_style = "alpha"
bibtex_reference_style = "author_year"

# -- HTML output -------------------------------------------------------------

html_theme = "furo"
html_static_path = []
