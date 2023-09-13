# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'AdversarialTensors'
copyright = 'Los Alamos National Laboratory'
author = 'Manish Bhattarai, Mehmet Kaymak, Ben Nebgen, Kim Rasmussen, Boian Alexandrov'
release = '1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autodoc",
    "sphinx_automodapi.automodapi",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinxcontrib.bibtex",
]

templates_path = ['_templates']
exclude_patterns = ['.ipynb_checkpoints', "version.py"]

language = 'Python'
autoclass_content = 'both'
bibtex_bibfiles = ['refs.bib']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_book_theme'

#html_theme = 'alabaster'
html_static_path = ['_static']
