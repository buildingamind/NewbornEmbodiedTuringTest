'''Configuration file for the Sphinx documentation builder.'''

# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

sys.path.insert(0, os.path.abspath('../../src/'))

import os
import sys
from nett import __version__

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'NETT'
copyright = '2024, Zachary Laborde'
author = 'Zachary Laborde'
release = __version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
  'sphinx.ext.autodoc',
  'sphinx.ext.napoleon',
  'myst_parser',
]

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "body_max_width": "none",
}

html_static_path = ['_static']

html_baseurl = "/html/"

# Example conf.py snippet
html_css_files = [
    'custom_styles.css',  # The name of your custom CSS file
]
