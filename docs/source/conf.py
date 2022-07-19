import os
import sys
import sphinx_rtd_theme
from recommonmark.parser import CommonMarkParser
from recommonmark.transform import AutoStructify


sys.path.insert(0, os.path.abspath('../../sgl/'))

# Configuration file for the Sphinx documentation builder.

# -- Project information

project = 'SGL'
copyright = '2022, DAIR @PKU'
author = 'DAIR @PKU'

release = 'beta'
version = '0.1.2'

# -- General configuration
extensions = ['recommonmark',
              'sphinx.ext.autodoc',
              'sphinx.ext.doctest',
              'sphinx.ext.intersphinx',
              'sphinx.ext.todo',
              'sphinx.ext.coverage',
              'sphinx.ext.mathjax',
              'sphinx.ext.napoleon',
              'sphinx.ext.autosummary',
              'sphinx.ext.viewcode',
              'sphinx.ext.githubpages',]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}
intersphinx_disabled_domains = ['std']

templates_path = ['_templates']

# -- Options for HTML output

language = 'en'
master_doc = 'index'
html_theme = "sphinx_rtd_theme"
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

source_parsers = {
    '.md': CommonMarkParser,
}
source_suffix = ['.rst', '.md']
github_doc_root = 'https://github.com/rtfd/recommonmark/tree/master/docs/'

# -- Options for EPUB output
epub_show_urls = 'footnote'


# app setup hook
def setup(app):
    app.add_config_value('recommonmark_config', {
        'url_resolver': lambda url: github_doc_root + url,
        'auto_toc_tree_section': 'Contents',
        'enable_math': False,
        'enable_inline_math': False,
        'enable_eval_rst': True
    }, True)
    app.add_transform(AutoStructify)
