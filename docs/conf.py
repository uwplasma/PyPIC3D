import os
import sys

sys.path.insert(0, os.path.abspath('..'))

project = 'PyPIC3D'
author = 'Christopher Woolford'
release = '0.1.3'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx_design',
]

templates_path = ['_templates']
exclude_patterns = []

html_theme = 'furo'
html_static_path = ['images']
html_logo = 'images/PyPICLogo.png'
html_css_files = ['custom.css']
