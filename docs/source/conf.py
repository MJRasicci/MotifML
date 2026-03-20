#!/usr/bin/env python3

"""Sphinx configuration for the MotifML documentation set."""

# ruff: noqa: E402

import re
import sys
from pathlib import Path

DOCS_SOURCE = Path(__file__).resolve().parent
PROJECT_ROOT = DOCS_SOURCE.parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from motifml import __version__ as release

project = "MotifML"
author = "MotifML maintainers"
version = re.match(r"^([0-9]+\.[0-9]+).*", release).group(1)

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "sphinx.ext.doctest",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.ifconfig",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "nbsphinx",
    "myst_parser",
    "sphinx_copybutton",
]

autosummary_generate = True
templates_path = [path.name for path in (DOCS_SOURCE / "_templates",) if path.exists()]
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
master_doc = "index"
language = "en"
exclude_patterns = ["_build", "**.ipynb_checkpoints"]
pygments_style = "sphinx"

html_theme = "sphinx_rtd_theme"
html_theme_options = {"collapse_navigation": False, "style_external_links": True}
html_static_path = [path.name for path in (DOCS_SOURCE / "_static",) if path.exists()]
html_show_sourcelink = False
htmlhelp_basename = "motifmldoc"

latex_elements: dict[str, str] = {}
latex_documents = [
    (
        master_doc,
        "MotifML.tex",
        "MotifML Documentation",
        author,
        "manual",
    )
]

man_pages = [
    (
        master_doc,
        "motifml",
        "MotifML Documentation",
        [author],
        1,
    )
]

texinfo_documents = [
    (
        master_doc,
        "motifml",
        "MotifML Documentation",
        author,
        "motifml",
        "Project MotifML codebase.",
        "Data-Science",
    )
]

todo_include_todos = False
nbsphinx_kernel_name = "python3"


def remove_arrows_in_examples(lines):
    for i, line in enumerate(lines):
        lines[i] = line.replace(">>>", "")


def autodoc_process_docstring(app, what, name, obj, options, lines):  # noqa: PLR0913
    del app, what, name, obj, options
    remove_arrows_in_examples(lines)


def skip(app, what, name, obj, should_skip, options):  # noqa: PLR0913
    del app, what, obj, options
    if name == "__init__":
        return False
    return should_skip


def setup(app):
    app.connect("autodoc-process-docstring", autodoc_process_docstring)
    app.connect("autodoc-skip-member", skip)
