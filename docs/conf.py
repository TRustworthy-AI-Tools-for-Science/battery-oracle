"""Sphinx configuration for battery-oracle."""
from __future__ import annotations

import importlib
import importlib.metadata
import inspect
import os
import sys

# Help ad-hoc `sphinx-build` runs where the package is not installed into the env.
sys.path.insert(0, os.path.abspath("../src"))

# -- Project information -----------------------------------------------------
project = "battery-oracle"
author = "Ashley Dale"
copyright = "2026, Ashley Dale"

try:
    release = importlib.metadata.version("battery-oracle")
except importlib.metadata.PackageNotFoundError:
    release = "0.1.0"
version = ".".join(release.split(".")[:2])

_REPO = "https://github.com/TRustworthy-AI-Tools-for-Science/battery-oracle"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.linkcode",
    "sphinx.ext.mathjax",
    "myst_nb",              # markdown + notebook rendering (brings myst-parser)
    "sphinx_copybutton",
    "sphinxcontrib.bibtex",
]

# -- bibliography (sphinxcontrib-bibtex) -------------------------------------
bibtex_bibfiles = ["references.bib"]
bibtex_reference_style = "author_year"

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]

# -- HTML output (furo) ------------------------------------------------------
html_theme = "furo"
html_title = f"battery-oracle {release}"
html_static_path = ["_static"]
html_logo = "_static/logo.png"
html_favicon = "_static/logo.png"
html_theme_options = {
    "source_repository": _REPO + "/",
    "source_branch": "main",
    "source_directory": "docs/",
    "footer_icons": [
        {
            "name": "GitHub",
            "url": _REPO,
            "html": (
                '<svg stroke="currentColor" fill="currentColor" stroke-width="0" '
                'viewBox="0 0 16 16"><path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 '
                '8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37'
                '-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01'
                '1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89'
                '-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64'
                '-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92'
                '.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 '
                "1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0016 8c0-4.42-3.58-8-8-8z"
                '"></path></svg>'
            ),
            "class": "",
        },
    ],
}

# -- autodoc / autosummary ---------------------------------------------------
autosummary_generate = True
autodoc_typehints = "description"
autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}
# Optional deps are lazy-imported in the source, so importing battery_oracle never
# triggers them; mock them anyway to keep the docs env lean (no JAX / git deps).
autodoc_mock_imports = [
    "autoeis",
    "mpire",
    "optuna",
    "hybrid_drt",
    "hybdrt",
    "mittag_leffler",
    "tqdm",
]

# -- napoleon (NumPy-style docstrings) ---------------------------------------
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_use_param = True
napoleon_use_rtype = False
napoleon_preprocess_types = True

# -- intersphinx -------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "pybamm": ("https://docs.pybamm.org/en/latest/", None),
}

# -- myst-nb / myst-parser ---------------------------------------------------
nb_execution_mode = "off"          # NEVER execute notebooks on the docs builder
nb_execution_timeout = 300
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "linkify",
    "substitution",
    "tasklist",
]
myst_heading_anchors = 3
suppress_warnings = ["mystnb.unknown_mime_type"]


# -- linkcode: [source] links to the GitHub source line ----------------------
def linkcode_resolve(domain, info):
    """Map a documented Python object to its GitHub source URL (blob + line range).

    Robust to editable vs site-packages installs: the path is computed relative to
    the installed ``battery_oracle`` package directory and rebuilt as
    ``src/battery_oracle/<rel>`` under the repo.
    """
    if domain != "py" or not info.get("module"):
        return None
    modname, fullname = info["module"], info["fullname"]
    try:
        obj = importlib.import_module(modname)
        for part in fullname.split("."):
            obj = getattr(obj, part)
        obj = inspect.unwrap(obj)
        fn = inspect.getsourcefile(obj)
        source, lineno = inspect.getsourcelines(obj)
    except Exception:
        return None
    if not fn:
        return None
    try:
        import battery_oracle
        pkg_root = os.path.dirname(battery_oracle.__file__)
        rel = os.path.relpath(fn, pkg_root)
    except Exception:
        return None
    if rel.startswith(".."):
        return None
    end = lineno + len(source) - 1
    return f"{_REPO}/blob/main/src/battery_oracle/{rel}#L{lineno}-L{end}"
