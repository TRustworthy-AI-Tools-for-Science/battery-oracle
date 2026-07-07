"""Plotting + logging helpers vendored from ``battery_forecast``.

Contains the Slipstream colormap (``visualize/colormap.py``), the ``label_axes``
subplot-lettering helper (``visualize/plots.py``), and ``get_logger``
(``utils.py``). Kept dependency-light: matplotlib + numpy, with an optional
``rich`` logging handler that degrades to the stdlib handler when rich is absent.
"""
from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# --- Slipstream palette (hex, ordered light → dark) -------------------------
SLIPSTREAM_COLORS = [
    "#4E67C8",  # 0
    "#5ECCF3",  # 1
    "#5DCEAF",  # 2
    "#A7EA52",  # 3
    "#FF8021",  # 4
    "#F14124",  # 5
    "#903423",  # 6
    "#000000",  # 7
]

# Build continuous colormap from the palette
slipstream = LinearSegmentedColormap.from_list("Slipstream", SLIPSTREAM_COLORS)
slipstream_r = slipstream.reversed()


def register() -> None:
    """Register Slipstream colormaps with matplotlib (idempotent)."""
    for cmap in (slipstream, slipstream_r):
        if cmap.name not in plt.colormaps():
            plt.colormaps.register(cmap)


# Register on import
register()


def label_axes(axes, start=0):
    """Stamp bold subplot letters (a, b, c …) outside the y-axis of each axes.

    Hidden spacer subplots (axis turned off) are skipped automatically.
    """
    flat = np.asarray(axes).ravel()
    idx = start
    for ax in flat:
        if not ax.get_visible() or not getattr(ax, "axison", True):
            continue
        ax.text(
            -0.1, 0.97, chr(ord("a") + idx),
            transform=ax.transAxes,
            fontsize=9, fontweight="bold",
            va="top", ha="right",
        )
        idx += 1


def get_logger(name: str, level=logging.WARNING) -> logging.Logger:
    """Get a logger with the given name (rich handler if available)."""
    logger = logging.getLogger(name)
    if logger.hasHandlers():
        # If logger has handlers, return to avoid duplicates, just set level
        logger.setLevel(level)
        return logger

    logger.setLevel(level)
    try:
        from rich.console import Console
        from rich.logging import RichHandler

        # stderr=True keeps logging off fd 1, which autoeis' inference redirects.
        console = Console(force_jupyter=False, width=100, stderr=True)
        handler: logging.Handler = RichHandler(
            rich_tracebacks=True, console=console, show_path=False
        )
        handler.setFormatter(logging.Formatter("%(message)s", datefmt="[%X]"))
    except Exception:  # rich not installed — fall back to a plain stderr handler
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(name)s: %(message)s"))
    logger.addHandler(handler)
    return logger
