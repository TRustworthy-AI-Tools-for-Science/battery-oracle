"""Plotting + logging helpers vendored from ``battery_forecast``.

Contains the Slipstream colormap (``visualize/colormap.py``) and the ``label_axes``
subplot-lettering helper (``visualize/plots.py``). Kept dependency-light:
matplotlib + numpy.
"""
from __future__ import annotations

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

