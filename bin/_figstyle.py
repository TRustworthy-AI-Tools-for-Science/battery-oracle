"""Colours and subplot letters of the documentation figures (bin/generate_*.py)."""
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

#: Slipstream palette, light to dark
SLIPSTREAM_COLORS = ["#4E67C8", "#5ECCF3", "#5DCEAF", "#A7EA52", "#FF8021", "#F14124", "#903423", "#000000"]
slipstream = LinearSegmentedColormap.from_list("Slipstream", SLIPSTREAM_COLORS)


def label_axes(axes, start=0):
    """Bold subplot letters (a, b, c ...) outside the y-axis of each visible axes."""
    idx = start
    for ax in np.asarray(axes).ravel():
        if not ax.get_visible() or not getattr(ax, "axison", True):
            continue
        ax.text(-0.1, 0.97, chr(ord("a") + idx), transform=ax.transAxes, fontsize=9, fontweight="bold",
                va="top", ha="right")
        idx += 1
