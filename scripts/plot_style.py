"""Shared Plotly palette on a light chart surface."""
import numpy as np

# Stakes classes are ordered, so they take the blue ordinal ramp (lightest step still >= 2:1
# on the surface), never cycled hues.
STAKES_RAMP = ('#86b6ef', '#6da7ec', '#5598e7', '#3987e5', '#2a78d6',
               '#256abf', '#1c5cab', '#184f95', '#104281', '#0d366b')
SEQUENTIAL_SCALE = [[0.0, '#cde2fb'], [0.5, '#2a78d6'], [1.0, '#0d366b']]
SURFACE_COLOR = '#fcfcfb'
INK = '#0b0b0b'
INK_SECONDARY = '#52514e'
MUTED = '#898781'
GRID = '#e1e0d9'
BASELINE = '#c3c2b7'


def stakes_colors(order):
    """Ordered classes spread evenly over the ordinal ramp, lightest for the lowest stakes."""
    if len(order) > len(STAKES_RAMP):
        raise ValueError(f'At most {len(STAKES_RAMP)} stakes classes can be colored.')
    steps = np.linspace(0, len(STAKES_RAMP) - 1, len(order)).round().astype(int)
    return {name: STAKES_RAMP[step] for name, step in zip(order, steps)}
