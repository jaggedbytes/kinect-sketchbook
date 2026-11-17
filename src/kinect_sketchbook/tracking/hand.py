"""Hands found from depth alone, with no model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Hand:
    """mask:     (424, 512) bool, the depth pixels that make up the hand.
    centroid: (u, v) depth pixel at the hand's center.
    """

    mask: np.ndarray
    centroid: tuple[float, float]


def nearest_hand(depth: np.ndarray, slab_mm: float = 100, min_pixels: int = 300) -> Hand | None:
    """Treat whatever is closest to the camera as the hand.

    The hand is every reading within slab_mm of the nearest one, so anything
    closer than your hand (a desk edge, a shoulder) is picked instead.
    Returns None when nothing is in view or the blob has min_pixels or fewer (noise).
    """
    valid = depth[depth > 0]
    if not valid.size:
        return None
    mask = (depth > 0) & (depth < valid.min() + slab_mm)
    ys, xs = np.nonzero(mask)
    if xs.size <= min_pixels:
        return None
    return Hand(mask, (float(xs.mean()), float(ys.mean())))
