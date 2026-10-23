"""The Frame every sketch receives, plus small helpers for looking at it."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# Kinect v2 depth/IR resolution.
DEPTH_HEIGHT, DEPTH_WIDTH = 424, 512


@dataclass
class Frame:
    """One synchronized capture from the sensor.

    depth: (424, 512) float32, millimeters. 0 means "no reading".
    ir:    (424, 512) float32 intensity, or None if the sketch didn't ask for it.
    color: (1080, 1920, 3) uint8 BGR, or None if the sketch didn't ask for it.
    """

    index: int
    timestamp: float
    depth: np.ndarray
    ir: np.ndarray | None = None
    color: np.ndarray | None = None


def depth_to_bgr(depth: np.ndarray, near: float = 500.0, far: float = 4500.0) -> np.ndarray:
    """Turn a depth map into a viewable grayscale BGR image (near = bright, no reading = black)."""
    valid = depth > 0
    scaled = np.zeros(depth.shape, dtype=np.uint8)
    clipped = np.clip(depth[valid], near, far)
    scaled[valid] = (255 * (far - clipped) / (far - near)).astype(np.uint8)
    return cv2.cvtColor(scaled, cv2.COLOR_GRAY2BGR)
