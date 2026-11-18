"""Things inferred from Frames: bodies, hands, gestures. Sketches opt in to each one."""

from .gesture import GestureTracker
from .gestures import SwipeDetector, WaveDetector
from .hand import Hand, nearest_hand
from .pose import CONNECTIONS, Joint, PoseTracker, Skeleton, draw_skeleton

__all__ = [
    "CONNECTIONS", "GestureTracker", "Hand", "Joint", "PoseTracker", "Skeleton",
    "SwipeDetector", "WaveDetector", "draw_skeleton", "nearest_hand",
]
