"""Frame sources. A source is a context manager that yields Frames when iterated."""

from .kinect import KinectSource

__all__ = ["KinectSource"]
