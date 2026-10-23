"""The Sketch base class. Subclass it in a single file under sketches/."""

from __future__ import annotations

import numpy as np

from .frame import Frame, depth_to_bgr


class Sketch:
    """Override any of these. The runner owns the device and the loop.

    Class attributes:
        title:   window title.
        streams: which sensor streams to capture: any of "depth", "ir", "color".
                 Ask only for what you use; color in particular costs CPU to decode.
    """

    title: str = "depthsketch"
    streams: tuple[str, ...] = ("depth",)

    def setup(self) -> None:
        """Called once before the first frame."""

    def update(self, frame: Frame) -> None:
        """Called for every frame. Do tracking and logic here."""

    def draw(self, frame: Frame) -> np.ndarray | None:
        """Return a BGR image to show, or None to show nothing. Default: the depth view."""
        return depth_to_bgr(frame.depth)

    def on_key(self, key: str) -> None:
        """Called with a single character when a key is pressed in the window."""

    def teardown(self) -> None:
        """Called once on exit, even after an error or Ctrl-C."""
