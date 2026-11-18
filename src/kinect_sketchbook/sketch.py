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
        keep_aspect: when True (default), the window letterboxes to the image's
                 aspect ratio. When False, the image is stretched to fill the window.
        fill_display: when True, open the window maximized/fullscreen so it covers
                 the display; useful with keep_aspect=False.
        window_size: optional (width, height) for the initial window. None leaves
                 the backend default.
    """

    title: str = "sketchbook"
    streams: tuple[str, ...] = ("depth",)
    keep_aspect: bool = True
    fill_display: bool = False
    window_size: tuple[int, int] | None = None

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
