"""Consent: the picture stays black until you show an open palm, then goes black when you leave.

Hold an open palm to the camera (MediaPipe's Open_Palm). Color stays on while someone
is clearly in view, including through brief pose dropouts. Once the torso has been out
of frame for a moment, it goes dark and waits for another open palm.

Run:  sketchbook run sketches/consent.py
Keys: q / Esc to quit.

Needs models/gesture_recognizer.task and models/pose_landmarker_full.task (see README).
"""

import cv2
import numpy as np

from kinect_sketchbook import Frame, Sketch
from kinect_sketchbook.tracking import GestureTracker, Joint, PoseTracker

# Enough of these must be visible, and their center inside the frame, to count as present.
CORE = (
    Joint.NOSE,
    Joint.LEFT_SHOULDER,
    Joint.RIGHT_SHOULDER,
    Joint.LEFT_HIP,
    Joint.RIGHT_HIP,
)

PROMPT = "Show open palm to consent to capture"
OPEN_PALM = "Open_Palm"


class Consent(Sketch):
    """Display color only after an open palm, until the person leaves."""

    title = "consent"
    streams = ("depth", "color")

    ABSENT_S = 0.5       # out of frame this long before color turns off
    EDGE_MARGIN = 80     # torso center must stay this far inside the image
    MIN_CORE = 3         # how many CORE joints must be visible
    PALM_FRAMES = 2      # Open_Palm must hold this many frames in a row

    def setup(self):
        self.pose = PoseTracker(variant="full")
        self.gestures = GestureTracker(gestures=(OPEN_PALM,), min_score=0.5)
        self.consented = False
        self.absent_since: float | None = None
        self._palm_streak = 0

    def update(self, frame: Frame):
        bodies = [b for b in self.pose.update(frame) if self._in_frame(b, frame)]
        open_palm = OPEN_PALM in self.gestures.update(frame)

        if not bodies:
            self._note_absence(frame.timestamp)
            self._palm_streak = 0
            return

        # In-frame torso holds the latch. Edge ghosts and brief dropouts do not.
        self.absent_since = None
        if open_palm:
            self._palm_streak += 1
            if self._palm_streak >= self.PALM_FRAMES:
                self.consented = True
        else:
            self._palm_streak = 0

    def draw(self, frame: Frame):
        # frame.color is already BGR, which is what the window shows.
        if self.consented and frame.color is not None:
            return frame.color
        shape = frame.color.shape if frame.color is not None else (1080, 1920, 3)
        image = np.zeros(shape, dtype=np.uint8)
        self._draw_prompt(image)
        return image

    def teardown(self):
        self.pose.close()
        self.gestures.close()

    def _note_absence(self, now: float) -> None:
        if self.absent_since is None:
            self.absent_since = now
            return
        if self.consented and now - self.absent_since >= self.ABSENT_S:
            self.consented = False
            self._palm_streak = 0

    def _in_frame(self, body, frame: Frame) -> bool:
        """True when the torso is clearly inside the color image, not an edge ghost."""
        height, width = (frame.color.shape[:2] if frame.color is not None
                         else (1080, 1920))
        core = [j for j in CORE if body.visible(j)]
        if len(core) < self.MIN_CORE:
            return False
        xs = body.image_xy[core, 0]
        ys = body.image_xy[core, 1]
        cx, cy = float(xs.mean()), float(ys.mean())
        m = self.EDGE_MARGIN
        if not (m <= cx < width - m and m <= cy < height - m):
            return False
        # Prefer people the depth camera can still see; wall-edge ghosts often
        # have no reading under the joints the model invents.
        depths = body.xyz[core, 2]
        finite = np.isfinite(depths)
        return bool(finite.any() and np.any(depths[finite] > 0.5))

    @staticmethod
    def _draw_prompt(image: np.ndarray) -> None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale, thickness = 0.85, 2
        (tw, th), _ = cv2.getTextSize(PROMPT, font, scale, thickness)
        x = image.shape[1] - tw - 36
        y = image.shape[0] - 36
        cv2.putText(image, PROMPT, (x, y), font, scale, (210, 210, 210),
                    thickness, cv2.LINE_AA)
