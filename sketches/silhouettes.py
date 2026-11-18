"""Silhouettes: each person opts in with an open palm, then only their cutout is shown.

Several people can stand in front of the Kinect. Each one who holds an open palm
gets a solid-color silhouette on a black frame; everyone else stays hidden. Open
palms are detected on a depth-scaled crop around the hand so farther people still
register. Colors are unique per person from a fixed palette (reuse only when there
are more people than colors). When a consented person leaves the frame, their
cutout disappears and they must opt in again.

Run:  sketchbook run sketches/silhouettes.py
Keys: q / Esc to quit.

Needs models/gesture_recognizer.task and models/pose_landmarker_full.task (see README).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from kinect_sketchbook import Frame, Sketch
from kinect_sketchbook.tracking import (
    GestureTracker, Joint, PoseTracker, Skeleton, hand_roi,
)

CORE = (
    Joint.NOSE,
    Joint.LEFT_SHOULDER,
    Joint.RIGHT_SHOULDER,
    Joint.LEFT_HIP,
    Joint.RIGHT_HIP,
)
# Wrist plus a fingertip so the crop centers on the palm, not the forearm.
HAND_ANCHORS = (
    (Joint.LEFT_WRIST, Joint.LEFT_INDEX),
    (Joint.RIGHT_WRIST, Joint.RIGHT_INDEX),
)
OPEN_PALM = "Open_Palm"
PROMPT = "Show open palm to consent to capture"

# Unique fills for consented people (BGR for OpenCV). Hex: #FF1493, #7FFF00,
# #FF6A00, #0A84FF, #FFE600, #9B30FF.
PALETTE = (
    (0x93, 0x14, 0xFF),
    (0x00, 0xFF, 0x7F),
    (0x00, 0x6A, 0xFF),
    (0xFF, 0x84, 0x0A),
    (0x00, 0xE6, 0xFF),
    (0xFF, 0x30, 0x9B),
)


@dataclass
class Person:
    """One tracked figure. Consent is per person and drops when they leave."""

    id: int
    torso_xy: np.ndarray
    color: tuple[int, int, int]
    consented: bool = False
    absent_since: float | None = None
    palm_streak: int = 0
    body: Skeleton | None = field(default=None, repr=False)


class Silhouettes(Sketch):
    """Black frame with solid-color cutouts of people who have shown an open palm."""

    title = "silhouettes"
    streams = ("depth", "color")

    MAX_PEOPLE = 6
    ABSENT_S = 0.5
    EDGE_MARGIN = 80
    MIN_CORE = 3
    PALM_FRAMES = 2
    MATCH_PX = 280          # torso may move this far between frames and stay the same person
    DEPTH_SLAB_MM = 400     # keep color_depth within this of the torso depth
    JOINT_PAD_PX = 90       # expand the skeleton bbox so arms/head are inside the cutout
    MIN_MASK_PX = 800       # ignore tiny depth blobs (noise)
    HAND_M = 0.45           # meters of space the open-palm crop should cover

    def setup(self):
        self.pose = PoseTracker(variant="full", num_poses=self.MAX_PEOPLE)
        # ROI path uses IMAGE mode; num_hands here is only for a full-frame fallback.
        self.gestures = GestureTracker(gestures=(OPEN_PALM,), num_hands=2, min_score=0.5)
        self.people: list[Person] = []
        self._next_id = 1

    def update(self, frame: Frame):
        bodies = [b for b in self.pose.update(frame) if self._in_frame(b, frame)]
        self._match_people(bodies, frame.timestamp)

        # Depth-scaled wrist crops keep far hands large enough for Open_Palm.
        # Skip people who already consented — no need to re-run the model on them.
        pending: list[Person] = []
        rois: list[tuple[int, int, int, int]] = []
        for person in self.people:
            if person.consented or person.body is None:
                continue
            roi = self._hand_roi(person.body, frame)
            if roi is None:
                person.palm_streak = 0
                continue
            pending.append(person)
            rois.append(roi)

        if not rois:
            return

        for person, hands in zip(pending, self.gestures.recognize_rois(frame, rois)):
            if any(h.name == OPEN_PALM for h in hands):
                person.palm_streak += 1
                if person.palm_streak >= self.PALM_FRAMES:
                    person.consented = True
            else:
                person.palm_streak = 0

    def draw(self, frame: Frame):
        shape = frame.color.shape if frame.color is not None else (1080, 1920, 3)
        image = np.zeros(shape, dtype=np.uint8)
        if frame.color is None or frame.color_depth is None:
            self._draw_prompt(image)
            return image

        for person in self.people:
            if not person.consented or person.body is None:
                continue
            mask = self._silhouette(person.body, frame.color_depth)
            if mask is None:
                continue
            image[mask] = person.color

        if not any(p.consented for p in self.people):
            self._draw_prompt(image)
        return image

    def teardown(self):
        self.pose.close()
        self.gestures.close()

    def _match_people(self, bodies: list[Skeleton], now: float) -> None:
        """Assign this frame's bodies to existing people by nearest torso; drop absentees."""
        torsos = [self._torso_xy(b) for b in bodies]
        assigned: set[int] = set()

        for person in self.people:
            person.body = None
            best_i, best_d = -1, self.MATCH_PX
            for i, torso in enumerate(torsos):
                if i in assigned:
                    continue
                d = float(np.linalg.norm(torso - person.torso_xy))
                if d < best_d:
                    best_i, best_d = i, d
            if best_i < 0:
                if person.absent_since is None:
                    person.absent_since = now
                continue
            assigned.add(best_i)
            person.body = bodies[best_i]
            person.torso_xy = torsos[best_i]
            person.absent_since = None

        for i, body in enumerate(bodies):
            if i in assigned:
                continue
            self.people.append(Person(
                id=self._next_id,
                torso_xy=torsos[i],
                color=self._assign_color(),
                body=body,
            ))
            self._next_id += 1

        kept: list[Person] = []
        for person in self.people:
            if person.body is not None:
                kept.append(person)
                continue
            assert person.absent_since is not None
            if now - person.absent_since < self.ABSENT_S:
                kept.append(person)  # brief dropout: keep the latch
            # else: left the frame — drop track and consent
        self.people = kept

    def _assign_color(self) -> tuple[int, int, int]:
        """Next unused palette color; wrap when every color is already taken."""
        used = {p.color for p in self.people}
        for color in PALETTE:
            if color not in used:
                return color
        return PALETTE[(self._next_id - 1) % len(PALETTE)]

    def _hand_roi(self, body: Skeleton, frame: Frame) -> tuple[int, int, int, int] | None:
        """Depth-scaled crop around the clearest hand (palm-centered when possible)."""
        if frame.color is None:
            return None
        height, width = frame.color.shape[:2]
        fx = frame.color_intrinsics.fx if frame.color_intrinsics is not None else 1081.0

        best = None  # (visibility, center, depth_m)
        for wrist_j, tip_j in HAND_ANCHORS:
            if not body.visible(wrist_j):
                continue
            z = body.xyz[wrist_j, 2]
            if not np.isfinite(z):
                # Fall back to torso depth so a missing wrist reading still gets a crop.
                core_z = body.xyz[[j for j in CORE if body.visible(j)], 2]
                core_z = core_z[np.isfinite(core_z)]
                if core_z.size == 0:
                    continue
                z = float(np.median(core_z))
            if body.visible(tip_j):
                center = 0.5 * (body.image_xy[wrist_j] + body.image_xy[tip_j])
            else:
                center = body.image_xy[wrist_j]
            visibility = float(body.visibility[wrist_j])
            if best is None or visibility > best[0]:
                best = (visibility, center, float(z))
        if best is None:
            return None
        return hand_roi(best[1], best[2], (height, width), fx=fx, hand_m=self.HAND_M)

    def _silhouette(self, body: Skeleton, color_depth: np.ndarray) -> np.ndarray | None:
        """Bool mask of this person in the color image, from depth near the torso."""
        core = [j for j in CORE if body.visible(j) and np.isfinite(body.xyz[j, 2])]
        if len(core) < self.MIN_CORE:
            return None
        z_mm = float(np.median(body.xyz[core, 2])) * 1000.0
        if z_mm < 500:
            return None

        visible = [j for j in range(len(body.image_xy)) if body.visible(j)]
        if len(visible) < self.MIN_CORE:
            return None
        xs = body.image_xy[visible, 0]
        ys = body.image_xy[visible, 1]
        h, w = color_depth.shape
        x0 = max(0, int(xs.min()) - self.JOINT_PAD_PX)
        x1 = min(w, int(xs.max()) + self.JOINT_PAD_PX + 1)
        y0 = max(0, int(ys.min()) - self.JOINT_PAD_PX)
        y1 = min(h, int(ys.max()) + self.JOINT_PAD_PX + 1)
        if x1 <= x0 or y1 <= y0:
            return None

        region = color_depth[y0:y1, x0:x1]
        near = (region > 0) & (np.abs(region - z_mm) <= self.DEPTH_SLAB_MM)
        if int(near.sum()) < self.MIN_MASK_PX:
            return None

        # Fill small holes so clothing folds don't punch through the cutout.
        kernel = np.ones((7, 7), np.uint8)
        near_u8 = cv2.morphologyEx(near.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        mask = np.zeros((h, w), dtype=bool)
        mask[y0:y1, x0:x1] = near_u8.astype(bool)
        return mask

    def _in_frame(self, body: Skeleton, frame: Frame) -> bool:
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
        depths = body.xyz[core, 2]
        finite = np.isfinite(depths)
        return bool(finite.any() and np.any(depths[finite] > 0.5))

    @staticmethod
    def _torso_xy(body: Skeleton) -> np.ndarray:
        core = [j for j in CORE if body.visible(j)]
        return body.image_xy[core].mean(axis=0).astype(np.float32)

    @staticmethod
    def _draw_prompt(image: np.ndarray) -> None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale, thickness = 0.85, 2
        (tw, th), _ = cv2.getTextSize(PROMPT, font, scale, thickness)
        x = image.shape[1] - tw - 36
        y = image.shape[0] - 36
        cv2.putText(image, PROMPT, (x, y), font, scale, (210, 210, 210),
                    thickness, cv2.LINE_AA)
