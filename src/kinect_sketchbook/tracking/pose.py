"""Body pose from a machine-learning model, placed in 3D with the Kinect's depth.

MediaPipe's Pose Landmarker guesses 33 joints from the color image. Each joint is
then given a measured position by looking up the Kinect depth under it
(Frame.color_depth). Both are kept on the Skeleton: what the model guessed, and
what the sensor measured.

Needs MediaPipe and a model file (see the README):
    pip install mediapipe "numpy<2"
    models/pose_landmarker_full.task

    from kinect_sketchbook.tracking import Joint, PoseTracker

    class Pose(Sketch):
        streams = ("depth", "color")
        def setup(self):
            self.pose = PoseTracker()
        def update(self, frame):
            for body in self.pose.update(frame):
                print(body.xyz[Joint.RIGHT_WRIST])
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ..frame import Frame

MODEL_DIR = Path("models")
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
             "pose_landmarker_{variant}/float16/latest/pose_landmarker_{variant}.task")
VARIANTS = ("lite", "full", "heavy")

# The model's input is 256 px square, so a full 1920x1080 frame only costs conversion time.
DETECT_WIDTH = 960


class Joint(enum.IntEnum):
    """MediaPipe's 33 pose landmarks. Left and right are the person's own."""

    NOSE = 0
    LEFT_EYE_INNER = 1
    LEFT_EYE = 2
    LEFT_EYE_OUTER = 3
    RIGHT_EYE_INNER = 4
    RIGHT_EYE = 5
    RIGHT_EYE_OUTER = 6
    LEFT_EAR = 7
    RIGHT_EAR = 8
    MOUTH_LEFT = 9
    MOUTH_RIGHT = 10
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_PINKY = 17
    RIGHT_PINKY = 18
    LEFT_INDEX = 19
    RIGHT_INDEX = 20
    LEFT_THUMB = 21
    RIGHT_THUMB = 22
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HEEL = 29
    RIGHT_HEEL = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32


# Bones, as pairs of joints (MediaPipe's PoseLandmarksConnections).
CONNECTIONS: list[tuple[int, int]] = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24), (23, 25), (24, 26), (25, 27), (26, 28),
    (27, 29), (28, 30), (29, 31), (30, 32), (27, 31), (28, 32),
]


@dataclass
class Skeleton:
    """One person. Every array has one row per Joint, so body.xyz[Joint.NOSE] works.

    image_xy:   (33, 2) pixels in the color image (x right, y down).
    visibility: (33,) the model's confidence, 0-1, that the joint is in view and not hidden.
    xyz:        (33, 3) meters in camera space (x right, y down, z forward), measured
                from Kinect depth. NaN where the sensor had no reading under the joint,
                or where the joint is hidden and the depth belongs to whatever is in front of it.
    model_xyz:  (33, 3) meters, the model's own 3D guess, centered between the hips.
                Not measured; it exists even with no depth reading.
    """

    image_xy: np.ndarray
    visibility: np.ndarray
    xyz: np.ndarray
    model_xyz: np.ndarray

    def visible(self, joint: int, threshold: float = 0.5) -> bool:
        return bool(self.visibility[joint] >= threshold)


def _require_mediapipe():
    try:
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision
    except ImportError as e:
        raise SystemExit(f'Pose tracking needs MediaPipe:  pip install mediapipe "numpy<2"\n  ({e})') from e
    return mp, BaseOptions, vision


def _model_path(variant: str, model_path: str | Path | None) -> Path:
    if model_path is None:
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}")
        model_path = MODEL_DIR / f"pose_landmarker_{variant}.task"
    path = Path(model_path)
    if not path.is_file():
        url = MODEL_URL.format(variant=variant)
        raise SystemExit(
            f"No pose model at {path}. Download it (from the repo root) with:\n"
            f"  mkdir -p {path.parent} && curl -L -o {path} {url}"
        )
    return path


class PoseTracker:
    """Runs the pose model on each frame's color image. Needs streams ("depth", "color").

    variant:     "lite" (fastest), "full", or "heavy" (most accurate, slowest).
    num_poses:   most people to track at once.
    depth_radius: pixels around each joint to search for a depth reading.

    With --mirror the image is flipped before the model sees it, so the model
    calls your right hand LEFT_*. Joint positions are still correct on screen.
    """

    def __init__(
        self,
        variant: str = "full",
        num_poses: int = 2,
        min_confidence: float = 0.5,
        depth_radius: int = 6,
        model_path: str | Path | None = None,
    ) -> None:
        self._mp, BaseOptions, vision = _require_mediapipe()
        options = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(_model_path(variant, model_path))),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=num_poses,
            min_pose_detection_confidence=min_confidence,
            min_pose_presence_confidence=min_confidence,
            min_tracking_confidence=min_confidence,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(options)
        self._last_ms = -1
        self.depth_radius = depth_radius
        self.skeletons: list[Skeleton] = []

    def update(self, frame: Frame) -> list[Skeleton]:
        """Find the people in this frame. Also kept in self.skeletons."""
        if frame.color is None:
            raise ValueError('PoseTracker needs the color stream: set streams = ("depth", "color")')

        height, width = frame.color.shape[:2]
        small = cv2.resize(frame.color, (DETECT_WIDTH, DETECT_WIDTH * height // width),
                           interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        # Video mode tracks between frames and needs strictly increasing timestamps.
        ms = max(int(frame.timestamp * 1000), self._last_ms + 1)
        self._last_ms = ms
        result = self._landmarker.detect_for_video(self._mp.Image(self._mp.ImageFormat.SRGB, rgb), ms)

        self.skeletons = []
        for landmarks, world in zip(result.pose_landmarks, result.pose_world_landmarks):
            image_xy = np.array([(p.x * width, p.y * height) for p in landmarks], np.float32)
            visibility = np.array([p.visibility or 0.0 for p in landmarks], np.float32)
            model_xyz = np.array([(p.x, p.y, p.z) for p in world], np.float32)
            xyz = self._measure(image_xy, frame)
            self.skeletons.append(Skeleton(image_xy, visibility, xyz, model_xyz))
        return self.skeletons

    def _measure(self, image_xy: np.ndarray, frame: Frame) -> np.ndarray:
        """Each joint's 3D position from the depth under it (median of a small window)."""
        xyz = np.full((len(image_xy), 3), np.nan, np.float32)
        depth, k = frame.color_depth, frame.color_intrinsics
        if depth is None or k is None:
            return xyz
        h, w = depth.shape
        r = self.depth_radius
        for i, (u, v) in enumerate(image_xy):
            x, y = int(round(u)), int(round(v))
            if not (0 <= x < w and 0 <= y < h):
                continue
            patch = depth[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1]
            readings = patch[patch > 0]
            if readings.size:
                # Depth is measured from the depth camera, a few cm beside the color
                # camera; close enough for placing joints.
                z = float(np.median(readings)) / 1000.0
                xyz[i] = ((u - k.cx) * z / k.fx, (v - k.cy) * z / k.fy, z)
        return xyz

    def close(self) -> None:
        self._landmarker.close()


def draw_skeleton(
    image: np.ndarray,
    skeleton: Skeleton,
    color: tuple[int, int, int] = (0, 255, 255),
    threshold: float = 0.5,
    thickness: int = 3,
) -> np.ndarray:
    """Draw bones and joints in place. The image can be any size with the color
    camera's 16:9 framing (e.g. the 1920x1080 color image, or a scaled canvas)."""
    scale = image.shape[1] / 1920.0
    points = (skeleton.image_xy * scale).astype(int)
    for a, b in CONNECTIONS:
        if skeleton.visible(a, threshold) and skeleton.visible(b, threshold):
            cv2.line(image, tuple(points[a]), tuple(points[b]), color, thickness, cv2.LINE_AA)
    for i, p in enumerate(points):
        if skeleton.visible(i, threshold):
            cv2.circle(image, tuple(p), thickness + 2, color, -1, cv2.LINE_AA)
    return image
