"""Hand gestures from MediaPipe's Gesture Recognizer.

The canned classifier names: None, Closed_Fist, Open_Palm, Pointing_Up,
Thumb_Down, Thumb_Up, Victory, ILoveYou.

Needs MediaPipe and the model file (see the README):
    pip install mediapipe "numpy<2"
    models/gesture_recognizer.task

    from kinect_sketchbook.tracking import GestureTracker

    class Palm(Sketch):
        streams = ("color",)
        def setup(self):
            self.gestures = GestureTracker()
        def update(self, frame):
            if self.gestures.sees(frame, "Open_Palm"):
                print("open palm")
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ..frame import Frame

MODEL_DIR = Path("models")
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/"
    "gesture_recognizer/float16/1/gesture_recognizer.task"
)
DEFAULT_MODEL = MODEL_DIR / "gesture_recognizer.task"

# Same shrink PoseTracker uses so a full-frame pass stays cheap.
DETECT_WIDTH = 960

# Upscale tight (far) crops so the short-range model still sees a large hand.
ROI_MIN_INPUT = 256

# MediaPipe hand landmark used as the hand's anchor in the image.
_WRIST = 0

# Kinect v2 color focal length, used when the frame has no calibration yet.
_DEFAULT_COLOR_FX = 1081.0


@dataclass
class HandGesture:
    """One hand's top gesture and where it is in the color image."""

    name: str
    score: float
    image_xy: np.ndarray  # (2,) pixels in the full color frame


def _require_mediapipe():
    try:
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision
        from mediapipe.tasks.python.components.processors import ClassifierOptions
    except ImportError as e:
        raise SystemExit(
            f'Gesture recognition needs MediaPipe:  pip install mediapipe "numpy<2"\n  ({e})'
        ) from e
    return mp, BaseOptions, vision, ClassifierOptions


def _model_path(model_path: str | Path | None) -> Path:
    path = Path(model_path) if model_path is not None else DEFAULT_MODEL
    if not path.is_file():
        raise SystemExit(
            f"No gesture model at {path}. Download it (from the repo root) with:\n"
            f"  mkdir -p {path.parent} && curl -L -o {path} {MODEL_URL}"
        )
    return path


class GestureTracker:
    """Runs MediaPipe's Gesture Recognizer on each frame's color image.

    gestures: names to keep (e.g. ("Open_Palm",)). Empty means every canned gesture.
    num_hands: most hands to look at in a full-frame pass.
    min_score: classifier score below which a gesture is ignored.

    Prefer recognize_rois() when you already know where a hand should be: a
    depth-scaled crop keeps far hands large enough for the short-range model.
    """

    def __init__(
        self,
        gestures: tuple[str, ...] = (),
        num_hands: int = 2,
        min_score: float = 0.5,
        min_confidence: float = 0.5,
        model_path: str | Path | None = None,
    ) -> None:
        self._mp, BaseOptions, vision, ClassifierOptions = _require_mediapipe()
        self._vision = vision
        self._BaseOptions = BaseOptions
        self._ClassifierOptions = ClassifierOptions
        self._model_path = str(_model_path(model_path))
        self._gestures = gestures
        self._min_score = min_score
        self._min_confidence = min_confidence
        self._num_hands = num_hands

        self._recognizer = self._make_recognizer(vision.RunningMode.VIDEO, num_hands)
        self._roi_recognizer = None  # IMAGE mode, created on first ROI call
        self._last_ms = -1
        self.hands: list[HandGesture] = []
        self.names: list[str] = []
        self.scores: list[float] = []

    def _make_recognizer(self, running_mode, num_hands):
        classifier = self._ClassifierOptions(
            score_threshold=self._min_score,
            category_allowlist=list(self._gestures) if self._gestures else None,
        )
        options = self._vision.GestureRecognizerOptions(
            base_options=self._BaseOptions(model_asset_path=self._model_path),
            running_mode=running_mode,
            num_hands=num_hands,
            min_hand_detection_confidence=self._min_confidence,
            min_hand_presence_confidence=self._min_confidence,
            min_tracking_confidence=self._min_confidence,
            canned_gestures_classifier_options=classifier,
        )
        return self._vision.GestureRecognizer.create_from_options(options)

    def _roi_engine(self):
        if self._roi_recognizer is None:
            # One or two hands inside a wrist crop is enough.
            self._roi_recognizer = self._make_recognizer(self._vision.RunningMode.IMAGE, 2)
        return self._roi_recognizer

    def update(self, frame: Frame) -> list[str]:
        """Top gesture name for each hand this frame. Also kept in self.names / self.hands."""
        if frame.color is None:
            raise ValueError('GestureTracker needs the color stream: set streams = ("color",) or ("depth", "color")')

        height, width = frame.color.shape[:2]
        small = cv2.resize(
            frame.color,
            (DETECT_WIDTH, DETECT_WIDTH * height // width),
            interpolation=cv2.INTER_AREA,
        )
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        ms = max(int(frame.timestamp * 1000), self._last_ms + 1)
        self._last_ms = ms
        result = self._recognizer.recognize_for_video(
            self._mp.Image(self._mp.ImageFormat.SRGB, rgb), ms
        )
        hands = self._hands_from_result(result, 0, 0, width, height)
        self.hands = hands
        self.names = [h.name for h in hands]
        self.scores = [h.score for h in hands]
        return self.names

    def recognize_rois(
        self,
        frame: Frame,
        rois: list[tuple[int, int, int, int]],
    ) -> list[list[HandGesture]]:
        """Run the recognizer on each axis-aligned crop (x0, y0, x1, y1).

        Landmarks are returned in full color-image pixels. Empty rois are skipped
        and yield an empty list. IMAGE mode is used so unrelated crops don't
        confuse video tracking.
        """
        if frame.color is None:
            raise ValueError('GestureTracker needs the color stream: set streams = ("color",) or ("depth", "color")')

        engine = self._roi_engine()
        height, width = frame.color.shape[:2]
        out: list[list[HandGesture]] = []
        for x0, y0, x1, y1 in rois:
            x0 = max(0, min(width - 1, int(x0)))
            y0 = max(0, min(height - 1, int(y0)))
            x1 = max(x0 + 1, min(width, int(x1)))
            y1 = max(y0 + 1, min(height, int(y1)))
            crop = frame.color[y0:y1, x0:x1]
            if crop.size == 0:
                out.append([])
                continue
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            # Contiguous: MediaPipe rejects some strided views from slicing.
            rgb = np.ascontiguousarray(rgb)
            # Far hands make a small crop; enlarge so the model isn't starved of pixels.
            ch, cw = rgb.shape[:2]
            scale = 1.0
            if min(ch, cw) < ROI_MIN_INPUT:
                scale = ROI_MIN_INPUT / float(min(ch, cw))
                rgb = cv2.resize(
                    rgb,
                    (max(1, int(round(cw * scale))), max(1, int(round(ch * scale)))),
                    interpolation=cv2.INTER_LINEAR,
                )
            result = engine.recognize(self._mp.Image(self._mp.ImageFormat.SRGB, rgb))
            # Landmarks are normalized to the (possibly upscaled) crop; map via the
            # original crop size so image_xy stays in full-frame pixels.
            out.append(self._hands_from_result(result, x0, y0, x1 - x0, y1 - y0))
        return out

    def sees(self, frame: Frame, name: str) -> bool:
        """True when any hand shows this gesture on this frame."""
        return name in self.update(frame)

    def close(self) -> None:
        self._recognizer.close()
        if self._roi_recognizer is not None:
            self._roi_recognizer.close()
            self._roi_recognizer = None

    def _hands_from_result(self, result, origin_x, origin_y, box_w, box_h) -> list[HandGesture]:
        hands: list[HandGesture] = []
        for gesture_list, landmarks in zip(result.gestures, result.hand_landmarks):
            if not gesture_list or not landmarks:
                continue
            top = gesture_list[0]
            name = top.category_name or top.display_name or ""
            if not name or name == "None":
                continue
            wrist = landmarks[_WRIST]
            hands.append(HandGesture(
                name=name,
                score=float(top.score),
                image_xy=np.array(
                    [origin_x + wrist.x * box_w, origin_y + wrist.y * box_h],
                    np.float32,
                ),
            ))
        return hands


def hand_roi(
    image_xy: np.ndarray,
    depth_m: float,
    frame_shape: tuple[int, int],
    fx: float = _DEFAULT_COLOR_FX,
    hand_m: float = 0.45,
    min_px: int = 160,
    max_px: int = 560,
) -> tuple[int, int, int, int] | None:
    """Axis-aligned crop around a hand, sized from depth so far hands stay large.

    image_xy: hand center in color pixels.
    depth_m:  distance of that hand in meters.
    frame_shape: (height, width) of the color image.
    hand_m:   meters of world space the square crop should cover.
    """
    if not np.isfinite(depth_m) or depth_m < 0.4:
        return None
    height, width = frame_shape
    side = int(np.clip(hand_m * fx / depth_m, min_px, max_px))
    cx, cy = float(image_xy[0]), float(image_xy[1])
    if not (0 <= cx < width and 0 <= cy < height):
        return None
    half = side // 2
    x0 = int(round(cx)) - half
    y0 = int(round(cy)) - half
    x1 = x0 + side
    y1 = y0 + side
    # Shift into the image instead of shrinking (keeps the depth-scaled size).
    if x0 < 0:
        x1 -= x0
        x0 = 0
    if y0 < 0:
        y1 -= y0
        y0 = 0
    if x1 > width:
        x0 -= x1 - width
        x1 = width
    if y1 > height:
        y0 -= y1 - height
        y1 = height
    x0 = max(0, x0)
    y0 = max(0, y0)
    if x1 - x0 < min_px // 2 or y1 - y0 < min_px // 2:
        return None
    return x0, y0, x1, y1
