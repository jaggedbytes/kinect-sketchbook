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

from pathlib import Path

import cv2

from ..frame import Frame

MODEL_DIR = Path("models")
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/"
    "gesture_recognizer/float16/1/gesture_recognizer.task"
)
DEFAULT_MODEL = MODEL_DIR / "gesture_recognizer.task"

# Same shrink PoseTracker uses so a 1920×1080 frame is cheap to run.
DETECT_WIDTH = 960


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
    num_hands: most hands to look at.
    min_score: classifier score below which a gesture is ignored.
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
        classifier = ClassifierOptions(
            score_threshold=min_score,
            category_allowlist=list(gestures) if gestures else None,
        )
        options = vision.GestureRecognizerOptions(
            base_options=BaseOptions(model_asset_path=str(_model_path(model_path))),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=min_confidence,
            min_hand_presence_confidence=min_confidence,
            min_tracking_confidence=min_confidence,
            canned_gestures_classifier_options=classifier,
        )
        self._recognizer = vision.GestureRecognizer.create_from_options(options)
        self._last_ms = -1
        self.names: list[str] = []
        self.scores: list[float] = []

    def update(self, frame: Frame) -> list[str]:
        """Top gesture name for each hand this frame. Also kept in self.names."""
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

        names: list[str] = []
        scores: list[float] = []
        for hand in result.gestures:
            if not hand:
                continue
            top = hand[0]
            name = top.category_name or top.display_name or ""
            if not name or name == "None":
                continue
            names.append(name)
            scores.append(float(top.score))
        self.names = names
        self.scores = scores
        return names

    def sees(self, frame: Frame, name: str) -> bool:
        """True when any hand shows this gesture on this frame."""
        return name in self.update(frame)

    def close(self) -> None:
        self._recognizer.close()
