"""The main loop: source -> sketch.update -> sketch.draw -> window."""

from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path

import cv2

from .sketch import Sketch

QUIT_KEYS = {27, ord("q")}  # Esc or q


def load_sketch(path: str | Path) -> Sketch:
    """Import a sketch file and instantiate the one Sketch subclass it defines."""
    path = Path(path).resolve()
    if not path.is_file():
        raise SystemExit(f"No sketch file at {path}")

    spec = importlib.util.spec_from_file_location(f"kinect_sketchbook_sketches.{path.stem}", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Can't load {path.name} as Python; sketches must be .py files")
    module = importlib.util.module_from_spec(spec)
    # @dataclass (and similar) look the module up in sys.modules while the body runs.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    classes = [
        obj
        for obj in vars(module).values()
        if inspect.isclass(obj) and issubclass(obj, Sketch) and obj is not Sketch
        and obj.__module__ == module.__name__
    ]
    if len(classes) != 1:
        names = ", ".join(c.__name__ for c in classes) or "none"
        raise SystemExit(f"{path.name} must define exactly one Sketch subclass (found: {names})")
    return classes[0]()


def run(sketch: Sketch, source=None, show: bool = True, max_frames: int | None = None) -> None:
    """Run a sketch until q/Esc, Ctrl-C, max_frames, or the source runs out."""
    if source is None:
        from .sources import KinectSource

        source = KinectSource(streams=sketch.streams)

    sketch.setup()
    if show:
        # Scaling to the window is done by HighGUI (imshow), not by resizing the
        # numpy buffer. keep_aspect picks letterbox vs free stretch.
        # Note: on this OpenCV build WINDOW_KEEPRATIO == 0 (default); free stretch
        # requires WINDOW_FREERATIO explicitly.
        keep_aspect = getattr(sketch, "keep_aspect", True)
        flags = cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL
        if keep_aspect:
            flags |= cv2.WINDOW_KEEPRATIO
        else:
            flags |= cv2.WINDOW_FREERATIO
        cv2.namedWindow(sketch.title, flags)
        size = getattr(sketch, "window_size", None)
        if size is not None:
            cv2.resizeWindow(sketch.title, int(size[0]), int(size[1]))
        if getattr(sketch, "fill_display", False):
            cv2.setWindowProperty(
                sketch.title, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN,
            )
    try:
        with source:
            for frame in source:
                sketch.update(frame)

                if show:
                    image = sketch.draw(frame)
                    if image is not None:
                        cv2.imshow(sketch.title, image)
                    key = cv2.waitKey(1) & 0xFF
                    if key in QUIT_KEYS:
                        break
                    if key != 0xFF:
                        sketch.on_key(chr(key))

                if max_frames is not None and frame.index + 1 >= max_frames:
                    break
    except KeyboardInterrupt:
        pass
    finally:
        sketch.teardown()
        if show:
            cv2.destroyAllWindows()
