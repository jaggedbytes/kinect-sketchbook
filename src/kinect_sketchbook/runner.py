"""The main loop: source -> sketch.update -> sketch.draw -> window."""

from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path

import cv2

from .sketch import Sketch

QUIT_KEYS = {27, ord("q")}  # Esc or q


def load_sketch(path: str | Path) -> Sketch:
    """Import a sketch file and instantiate the one Sketch subclass it defines."""
    path = Path(path).resolve()
    if not path.is_file():
        raise SystemExit(f"No sketch file at {path}")

    spec = importlib.util.spec_from_file_location(f"depthsketch_user.{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
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
