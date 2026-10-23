"""Command line: `depthsketch run sketches/swipe.py`."""

from __future__ import annotations

import argparse

from .runner import load_sketch, run
from .sources.kinect import PIPELINES, KinectSource


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="depthsketch")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="run a sketch file")
    run_p.add_argument("sketch", help="path to a sketch file, e.g. sketches/swipe.py")
    run_p.add_argument("--pipeline", choices=sorted(PIPELINES), default="gl",
                       help="depth processing pipeline (default: gl)")
    run_p.add_argument("--serial", help="Kinect serial number, if more than one is connected")
    run_p.add_argument("--mirror", action="store_true",
                       help="flip frames horizontally (use if on-screen motion is opposite your hand)")
    run_p.add_argument("--no-window", action="store_true", help="don't open a window")
    run_p.add_argument("--frames", type=int, help="stop after this many frames")

    args = parser.parse_args(argv)

    if args.command == "run":
        sketch = load_sketch(args.sketch)
        source = KinectSource(
            streams=sketch.streams,
            pipeline=args.pipeline,
            serial=args.serial,
            mirror=args.mirror,
        )
        run(sketch, source=source, show=not args.no_window, max_frames=args.frames)


if __name__ == "__main__":
    main()
