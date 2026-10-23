"""Kinect v2 source, via libfreenect2's Python bindings (pylibfreenect2)."""

from __future__ import annotations

import time
from collections.abc import Iterator

import numpy as np

from ..frame import Frame

PIPELINES = {
    "gl": "OpenGLPacketPipeline",
    "cl": "OpenCLPacketPipeline",
    "cpu": "CpuPacketPipeline",
}


def _import_pylibfreenect2():
    try:
        import pylibfreenect2  # noqa: F401
    except ImportError as e:
        raise SystemExit(
            "pylibfreenect2 isn't importable.\n"
            "  Install it with:  LIBFREENECT2_INSTALL_PREFIX=$HOME/.local pip install -e '.[kinect]'\n"
            "  and make sure the libfreenect2 shared library is on your library path\n"
            "  (LD_LIBRARY_PATH on Linux, DYLD_LIBRARY_PATH on macOS).\n"
            f"  Original error: {e}"
        ) from e
    return pylibfreenect2


class KinectSource:
    """Opens the first (or a given) Kinect v2 and yields Frames until closed.

    Use as a context manager:
        with KinectSource(streams=("depth",)) as source:
            for frame in source: ...
    """

    def __init__(
        self,
        streams: tuple[str, ...] = ("depth",),
        pipeline: str = "gl",
        serial: str | None = None,
        mirror: bool = False,
    ) -> None:
        unknown = set(streams) - {"depth", "ir", "color"}
        if unknown:
            raise ValueError(f"unknown streams: {sorted(unknown)}")
        if pipeline not in PIPELINES:
            raise ValueError(f"pipeline must be one of {sorted(PIPELINES)}")
        self.streams = streams
        self.pipeline = pipeline
        self.serial = serial
        self.mirror = mirror
        self._fn = self._device = self._listener = None

    def __enter__(self) -> KinectSource:
        lib = _import_pylibfreenect2()

        pipeline_cls = getattr(lib, PIPELINES[self.pipeline], None)
        if pipeline_cls is None:
            raise SystemExit(
                f"The '{self.pipeline}' pipeline isn't available in this libfreenect2 build. "
                "Try --pipeline gl or --pipeline cpu."
            )

        self._fn = lib.Freenect2()
        if self._fn.enumerateDevices() == 0:
            raise SystemExit("No Kinect v2 found. Check the USB 3 connection (and the udev rule on Linux).")
        serial = self.serial or self._fn.getDeviceSerialNumber(0)
        self._device = self._fn.openDevice(serial, pipeline=pipeline_cls())

        # Depth is always captured: every Frame carries it.
        types = lib.FrameType.Depth
        if "ir" in self.streams:
            types |= lib.FrameType.Ir
        if "color" in self.streams:
            types |= lib.FrameType.Color
        self._listener = lib.SyncMultiFrameListener(types)

        if "color" in self.streams:
            self._device.setColorFrameListener(self._listener)
        self._device.setIrAndDepthFrameListener(self._listener)
        self._device.startStreams(rgb="color" in self.streams, depth=True)
        return self

    def __iter__(self) -> Iterator[Frame]:
        if self._listener is None:
            raise RuntimeError("use KinectSource as a context manager")
        index = 0
        while True:
            frames = self._listener.waitForNewFrame()
            try:
                # Copy out before release(): the buffers are reused by libfreenect2.
                depth = frames["depth"].asarray().copy()
                ir = frames["ir"].asarray().copy() if "ir" in self.streams else None
                color = None
                if "color" in self.streams:
                    # libfreenect2 delivers BGRX; drop the padding byte.
                    color = np.ascontiguousarray(frames["color"].asarray()[:, :, :3])
            finally:
                self._listener.release(frames)

            if self.mirror:
                depth = np.fliplr(depth).copy()
                ir = None if ir is None else np.fliplr(ir).copy()
                color = None if color is None else np.fliplr(color).copy()

            yield Frame(index=index, timestamp=time.monotonic(), depth=depth, ir=ir, color=color)
            index += 1

    def __exit__(self, *exc) -> None:
        if self._device is not None:
            self._device.stop()
            self._device.close()
        self._fn = self._device = self._listener = None
