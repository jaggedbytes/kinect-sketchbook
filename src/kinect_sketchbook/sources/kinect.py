"""Kinect v2 source, via libfreenect2's Python bindings (pylibfreenect2)."""

from __future__ import annotations

import time
from collections.abc import Iterator

import numpy as np

from ..frame import (
    COLOR_HEIGHT, COLOR_WIDTH, DEFAULT_INTRINSICS, DEPTH_HEIGHT, DEPTH_WIDTH, Frame, Intrinsics,
)

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
            "  It needs Python 3.12 with cython<3, numpy<2 and setuptools<80; see Setup in the README.\n"
            "  Also make sure the libfreenect2 shared library is on your library path\n"
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
        self._registration = None
        self.intrinsics: Intrinsics = DEFAULT_INTRINSICS
        self.color_intrinsics: Intrinsics | None = None

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
        self.intrinsics = self._read_intrinsics()
        if "color" in self.streams:
            self._start_registration(lib)
        return self

    def _start_registration(self, lib) -> None:
        """Reproject depth into the color view each frame, using the factory calibration."""
        ir_params = self._device.getIrCameraParams()
        color_params = self._device.getColorCameraParams()
        self._registration = lib.Registration(ir_params, color_params)
        self._undistorted = lib.Frame(DEPTH_WIDTH, DEPTH_HEIGHT, 4)
        self._registered = lib.Frame(DEPTH_WIDTH, DEPTH_HEIGHT, 4)
        # libfreenect2 pads bigdepth with one row above and below the color image.
        self._bigdepth = lib.Frame(COLOR_WIDTH, COLOR_HEIGHT + 2, 4)
        cx = color_params.cx
        if self.mirror:
            cx = COLOR_WIDTH - 1 - cx
        self.color_intrinsics = Intrinsics(
            fx=color_params.fx, fy=color_params.fy, cx=cx, cy=color_params.cy,
            width=COLOR_WIDTH, height=COLOR_HEIGHT,
        )

    def _read_intrinsics(self) -> Intrinsics:
        """The depth camera's factory calibration (only readable once streams have started)."""
        try:
            p = self._device.getIrCameraParams()
            intrinsics = Intrinsics(fx=p.fx, fy=p.fy, cx=p.cx, cy=p.cy)
        except Exception:
            intrinsics = DEFAULT_INTRINSICS
        if self.mirror:
            # Flipping the image horizontally moves the principal point too.
            intrinsics = Intrinsics(
                fx=intrinsics.fx, fy=intrinsics.fy,
                cx=intrinsics.width - 1 - intrinsics.cx, cy=intrinsics.cy,
            )
        return intrinsics

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
                color = color_depth = None
                if "color" in self.streams:
                    # libfreenect2 delivers BGRX; drop the padding byte.
                    color = np.ascontiguousarray(frames["color"].asarray()[:, :, :3])
                    self._registration.apply(
                        frames["color"], frames["depth"], self._undistorted, self._registered,
                        bigdepth=self._bigdepth,
                    )
                    color_depth = self._bigdepth.asarray(np.float32)[1:-1].copy()
                    color_depth[~np.isfinite(color_depth)] = 0  # match depth: 0 = no reading
            finally:
                self._listener.release(frames)

            if self.mirror:
                depth = np.fliplr(depth).copy()
                ir = None if ir is None else np.fliplr(ir).copy()
                color = None if color is None else np.fliplr(color).copy()
                color_depth = None if color_depth is None else np.fliplr(color_depth).copy()

            yield Frame(
                index=index, timestamp=time.monotonic(),
                depth=depth, ir=ir, color=color, intrinsics=self.intrinsics,
                color_depth=color_depth, color_intrinsics=self.color_intrinsics,
            )
            index += 1

    def __exit__(self, *exc) -> None:
        if self._device is not None:
            self._device.stop()
            self._device.close()
        self._fn = self._device = self._listener = self._registration = None

