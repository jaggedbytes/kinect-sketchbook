"""Mesh: capture depth snapshots and fuse them into a 3D mesh.

Point the Kinect at an object, set the far limit so only the object is highlighted,
then take snapshots while moving the Kinect (or turning the object) a little
between each one. Press m to align the snapshots, fuse them into a mesh, save it,
and open it in a viewer with x/y/z grid planes.

Run:  sketchbook run sketches/mesh.py
Keys: space  take a snapshot
      [ / ]  move the far limit closer / farther (10 cm steps)
      m      build, save and view the mesh
      c      clear snapshots
      q/Esc  quit

Rebuild a saved capture without the Kinect:
      python sketches/mesh.py captures/<timestamp>/snaps.npz

Tips: keep each move small (about 10-15 degrees, or a hand's width) so consecutive
snapshots overlap; textured, matte objects align best; shiny, black or
see-through surfaces return little or no depth.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

try:
    import open3d as o3d
except ImportError:
    raise SystemExit('This sketch needs Open3D:  pip install open3d "numpy<2"')

from kinect_sketchbook import Frame, Intrinsics, Sketch, depth_to_bgr

CAPTURE_DIR = Path("captures")

NEAR_MM = 500          # the Kinect v2 can't see closer than this
VOXEL_M = 0.004        # mesh resolution: 4 mm
GRID_STEP_M = 0.05     # viewer grid spacing: 5 cm
MIN_FITNESS = 0.3      # warn if less than 30% of a snapshot overlaps the previous one

# Camera space is x right, y down, z forward. Flip to y up, z toward the viewer.
CAMERA_TO_VIEW = np.diag([1.0, -1.0, -1.0, 1.0])


# --- Building the mesh -------------------------------------------------------

def ir_to_gray(ir: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray:
    """IR intensity as an 8-bit image, used to shade the mesh."""
    if ir is None:
        return np.full(shape, 180, np.uint8)
    return (np.sqrt(np.clip(ir / 4000.0, 0, 1)) * 255).astype(np.uint8)


def clip(depth_mm, near_mm, far_mm) -> np.ndarray:
    """Depth as uint16 millimeters, with everything outside [near, far] set to 0 (ignored)."""
    depth = depth_mm.copy()
    depth[(depth < near_mm) | (depth > far_mm)] = 0
    return np.ascontiguousarray(depth.astype(np.uint16))


def shading(ir, shape) -> np.ndarray:
    gray = ir_to_gray(ir, shape)
    return np.ascontiguousarray(np.dstack([gray, gray, gray]))


def to_rgbd(depth_mm, ir, near_mm, far_mm) -> o3d.geometry.RGBDImage:
    return o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(shading(ir, depth_mm.shape)),
        o3d.geometry.Image(clip(depth_mm, near_mm, far_mm)),
        depth_scale=1000.0,
        depth_trunc=far_mm / 1000.0 + 0.01,
        convert_rgb_to_intensity=False,
    )


def to_o3d_intrinsic(k: Intrinsics) -> o3d.camera.PinholeCameraIntrinsic:
    return o3d.camera.PinholeCameraIntrinsic(k.width, k.height, k.fx, k.fy, k.cx, k.cy)


def to_cloud(rgbd, intrinsic) -> o3d.geometry.PointCloud:
    cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
    cloud = cloud.voxel_down_sample(VOXEL_M)
    cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=VOXEL_M * 4, max_nn=30))
    return cloud


def align(rgbds, intrinsic) -> tuple[list[np.ndarray], list[float]]:
    """Camera pose of each snapshot, relative to the first, by chaining ICP between neighbors.

    Returns (poses, fitness), where fitness[i] is the fraction of snapshot i that
    matched snapshot i-1 (fitness[0] is 1.0).
    """
    reg = o3d.pipelines.registration
    poses, fitness = [np.eye(4)], [1.0]
    prev = to_cloud(rgbds[0], intrinsic)
    for rgbd in rgbds[1:]:
        cur = to_cloud(rgbd, intrinsic)
        transform = np.eye(4)
        result = None
        # Coarse to fine. Point-to-point at the wide radii, then point-to-plane to
        # tighten: point-to-plane alone tends to slide along flat faces and lock
        # in a wrong rotation.
        stages = [
            (VOXEL_M * 10, reg.TransformationEstimationPointToPoint()),
            (VOXEL_M * 4, reg.TransformationEstimationPointToPoint()),
            (VOXEL_M * 1.5, reg.TransformationEstimationPointToPlane()),
        ]
        for radius, estimation in stages:
            result = reg.registration_icp(
                cur, prev, radius, transform, estimation,
                reg.ICPConvergenceCriteria(max_iteration=50),
            )
            transform = result.transformation
        poses.append(poses[-1] @ transform)
        fitness.append(result.fitness)
        prev = cur
    return poses, fitness


def fuse(depths, irs, intrinsic, poses, near_mm, far_mm) -> o3d.geometry.TriangleMesh:
    """Integrate every snapshot into a TSDF voxel grid and extract a surface.

    Uses Open3D's VoxelBlockGrid; the older ScalableTSDFVolume returns an empty
    mesh in Open3D 0.20.
    """
    core = o3d.core
    grid = o3d.t.geometry.VoxelBlockGrid(
        attr_names=("tsdf", "weight", "color"),
        attr_dtypes=(core.float32, core.float32, core.float32),
        attr_channels=((1), (1), (3)),
        voxel_size=VOXEL_M,
        block_resolution=16,
        block_count=50_000,
        device=core.Device("CPU:0"),
    )
    k = core.Tensor(intrinsic.intrinsic_matrix, core.float64)
    depth_max = far_mm / 1000.0 + 0.01
    for depth_mm, ir, pose in zip(depths, irs, poses):
        depth = o3d.t.geometry.Image(clip(depth_mm, near_mm, far_mm))
        color = o3d.t.geometry.Image(shading(ir, depth_mm.shape))
        extrinsic = core.Tensor(np.linalg.inv(pose), core.float64)  # world -> camera
        blocks = grid.compute_unique_block_coordinates(depth, k, extrinsic, 1000.0, depth_max)
        grid.integrate(blocks, depth, color, k, k, extrinsic, 1000.0, depth_max)

    # weight_threshold below 1 keeps surfaces seen by a single snapshot.
    mesh = grid.extract_triangle_mesh(weight_threshold=0.5).to_legacy()
    if mesh.has_vertex_colors():
        colors = np.asarray(mesh.vertex_colors)
        if colors.max() > 1.0:  # stored as 0-255
            mesh.vertex_colors = o3d.utility.Vector3dVector(colors / 255.0)
    mesh.compute_vertex_normals()
    return mesh


def place_on_grid(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    """Rotate to y-up and sit the mesh on y=0, centered on x and z."""
    mesh.transform(CAMERA_TO_VIEW)
    lo, hi = mesh.get_min_bound(), mesh.get_max_bound()
    mesh.translate([-(lo[0] + hi[0]) / 2, -lo[1], -(lo[2] + hi[2]) / 2])
    return mesh


def build_mesh(depths, irs, intrinsics, near_mm, far_mm):
    """Snapshots in, (mesh, fitness) out. The mesh is y-up, in meters, sitting on y=0."""
    intrinsic = to_o3d_intrinsic(intrinsics)
    rgbds = [to_rgbd(d, ir, near_mm, far_mm) for d, ir in zip(depths, irs)]
    poses, fitness = align(rgbds, intrinsic)
    mesh = fuse(depths, irs, intrinsic, poses, near_mm, far_mm)
    if mesh.has_triangles():
        place_on_grid(mesh)
    return mesh, fitness


# --- Viewing -----------------------------------------------------------------

def grid_planes(mesh, step=GRID_STEP_M) -> o3d.geometry.LineSet:
    """Three grid planes boxing in the mesh: floor (y=0), back wall (min z), left wall (min x)."""
    lo, hi = mesh.get_min_bound(), mesh.get_max_bound()
    lo = np.floor(lo / step) * step - step
    hi = np.ceil(hi / step) * step + step
    lo[1] = 0.0
    xs, ys, zs = (np.arange(lo[i], hi[i] + step / 2, step) for i in range(3))
    x0, y0, z0 = lo
    x1, y1, z1 = hi

    segments = []
    for z in zs:  # floor
        segments.append(([x0, y0, z], [x1, y0, z]))
    for x in xs:
        segments.append(([x, y0, z0], [x, y0, z1]))
    for y in ys:  # back wall
        segments.append(([x0, y, z0], [x1, y, z0]))
    for x in xs:
        segments.append(([x, y0, z0], [x, y1, z0]))
    for y in ys:  # left wall
        segments.append(([x0, y, z0], [x0, y, z1]))
    for z in zs:
        segments.append(([x0, y0, z], [x0, y1, z]))

    points = np.array([p for seg in segments for p in seg])
    lines = np.arange(len(points)).reshape(-1, 2)
    grid = o3d.geometry.LineSet(o3d.utility.Vector3dVector(points), o3d.utility.Vector2iVector(lines))
    grid.paint_uniform_color([0.6, 0.6, 0.6])
    return grid


def show(mesh, title="mesh") -> None:
    """Open a viewer: mesh, x/y/z grid planes, and axes at the grid corner
    (red = x, green = y, blue = z; grid lines every 5 cm). Blocks until closed."""
    grid = grid_planes(mesh)
    lo = np.asarray(grid.get_min_bound())
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=GRID_STEP_M * 2, origin=lo)
    o3d.visualization.draw_geometries(
        [mesh, grid, axes], window_name=title, width=1280, height=800, mesh_show_back_face=True,
    )


# --- Saving ------------------------------------------------------------------

def save(folder: Path, mesh, depths, irs, intrinsics: Intrinsics, near_mm, far_mm) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    o3d.io.write_triangle_mesh(str(folder / "mesh.ply"), mesh)
    np.savez_compressed(
        folder / "snaps.npz",
        depth=np.stack(depths), ir=np.stack(irs),
        intrinsics=np.array([intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy]),
        near_mm=near_mm, far_mm=far_mm,
    )


# --- The sketch --------------------------------------------------------------

class Mesh(Sketch):
    title = "mesh"
    streams = ("depth", "ir")
    show_viewer = True  # tests turn this off

    def setup(self):
        self.far_mm = 1500
        self.depths, self.irs = [], []
        self.intrinsics: Intrinsics | None = None
        self.latest: Frame | None = None
        self.message, self.message_until = "", 0.0

    def update(self, frame: Frame):
        self.latest = frame

    def notify(self, text: str, seconds: float = 2.0):
        print(text, flush=True)
        self.message, self.message_until = text, time.monotonic() + seconds

    def on_key(self, key: str):
        if key == " " and self.latest is not None:
            self.depths.append(self.latest.depth.copy())
            self.irs.append(self.latest.ir.copy() if self.latest.ir is not None
                            else np.zeros_like(self.latest.depth))
            self.intrinsics = self.latest.intrinsics
            self.notify(f"snapshot {len(self.depths)}")
        elif key == "[":
            self.far_mm = max(NEAR_MM + 100, self.far_mm - 100)
        elif key == "]":
            self.far_mm = min(4500, self.far_mm + 100)
        elif key == "c":
            self.depths, self.irs = [], []
            self.notify("cleared")
        elif key == "m":
            self.build()

    def build(self):
        if not self.depths:
            self.notify("take a snapshot first (space)")
            return
        print(f"building mesh from {len(self.depths)} snapshot(s)...", flush=True)
        mesh, fitness = build_mesh(self.depths, self.irs, self.intrinsics, NEAR_MM, self.far_mm)
        if not mesh.has_triangles():
            self.notify("no surface found: is anything highlighted between the near and far limits?")
            return

        for i, f in enumerate(fitness[1:], start=2):
            if f < MIN_FITNESS:
                print(f"  warning: snapshot {i} overlapped the previous one only {f:.0%}; "
                      "take smaller steps between snapshots", flush=True)

        folder = CAPTURE_DIR / time.strftime("%Y%m%d-%H%M%S")
        save(folder, mesh, self.depths, self.irs, self.intrinsics, NEAR_MM, self.far_mm)
        size = mesh.get_max_bound() - mesh.get_min_bound()
        self.notify(f"saved {folder}/mesh.ply  "
                    f"({size[0]*100:.0f} x {size[1]*100:.0f} x {size[2]*100:.0f} cm, "
                    f"{len(mesh.triangles):,} triangles)", seconds=5)
        if self.show_viewer:
            show(mesh, title=f"mesh: {folder.name}")

    def draw(self, frame: Frame):
        image = depth_to_bgr(frame.depth)
        in_range = (frame.depth >= NEAR_MM) & (frame.depth <= self.far_mm)
        image[in_range] = (0.5 * image[in_range] + (120, 60, 0)).astype(np.uint8)

        font = cv2.FONT_HERSHEY_SIMPLEX
        status = f"snapshots: {len(self.depths)}   far: {self.far_mm / 1000:.1f} m"
        cv2.putText(image, status, (10, 25), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(image, "space snap  [ ] far  m mesh  c clear", (10, image.shape[0] - 12),
                    font, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        if time.monotonic() < self.message_until:
            cv2.putText(image, self.message[:60], (10, 50), font, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        return image


# --- Rebuild a saved capture -------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python sketches/mesh.py captures/<timestamp>/snaps.npz")
    data = np.load(sys.argv[1])
    fx, fy, cx, cy = data["intrinsics"]
    k = Intrinsics(fx=fx, fy=fy, cx=cx, cy=cy)
    mesh, fitness = build_mesh(list(data["depth"]), list(data["ir"]), k,
                               float(data["near_mm"]), float(data["far_mm"]))
    print(f"{len(mesh.triangles):,} triangles; overlap per snapshot: "
          + ", ".join(f"{f:.0%}" for f in fitness))
    show(mesh)
