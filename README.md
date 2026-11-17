# kinect-sketchbook

A tiny framework for writing standalone Kinect v2 sketches, one file per sketch. The runner owns the device and the loop; a sketch just reacts to frames.

```python
from kinect_sketchbook import Sketch

class Hello(Sketch):
    def update(self, frame):
        print(frame.index, frame.depth[212, 256], "mm at the center")
```

```sh
sketchbook run sketches/hello.py
```

## Requirements

- A Kinect v2 with its power/USB adapter, plugged **directly** into a USB 3 port (not through a hub).
- [libfreenect2](https://github.com/OpenKinect/libfreenect2), built and installed (see [Building libfreenect2](#building-libfreenect2)).
- **Python 3.12.** The Kinect bindings (`pylibfreenect2`) are old and won't build on Python 3.13+, Cython 3, or numpy 2.

## Setup

Run these from the repo root.

```sh
# 1. A Python 3.12 venv inside the repo
sudo dnf install python3.12 python3.12-devel     # Fedora
# brew install python@3.12                       # macOS
python3.12 -m venv .venv && source .venv/bin/activate

# 2. Build tools pinned to versions pylibfreenect2 works with
pip install "cython<3" "numpy<2" "setuptools<80" wheel

# 3. The Kinect bindings, built against your libfreenect2 install
LIBFREENECT2_INSTALL_PREFIX=$HOME/.local pip install --no-build-isolation pylibfreenect2

# 4. This project (keep numpy<2 so pip doesn't upgrade it)
pip install -e . "numpy<2"
```

Make the libfreenect2 library findable at runtime by adding this to `~/.bashrc` (or `~/.zshrc` on macOS):

```sh
export LD_LIBRARY_PATH=$HOME/.local/lib:$LD_LIBRARY_PATH        # Linux
# export DYLD_LIBRARY_PATH=$HOME/.local/lib:$DYLD_LIBRARY_PATH  # macOS
```

Check it, then run the example:

```sh
python -c "import pylibfreenect2; print('ok')"
sketchbook run sketches/swipe.py
```

Why each pin is there:

| Pin | Reason |
| --- | --- |
| Python 3.12 | Cython 0.29 can't generate code for 3.13+. |
| `cython<3` | Under Cython 3, pylibfreenect2's declarations aren't found (`'Frame' is not a type identifier`). |
| `numpy<2` | pylibfreenect2 is compiled against the numpy 1 C API and fails to import under numpy 2. |
| `setuptools<80` | pylibfreenect2 imports `pkg_resources`, which newer setuptools removed. |
| `--no-build-isolation` | pylibfreenect2's `setup.py` needs numpy but doesn't declare it, so an isolated build fails with `No module named 'numpy'`. |

## Writing a sketch

Put one file in `sketches/` containing one `Sketch` subclass, and override what you need:

| Method | When |
| --- | --- |
| `setup()` | once, before the first frame |
| `update(frame)` | every frame; tracking and logic go here |
| `draw(frame)` | every frame; return a BGR image to show (default: the depth view) |
| `on_key(key)` | a key was pressed in the window |
| `teardown()` | once, on exit |

Set `streams = ("depth", "ir", "color")` (any subset) to choose what's captured. Only ask for what you use; color costs CPU to decode.

What a frame contains:

| Field | Shape | Units |
| --- | --- | --- |
| `frame.depth` | 424×512 float32 | millimeters; 0 = no reading. Always present. |
| `frame.ir` | 424×512 float32 | intensity, or `None` |
| `frame.color` | 1080×1920×3 uint8 | BGR, or `None` |
| `frame.intrinsics` | `fx`, `fy`, `cx`, `cy` | depth camera calibration in pixels, read from the device |
| `frame.color_depth` | 1080×1920 float32 | millimeters, depth reprojected into the color camera's view, so `color_depth[y, x]` is the distance at `color[y, x]`; 0 = no reading. Only with `"color"`. |
| `frame.color_intrinsics` | `fx`, `fy`, `cx`, `cy` | color camera calibration, for `color` and `color_depth`. Only with `"color"`. |
| `frame.index`, `frame.timestamp` | | frame count, seconds (monotonic) |

To turn a depth pixel `(u, v)` into a 3D point in meters: `z = depth / 1000`, `x = (u - cx) * z / fx`, `y = (v - cy) * z / fy` (x right, y down, z away from the camera).

`depth_to_bgr(depth)` turns a depth map into a viewable image (near = bright).

## Tracking

`kinect_sketchbook.tracking` holds the reusable building blocks; a sketch combines them into an interaction. Sketches opt in by creating what they need in `setup()` and calling it in `update()`.

| Module | What it gives you |
| --- | --- |
| `hand` | `nearest_hand(depth)`: whatever is closest to the camera, as a mask and centroid. Depth only, no model. |
| `gestures` | `SwipeDetector`: feed it an x position each frame; it returns `"left"`/`"right"` when a swipe happens. Takes plain numbers, so it works with any tracker. |
| `pose` | `PoseTracker`: skeletons from a machine-learning model, measured with depth (below). |

### Pose

`PoseTracker` runs a machine-learning model ([MediaPipe Pose Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker)) on the color image to find up to `num_poses` people, 33 joints each. Each joint is then placed in 3D by looking up the Kinect depth under it.

```sh
pip install mediapipe "numpy<2"
mkdir -p models && curl -L -o models/pose_landmarker_full.task \
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task
```

```python
from kinect_sketchbook.tracking import Joint, PoseTracker, draw_skeleton

class Wave(Sketch):
    streams = ("depth", "color")   # the model needs color

    def setup(self):
        self.pose = PoseTracker()     # variant="lite" | "full" | "heavy"

    def update(self, frame):
        for body in self.pose.update(frame):
            if body.visible(Joint.RIGHT_WRIST):
                print(body.xyz[Joint.RIGHT_WRIST])   # meters, measured
```

Each `Skeleton` keeps the model's guess and the sensor's measurement apart:

| Field | Shape | Meaning |
| --- | --- | --- |
| `image_xy` | 33×2 | joint positions in color-image pixels |
| `visibility` | 33 | the model's confidence (0–1) that the joint is in view |
| `xyz` | 33×3 | meters in camera space, **measured** from depth; NaN where there's no reading |
| `model_xyz` | 33×3 | meters, centered between the hips, **guessed** by the model |

Things to know:

- Running the model and the color stream costs about 20–25 ms per frame (measured on the aarch64 Linux machine this was written on). Use `variant="lite"` if it can't keep up.
- With `--mirror`, the model sees a flipped image, so it labels your right hand `LEFT_*`. Positions on screen are still correct.
- A joint that's hidden behind something gets that thing's depth in `xyz`. Check `visible()` first.
- The model only sees what's in the color image, which needs light. Depth works in the dark; pose doesn't.

## Running

```
sketchbook run SKETCH [--pipeline gl|cl|cpu] [--mirror] [--no-window] [--frames N] [--serial S]
```

- `--pipeline` picks the depth decoder. `gl` is the default and the safest; `cl` uses OpenCL; `cpu` always works but is slower.
- `--mirror` flips frames horizontally. Use it if on-screen motion goes the opposite way from your hand.
- Press `q` or `Esc` in the window to quit.

## Sketches

- **`swipe.py`**: treats whatever is closest to the camera (within 10 cm of the nearest point) as your hand, shown in green, and reports left and right swipes. It's sensitive to anything else being closer than your hand, and its thresholds are untuned.
- **`pose.py`**: finds up to three people with `PoseTracker` and draws their skeletons over the depth view at 1920×1080, labeling each wrist with its measured distance. The model sees the color image, but only depth is shown. Needs the pose setup in [Tracking](#tracking).
- **`mesh.py`**: builds a 3D mesh from a series of depth snapshots and shows it in a viewer with x/y/z grid planes. Needs Open3D (`pip install open3d "numpy<2"`).

### Using `mesh.py`

1. Run `sketchbook run sketches/mesh.py` and point the Kinect at an object at least 50 cm away.
2. Press `[` and `]` to move the far limit until only the object is highlighted in blue. Anything outside the limit (the wall behind it, for example) is left out of the mesh.
3. Press space to take a snapshot. Move the Kinect a little around the object, or turn the object about 10–15°, and take another. Repeat.
4. Press `m` to build the mesh. It's saved to `captures/<timestamp>/mesh.ply` (along with the raw snapshots in `snaps.npz`) and opens in a viewer:
   - The mesh sits on the floor grid (y = 0), centered on x and z, in meters.
   - Grid lines are every 5 cm, on the floor, the back wall, and the left wall.
   - The axes at the grid corner are red = x, green = y (up), and blue = z.
   - Drag to orbit, scroll to zoom, and close the window to return to the live view.
5. Press `c` to clear the snapshots and start over.

Rebuild a saved capture without the Kinect (useful for experimenting on another machine):

```sh
python sketches/mesh.py captures/<timestamp>/snaps.npz
```

`.ply` files also open in MeshLab and Blender.

Tips:

- Overlap matters more than coverage. Each snapshot is aligned to the one before it, so small steps work better than big ones. If a step was too big, the console prints a warning with the overlap percentage.
- Matte, textured objects work best. Shiny, black, or see-through surfaces return little or no depth.
- The Kinect can't see anything closer than about 50 cm.

## Building libfreenect2

Install it to `~/.local`. Keep the source checkout wherever you keep code; only the install prefix matters to this project.

### Fedora (including Asahi)

```sh
sudo dnf install git cmake gcc-c++ make libusb1-devel glfw-devel turbojpeg-devel \
  mesa-libGL-devel OpenCL-ICD-Loader-devel opencl-headers mesa-libOpenCL

git clone https://github.com/OpenKinect/libfreenect2.git && cd libfreenect2

# Newer OpenCL headers define CL_ICDL_VERSION as a macro, which breaks libfreenect2's OpenCL code.
sed -i 's/CL_ICDL_VERSION/LFN2_ICDL_VERSION/g' \
  src/opencl_depth_packet_processor.cpp src/opencl_kde_depth_packet_processor.cpp

mkdir build && cd build
cmake .. -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DENABLE_CUDA=OFF -DCMAKE_INSTALL_PREFIX=$HOME/.local
make -j$(nproc) && make install

# Let non-root users open the Kinect, then unplug and replug it.
sudo cp ../platform/linux/udev/90-kinect2.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules && sudo udevadm trigger
```

### macOS (Apple Silicon)

```sh
brew install libusb glfw jpeg-turbo pkg-config
git clone https://github.com/OpenKinect/libfreenect2.git && cd libfreenect2
mkdir build && cd build
cmake .. -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DENABLE_CUDA=OFF -DCMAKE_INSTALL_PREFIX=$HOME/.local
make -j && make install
```

### Check the device

From the libfreenect2 `build` directory, run `./bin/Protonect gl`. You should see live depth, IR, and color windows. If Protonect doesn't work, nothing here will.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `Compatibility with CMake < 3.5 has been removed` | Add `-DCMAKE_POLICY_VERSION_MINIMUM=3.5` to `cmake`. |
| `make install` can't create `/usr/local/lib` | Set `-DCMAKE_INSTALL_PREFIX=$HOME/.local`, or use `sudo make install`. |
| `expected unqualified-id before numeric constant` at `CL_ICDL_VERSION` | Run the `sed` fix above, or build with `-DENABLE_OPENCL=OFF`. |
| dnf: `OpenCL-ICD-Loader conflicts with ocl-icd` | Install `OpenCL-ICD-Loader-devel`, not `ocl-icd-devel`. |
| Protonect says "viewer is turned off" | Install GLFW (`glfw-devel` / `brew install glfw`) and re-run `cmake`. |
| `LIBUSB_ERROR_ACCESS`, or it only works with `sudo` | Install the udev rule, then replug the Kinect. |
| Lots of "packets were lost," or no device found | The Kinect is probably on USB 2. Plug it straight into the computer; `lsusb -t` should show `5000M`. |
| `--pipeline cl` finds no OpenCL device on Asahi | Run with `RUSTICL_ENABLE=asahi`. |
| `pylibfreenect2 isn't importable` / `No module named 'pylibfreenect2'` | The bindings aren't in the active venv. Check `which sketchbook python pip` all point to the same `.venv`, then redo setup step 3. |
| `cannot open shared object file: libfreenect2.so` | Set `LD_LIBRARY_PATH` (see Setup). |
| `No module named 'numpy'` while building pylibfreenect2 | Use `--no-build-isolation`. |
| `'Frame' is not a type identifier` while building pylibfreenect2 | Use Python 3.12 with `cython<3`. |
| `No module named 'pkg_resources'` | `pip install "setuptools<80"`. |
