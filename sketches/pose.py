"""Pose: a machine-learning model guesses your skeleton; the Kinect measures it.

The model runs on the color image, but only depth is shown: depth reprojected into
the color camera's 1920x1080 view, with each skeleton drawn over it. The label at
each wrist is its measured distance from the sensor.

Run:  sketchbook run sketches/pose.py --mirror
Keys: q / Esc to quit.
"""

import cv2
import numpy as np

from kinect_sketchbook import Frame, Sketch, depth_to_bgr
from kinect_sketchbook.tracking import Joint, PoseTracker, draw_skeleton

COLORS = [(0, 255, 255), (255, 0, 255), (255, 255, 0)]  # one per person (BGR)
LABELLED = (Joint.LEFT_WRIST, Joint.RIGHT_WRIST)


class Pose(Sketch):
    title = "pose"
    streams = ("depth", "color")

    def setup(self):
        self.pose = PoseTracker(num_poses=len(COLORS))

    def update(self, frame: Frame):
        self.pose.update(frame)

    def draw(self, frame: Frame):
        assert frame.color_depth is not None, 'needs streams = ("depth", "color")'
        image = depth_to_bgr(frame.color_depth)
        for body, color in zip(self.pose.skeletons, COLORS):
            draw_skeleton(image, body, color)
            for joint in LABELLED:
                z = body.xyz[joint, 2]
                if body.visible(joint) and np.isfinite(z):
                    x, y = body.image_xy[joint].astype(int)
                    cv2.putText(image, f"{z:.2f} m", (x + 12, y - 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
        cv2.putText(image, f"people: {len(self.pose.skeletons)}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
        return image

    def teardown(self):
        self.pose.close()
