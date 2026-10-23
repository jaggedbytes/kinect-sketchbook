"""Swipe: hold a hand out in front of you and swipe left or right.

The hand is "whatever is closest to the camera": everything within HAND_SLAB_MM
of the nearest depth reading. Its centroid's horizontal movement over a short
window is the swipe.

Run:  kinect_sketchbook run sketches/swipe.py
Keys: q / Esc to quit.
"""

from collections import deque

import cv2
import numpy as np

from kinect_sketchbook import Frame, Sketch, depth_to_bgr


class Swipe(Sketch):
    title = "swipe"
    streams = ("depth",)

    HAND_SLAB_MM = 100   # depth band, from the nearest point, that counts as "hand"
    MIN_PIXELS = 300     # ignore blobs smaller than this (noise)
    WINDOW_S = 0.4       # how far back to look for motion
    MIN_DX = 150         # pixels the hand must travel within WINDOW_S
    COOLDOWN_S = 0.6     # ignore new swipes for this long after one fires

    def setup(self):
        self.trail = deque()          # (timestamp, centroid_x)
        self.mask = None
        self.centroid = None
        self.cooldown_until = 0.0
        self.label, self.label_until = "", 0.0

    def update(self, frame: Frame):
        now, depth = frame.timestamp, frame.depth

        # 1. Track: the nearest slab of depth is the hand.
        self.mask, self.centroid = None, None
        valid = depth[depth > 0]
        if valid.size:
            mask = (depth > 0) & (depth < valid.min() + self.HAND_SLAB_MM)
            ys, xs = np.nonzero(mask)
            if xs.size > self.MIN_PIXELS:
                self.mask = mask
                self.centroid = (int(xs.mean()), int(ys.mean()))
                self.trail.append((now, xs.mean()))

        # 2. Feature: horizontal travel over the last WINDOW_S seconds.
        while self.trail and now - self.trail[0][0] > self.WINDOW_S:
            self.trail.popleft()

        # 3 + 4. Recognize, with a cooldown so one swipe fires once.
        if now >= self.cooldown_until and len(self.trail) > 4:
            dx = self.trail[-1][1] - self.trail[0][1]
            if abs(dx) > self.MIN_DX:
                # Direction is in screen terms. If it's opposite your hand, run with --mirror.
                self.label = "swipe right" if dx > 0 else "swipe left"
                print(self.label, flush=True)
                self.trail.clear()
                self.cooldown_until = now + self.COOLDOWN_S
                self.label_until = now + 0.8

    def draw(self, frame: Frame):
        image = depth_to_bgr(frame.depth)
        if self.mask is not None:
            image[self.mask] = (0.4 * image[self.mask] + (0, 150, 0)).astype(np.uint8)
        if self.centroid is not None:
            cv2.circle(image, self.centroid, 8, (0, 255, 255), -1)
        if frame.timestamp < self.label_until:
            cv2.putText(image, self.label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)
        return image
