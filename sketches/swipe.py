"""Swipe: hold a hand out in front of you and swipe left or right.

The hand is "whatever is closest to the camera" (tracking.nearest_hand):
everything within HAND_SLAB_MM of the nearest depth reading. Its centroid's
horizontal movement over a short window is the swipe (tracking.SwipeDetector).

Run:  sketchbook run sketches/swipe.py
Keys: q / Esc to quit.
"""

import cv2
import numpy as np

from kinect_sketchbook import Frame, Sketch, depth_to_bgr
from kinect_sketchbook.tracking import SwipeDetector, nearest_hand


class Swipe(Sketch):
    title = "swipe"
    streams = ("depth",)

    HAND_SLAB_MM = 100   # depth band, from the nearest point, that counts as "hand"
    MIN_PIXELS = 300     # ignore blobs smaller than this (noise)
    WINDOW_S = 0.4       # how far back to look for motion
    MIN_DX = 150         # pixels the hand must travel within WINDOW_S
    COOLDOWN_S = 0.6     # ignore new swipes for this long after one fires

    def setup(self):
        self.swipes = SwipeDetector(self.WINDOW_S, self.MIN_DX, self.COOLDOWN_S)
        self.mask = None
        self.centroid = None
        self.label, self.label_until = "", 0.0

    def update(self, frame: Frame):
        # 1. Track: the nearest slab of depth is the hand.
        hand = nearest_hand(frame.depth, self.HAND_SLAB_MM, self.MIN_PIXELS)
        self.mask = hand.mask if hand else None
        self.centroid = (int(hand.centroid[0]), int(hand.centroid[1])) if hand else None

        # 2. Recognize: horizontal travel, with a cooldown so one swipe fires once.
        direction = self.swipes.feed(frame.timestamp, hand.centroid[0] if hand else None)
        if direction:
            # Direction is in screen terms. If it's opposite your hand, run with --mirror.
            self.label = f"swipe {direction}"
            print(self.label, flush=True)
            self.label_until = frame.timestamp + 0.8

    def draw(self, frame: Frame):
        image = depth_to_bgr(frame.depth)
        if self.mask is not None:
            image[self.mask] = (0.4 * image[self.mask] + (0, 150, 0)).astype(np.uint8)
        if self.centroid is not None:
            cv2.circle(image, self.centroid, 8, (0, 255, 255), -1)
        if frame.timestamp < self.label_until:
            cv2.putText(image, self.label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)
        return image
