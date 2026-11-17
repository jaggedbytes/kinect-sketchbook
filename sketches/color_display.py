"""Color Display: shows live color frames in a resizable window.

Run:  sketchbook run sketches/color_display.py
Keys: q / Esc to quit, double-click to toggle full screen.
"""

import cv2

from kinect_sketchbook import Sketch


class ColorDisplay(Sketch):
    """Displays color frames with a resizable window."""

    title = "color display"
    streams = ("color",)

    def draw(self, frame):
        # Convert BGR to RGB for proper color display
        if frame.color is not None:
            image = cv2.cvtColor(frame.color, cv2.COLOR_BGR2RGB)
        else:
            # Fallback to depth visualization if no color stream
            from kinect_sketchbook import depth_to_bgr
            image = depth_to_bgr(frame.depth)

        # Add frame count for reference
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, f"Frame: {frame.index}", (10, 30),
                    font, 0.7, (255, 255, 255), 1)

        return image
