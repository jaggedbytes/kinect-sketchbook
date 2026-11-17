"""Gestures: movement over time turned into events.

Detectors take plain positions, not Frames, so they work with any tracker
(a depth blob's centroid, or a pose model's wrist).
"""

from __future__ import annotations

from collections import deque


class SwipeDetector:
    """Fires once per fast horizontal movement.

    window_s:   how far back to look for motion.
    min_dx:     distance x must travel within window_s, in whatever units are fed in.
    cooldown_s: ignore new swipes for this long after one fires.
    """

    def __init__(self, window_s: float = 0.4, min_dx: float = 150, cooldown_s: float = 0.6) -> None:
        self.window_s = window_s
        self.min_dx = min_dx
        self.cooldown_s = cooldown_s
        self.trail: deque[tuple[float, float]] = deque()  # (timestamp, x)
        self.cooldown_until = 0.0

    def feed(self, now: float, x: float | None) -> str | None:
        """Call every frame, with x = None when nothing is tracked.

        Returns "right" when x increased fast enough, "left" when it decreased,
        otherwise None.
        """
        if x is not None:
            self.trail.append((now, x))
        while self.trail and now - self.trail[0][0] > self.window_s:
            self.trail.popleft()

        if now < self.cooldown_until or len(self.trail) <= 4:
            return None
        dx = self.trail[-1][1] - self.trail[0][1]
        if abs(dx) <= self.min_dx:
            return None
        self.trail.clear()
        self.cooldown_until = now + self.cooldown_s
        return "right" if dx > 0 else "left"
