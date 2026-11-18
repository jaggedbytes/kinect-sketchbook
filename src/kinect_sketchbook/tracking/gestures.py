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


class WaveDetector:
    """Fires once when x travels back and forth.

    A wave is min_reversals legs of at least min_dx, in alternating directions,
    all inside window_s. One reversal is a reach that comes back; two is a wave.
    A jump larger than jump_max clears the trail, so a swapped track doesn't
    look like a gesture.

    window_s:       how far back to look for the wave.
    min_dx:         how far x must travel in one direction before it counts.
    min_reversals:  direction changes required. 2 is out, back, and out again.
    cooldown_s:     ignore new waves for this long after one fires.
    jump_max:       a step larger than this starts the trail over.
    """

    def __init__(
        self,
        window_s: float = 1.2,
        min_dx: float = 60,
        min_reversals: int = 2,
        cooldown_s: float = 0.8,
        jump_max: float = 400,
    ) -> None:
        self.window_s = window_s
        self.min_dx = min_dx
        self.min_reversals = min_reversals
        self.cooldown_s = cooldown_s
        self.jump_max = jump_max
        self.trail: deque[tuple[float, float]] = deque()  # (timestamp, x)
        self.cooldown_until = 0.0

    def reset(self) -> None:
        self.trail.clear()
        self.cooldown_until = 0.0

    def feed(self, now: float, x: float | None) -> bool:
        """Call every frame, with x = None when nothing is tracked.

        Returns True on the frame a wave completes, otherwise False.
        """
        if x is not None:
            if self.trail and abs(x - self.trail[-1][1]) > self.jump_max:
                self.trail.clear()
            self.trail.append((now, x))
        while self.trail and now - self.trail[0][0] > self.window_s:
            self.trail.popleft()

        if now < self.cooldown_until or len(self.trail) < self.min_reversals + 2:
            return False
        if self._reversals() < self.min_reversals:
            return False
        self.trail.clear()
        self.cooldown_until = now + self.cooldown_s
        return True

    def _reversals(self) -> int:
        reversals = 0
        pivot = self.trail[0][1]
        direction = 0
        for _, x in self.trail:
            dx = x - pivot
            if abs(dx) < self.min_dx:
                continue
            sign = 1 if dx > 0 else -1
            if direction != 0 and sign != direction:
                reversals += 1
            direction = sign
            pivot = x
        return reversals
