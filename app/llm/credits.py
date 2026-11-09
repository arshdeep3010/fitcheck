"""Remembers that the paid scorer's account ran out of credits, so we stop asking for a while.

Without this, every check would first spend a round trip on a 402 before falling back.
After `pause_s` we try the paid scorer once more: credits may have been topped up, and
OpenRouter's free daily quota resets at midnight UTC.
"""

import time
from collections.abc import Callable


class CreditPause:
    def __init__(self, pause_s: float = 900, clock: Callable[[], float] = time.monotonic):
        self.pause_s = pause_s
        self._clock = clock
        self._until = 0.0
        self.reason = ""

    def trip(self, reason: str) -> None:
        self._until = self._clock() + self.pause_s
        self.reason = reason

    @property
    def paused(self) -> bool:
        return self._clock() < self._until

    def seconds_left(self) -> int:
        return max(0, int(self._until - self._clock()))
