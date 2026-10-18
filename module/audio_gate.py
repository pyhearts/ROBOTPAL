"""Mark playback intervals so buffered microphone audio cannot become commands."""

from __future__ import annotations

import threading
import time


class PlaybackGate:
    def __init__(self, tail_seconds: float = 0.3) -> None:
        self.tail_seconds = tail_seconds
        self._lock = threading.RLock()
        self._intervals: list[list[float]] = []
        self._epoch = 0

    @property
    def epoch(self) -> int:
        with self._lock:
            return self._epoch

    def begin(self) -> None:
        with self._lock:
            self._epoch += 1
            self._intervals.append([time.monotonic(), float("inf")])

    def end(self) -> None:
        with self._lock:
            if self._intervals:
                self._intervals[-1][1] = time.monotonic() + self.tail_seconds
            self._epoch += 1

    def accepts(self, start: float, end: float) -> bool:
        with self._lock:
            return not any(start < stop and end > begin for begin, stop in self._intervals)
