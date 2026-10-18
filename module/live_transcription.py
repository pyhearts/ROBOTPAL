"""A final-first work queue with a single replaceable local preview slot."""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PreviewTask:
    utterance_id: str
    audio: np.ndarray
    end_time: float


class LiveTranscriptionQueue:
    """Keep all final/chunk jobs; replace previews instead of building a backlog.

    Inference is serialized on one model. An already-running decoder finishes its
    bounded generation, but its stale preview is suppressed when a final arrives.
    """

    def __init__(self, stop_token: object) -> None:
        self.stop_token = stop_token
        self._condition = threading.Condition()
        self._finals: deque[Any] = deque()
        self._preview: PreviewTask | None = None
        self._active_id: str | None = None
        self._closed = False
        self._busy_final = False
        self.unfinished_tasks = 0

    def _discard_preview(self) -> None:
        if self._preview is not None:
            self._preview = None
            self.unfinished_tasks -= 1

    def activate(self, utterance_id: str | None) -> None:
        with self._condition:
            if self._active_id != utterance_id:
                self._discard_preview()
            self._active_id = utterance_id
            self._condition.notify_all()

    def offer_preview(self, preview: PreviewTask) -> bool:
        with self._condition:
            if self._closed or preview.utterance_id != self._active_id or self._finals or self._busy_final:
                return False
            if self._preview is None:
                self.unfinished_tasks += 1
            self._preview = preview
            self._condition.notify_all()
            return True

    def preview_is_current(self, preview: PreviewTask) -> bool:
        with self._condition:
            return not self._closed and not self._finals and preview.utterance_id == self._active_id

    def put(self, task: Any) -> None:
        with self._condition:
            if task is self.stop_token:
                if not self._closed:
                    self._closed = True
                    self._active_id = None
                    self._discard_preview()
                    self.unfinished_tasks += 1
            else:
                if self._closed:
                    raise RuntimeError("종료된 전사 큐에는 작업을 추가할 수 없습니다.")
                self._finals.append(task)
                self.unfinished_tasks += 1
                self._discard_preview()
                if task.final and self._active_id == task.utterance_id:
                    self._active_id = None
            self._condition.notify_all()

    def get(self) -> Any:
        with self._condition:
            while True:
                if self._finals:
                    self._busy_final = True
                    return self._finals.popleft()
                if self._preview is not None:
                    preview, self._preview = self._preview, None
                    return preview
                if self._closed:
                    return self.stop_token
                self._condition.wait()

    def task_done(self) -> None:
        with self._condition:
            if self.unfinished_tasks <= 0:
                raise ValueError("전사 작업 완료 횟수가 올바르지 않습니다.")
            self.unfinished_tasks -= 1
            self._busy_final = False
            self._condition.notify_all()

    def join(self) -> None:
        with self._condition:
            while self.unfinished_tasks:
                self._condition.wait()
