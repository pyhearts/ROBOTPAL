"""Capture console output and structured results in durable, per-run UTF-8 logs."""

from __future__ import annotations

import contextlib
import json
import os
import re
import sys
import threading
import traceback
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Iterator

LOG_ROOT = Path(__file__).resolve().parent.parent / "logs"
KST = timezone(timedelta(hours=9))
_active: SessionLog | None = None
_context = threading.local()


def _now() -> str:
    return datetime.now(KST).isoformat(timespec="milliseconds")


def as_json(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return value


class SessionLog:
    def __init__(self, root: Path) -> None:
        root.mkdir(parents=True, exist_ok=True)
        name = datetime.now(KST).strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
        self.console_path = root / f"{name}.log"
        self.events_path = root / f"{name}.jsonl"
        self._console = self.console_path.open("x", encoding="utf-8")
        try:
            self._events = self.events_path.open("x", encoding="utf-8")
        except BaseException:
            self._console.close()
            raise
        self._lock = threading.RLock()
        self._pending: dict[tuple[int, str], str] = {}
        self._key = os.getenv("OPENAI_API_KEY", "")

    def _redact(self, value: Any) -> Any:
        if isinstance(value, str):
            if self._key:
                value = value.replace(self._key, "[API_KEY]")
            return re.sub(r"\bsk-[A-Za-z0-9_-]{8,}", "[API_KEY]", value)
        if isinstance(value, dict):
            return {str(k): self._redact(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._redact(v) for v in value]
        return value

    def event(self, event: str, **data: Any) -> None:
        with self._lock:
            item = {"timestamp": _now(), "event": event,
                    "thread": threading.current_thread().name, **data}
            command_id = getattr(_context, "command_id", None)
            if command_id is not None:
                item["command_id"] = command_id
            self._events.write(json.dumps(self._redact(item), ensure_ascii=False, default=str) + "\n")
            self._events.flush()

    def _line(self, stream: str, text: str) -> None:
        self._console.write(f"{_now()} [{stream}] {self._redact(text)}\n")
        self._console.flush()
        self.event("output", stream=stream, text=text)

    def output(self, stream: str, text: str) -> None:
        with self._lock:
            key = (threading.get_ident(), stream)
            pending = self._pending.get(key, "") + text
            while "\n" in pending:
                line, pending = pending.split("\n", 1)
                self._line(stream, line)
            self._pending[key] = pending

    def flush(self, stream: str | None = None) -> None:
        with self._lock:
            for key, pending in list(self._pending.items()):
                if pending and (stream is None or key[1] == stream):
                    self._line(key[1], pending)
                    self._pending[key] = ""

    def close(self) -> None:
        self.flush()
        self.event("session_end")
        self._console.close()
        self._events.close()


class _Tee:
    def __init__(self, original: Any, session: SessionLog, name: str) -> None:
        self.original, self.session, self.name = original, session, name

    def write(self, text: str) -> int:
        self.session.output(self.name, text)
        return self.original.write(text)

    def flush(self) -> None:
        self.session.flush(self.name)
        self.original.flush()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.original, name)


def record_event(event: str, **data: Any) -> None:
    if _active is not None:
        _active.event(event, **data)


@contextlib.contextmanager
def command_context(command_id: str) -> Iterator[None]:
    previous = getattr(_context, "command_id", None)
    _context.command_id = command_id
    try:
        yield
    finally:
        _context.command_id = previous


@contextlib.contextmanager
def capture_output(root: Path | None = None) -> Iterator[SessionLog]:
    """Wrap a CLI run; nested calls reuse the same session and restore streams."""
    global _active
    if _active is not None:
        yield _active
        return
    session = SessionLog(LOG_ROOT if root is None else root)
    stdout, stderr = sys.stdout, sys.stderr
    _active = session
    sys.stdout, sys.stderr = _Tee(stdout, session, "stdout"), _Tee(stderr, session, "stderr")
    try:
        session.event("session_start")
        print(f"실행 로그: {session.console_path}", flush=True)
        print(f"결과 기록: {session.events_path}", flush=True)
        yield session
    except BaseException as exc:
        if not isinstance(exc, (SystemExit, KeyboardInterrupt)):
            traceback.print_exception(type(exc), exc, exc.__traceback__, file=sys.stderr)
        raise
    finally:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        finally:
            sys.stdout, sys.stderr = stdout, stderr
            _active = None
            session.close()
