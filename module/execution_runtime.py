"""Interactive text / final-transcription input for virtual execution only."""
from __future__ import annotations

import contextlib
import signal
import threading

from .decision_router import print_command_result
from .executable_plan import ExecutablePlan, is_direct_stop, process_execution_command
from .execution import ExecutionManager
from .session_log import command_context, record_event


def console_event(event):
    name = event["event"]
    if name == "device" and event["device_event"].get("event") == "gait":
        d = event["device_event"]
        print(f"[가상 보행] {d['phase']} / A={d['groups']['A']}, B={d['groups']['B']} / {d['intent']}", flush=True)
    elif name in {"state", "expired", "rejected", "error", "completed", "cancelled", "stop_completed"}:
        labels = {"expired": "요청 유효 시간 10초 초과: 실행 취소", "completed": "가상 계획 완료",
                  "stop_completed": "착지 동작 완료 (접촉 확인 아님)", "cancelled": "계획 취소",
                  "rejected": "실행 보류", "error": "실행 오류"}
        print(f"[가상 실행] {event.get('label', labels.get(name, name))} "
              f"{event.get('reason', event.get('error', ''))}", flush=True)


class ExecutionRuntime:
    """Two bounded inference workers, one latest pending input, one device loop.

    Execution input is conservative: any new nonempty request first cancels the
    previous plan. If classified as conversation or invalid, the robot stays stopped.
    """
    def __init__(self, *, manager=None, processor=process_execution_command):
        self.manager = manager or ExecutionManager(emit=console_event)
        self.processor = processor
        self.condition = threading.Condition(threading.RLock())
        self.pending = None
        self.closed = False
        self.shutdown = threading.Event()
        self.workers = []
        for number in range(2):
            worker = threading.Thread(target=self._plan, name=f"execution-planner-{number}", daemon=True)
            worker.start()
            self.workers.append(worker)
        self.loop = threading.Thread(target=self._tick, name="virtual-controller", daemon=True)
        self.loop.start()

    def submit(self, text):
        text = text.strip()
        if not text:
            return None
        with self.condition:
            if self.closed:
                return None
            if is_direct_stop(text):
                self.pending = None
                self.manager.stop()
                self.condition.notify_all()
                return None
            request = self.manager.begin()
            self.pending = (request, text)
            record_event("execution_input", command_id=request.command_id, text=text, simulation=True)
            self.condition.notify_all()
            return request

    def on_transcription(self, speaker, text):
        self.submit(text)

    def stop(self, reason="키보드 정지"):
        with self.condition:
            self.pending = None
            self.manager.stop(reason)
            self.condition.notify_all()

    def _plan(self):
        while True:
            with self.condition:
                self.condition.wait_for(lambda: self.closed or self.pending is not None)
                if self.closed:
                    return
                request, text = self.pending
                self.pending = None
                if (not self.manager.is_current(request)
                        or self.manager.clock() - request.received_at >= 10):
                    continue
            try:
                with command_context(request.command_id):
                    result = self.processor(text, command_id=request.command_id)
                with self.condition:
                    if not self.manager.is_current(request) or not self.manager.awaiting:
                        continue
                    print_command_result(result)
                    if isinstance(result.plan, ExecutablePlan):
                        self.manager.offer(request, result.plan)
                    else:
                        self.manager.reject(request, result.message or "실행 가능한 계획이 없습니다.")
                    self.condition.notify_all()
            except Exception as exc:
                with self.condition:
                    if not self.closed:
                        self.manager.reject(request, str(exc))
                        self.condition.notify_all()

    def _tick(self):
        while not self.shutdown.wait(0.02):
            with self.condition:
                self.manager.tick()
                self.condition.notify_all()

    def wait(self, timeout=35):
        with self.condition:
            return self.condition.wait_for(lambda: self.manager.settled or self.manager.state == "fault", timeout)

    def close(self):
        with self.condition:
            if self.shutdown.is_set():
                return
            self.closed = True
            self.pending = None
            self.manager.close()
            self.condition.notify_all()
            # Keep the logical device loop alive through its final landing.
            self.condition.wait_for(lambda: self.manager.device.idle, timeout=2)
        self.shutdown.set()
        self.loop.join(timeout=1)
        # In-flight API calls own their clients and are bounded by API timeouts.
        # They cannot publish plans after closed; do not block shutdown on inference.


@contextlib.contextmanager
def immediate_interrupt_stop(runtime):
    """Request stop before speech's Ctrl+C handler drains final transcription."""
    previous = signal.getsignal(signal.SIGINT)
    def interrupt(signum, frame):
        runtime.stop("Ctrl+C 정지")
        # Invalidate all subsequent final speech callbacks during shutdown.
        with runtime.condition:
            runtime.closed = True
            runtime.condition.notify_all()
        if callable(previous):
            previous(signum, frame)
        else:
            raise KeyboardInterrupt
    signal.signal(signal.SIGINT, interrupt)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, previous)


def run_execution(*, command=None, voice=False, speech_kwargs=None):
    if voice:
        from .ai_api import create_api_client
        client = create_api_client()  # Fail before opening the microphone.
        client.close()
    runtime = ExecutionRuntime()
    print("가상 실행 전용입니다. 새 입력은 기존 계획을 취소합니다. 실제 관절·모터는 제어하지 않습니다.", flush=True)
    keyboard_done = threading.Event()
    keyboard = None
    try:
        with immediate_interrupt_stop(runtime):
            if voice:
                from .speech import run_live_transcription
                # Windows console keyboard stop, independent of ASR and API workers.
                def keyboard_stop():
                    try:
                        import msvcrt
                    except ImportError:
                        return
                    while not keyboard_done.wait(0.05):
                        if msvcrt.kbhit() and msvcrt.getwch() == " ":
                            runtime.stop()
                keyboard = threading.Thread(target=keyboard_stop, name="keyboard-stop", daemon=True)
                keyboard.start()
                print("음성 실행: '멈춰/정지/그만', 키보드 Space 또는 Ctrl+C로 정지합니다.", flush=True)
                run_live_transcription(on_transcription=runtime.on_transcription,
                                       identify_speakers=False, **(speech_kwargs or {}))
            elif command is not None:
                runtime.submit(command)
                if not runtime.wait():
                    raise RuntimeError("가상 실행 대기 시간이 초과되었습니다.")
            else:
                print("명령을 입력하세요. 정지: 멈춰/정지/그만, 종료: /quit 또는 Ctrl+C", flush=True)
                while True:
                    text = input("> ")
                    if text.strip() == "/quit":
                        break
                    runtime.submit(text)
        return 1 if runtime.manager.state == "fault" or runtime.manager.outcome in {"rejected", "expired"} else 0
    finally:
        keyboard_done.set()
        if keyboard:
            keyboard.join(timeout=1)
        runtime.close()
