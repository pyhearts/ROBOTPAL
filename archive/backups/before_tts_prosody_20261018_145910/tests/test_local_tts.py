"""No models, network, live microphone, or audible playback are used here."""

import contextlib
import io
import json
from pathlib import Path
import queue
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import wave

import numpy as np

import main
from module import local_tts, session_log, speech, vision_assistant as vision
from module.audio_gate import PlaybackGate
from module.decision_router import CommandDecision, CommandResult
from module.action_planner import ActionPlan


def make_wav(path):
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16000)
        output.writeframes(np.full(160, 1234, dtype="<i2").tobytes())


class GateTests(unittest.TestCase):
    def test_tail_and_delayed_audio_remain_blocked(self):
        gate = PlaybackGate()
        with patch("module.audio_gate.time.monotonic", return_value=10):
            gate.begin()
        self.assertTrue(gate.accepts(9, 9.5))
        self.assertFalse(gate.accepts(9.9, 10.1))
        with patch("module.audio_gate.time.monotonic", return_value=12):
            gate.end()
        self.assertFalse(gate.accepts(11, 11.2))
        self.assertFalse(gate.accepts(12.1, 12.2))
        self.assertTrue(gate.accepts(12.31, 12.5))
        self.assertEqual(gate.epoch, 2)

    def test_queued_transcription_from_playback_never_calls_ai(self):
        gate = PlaybackGate()
        with patch("module.audio_gate.time.monotonic", return_value=5):
            gate.begin()
        with patch("module.audio_gate.time.monotonic", return_value=7):
            gate.end()
        jobs = queue.Queue()
        jobs.put(speech.TranscriptionTask("echo", -1, np.ones(1600), True, 6))
        jobs.put(speech.TranscriptionTask("user", -1, np.ones(1600), True, 8))
        jobs.put(speech._STOP)
        callback = Mock()
        with patch.object(speech, "transcribe_utterance", return_value="안녕") as transcribe, \
                patch.object(speech.time, "monotonic", return_value=8), \
                contextlib.redirect_stdout(io.StringIO()):
            speech._transcription_worker(object(), None, jobs, callback, gate)
        self.assertEqual(transcribe.call_count, 1)
        callback.assert_called_once_with("음성 입력", "안녕")
        self.assertEqual(jobs.unfinished_tasks, 0)

    def test_detector_reset_drops_partial_speech_and_preroll(self):
        detector = speech.MonoUtteranceDetector()
        detector.process(np.full(6400, .2), 0)
        detector.reset()
        self.assertEqual(detector.flush(), [])
        self.assertEqual(detector.process(np.zeros(16000), 1), [])

    def test_silero_reset_clears_recurrent_state(self):
        from module.speech_quality import SileroProbability
        scorer = SileroProbability()
        scorer.h.fill(1)
        scorer.c.fill(1)
        scorer.context.fill(1)
        scorer.reset()
        self.assertEqual(float(scorer.h.sum() + scorer.c.sum() + scorer.context.sum()), 0)

    def test_microphone_drops_playback_and_resumes_after_tail(self):
        gate = PlaybackGate()
        clock = [10.0]
        class Captured(queue.Queue):
            def get(self, block=True, timeout=None):
                try:
                    return super().get(False)
                except queue.Empty:
                    if block:
                        raise KeyboardInterrupt
                    raise
        class Probability:
            def __call__(self, frame):
                return float(np.mean(frame) > .05)
            def reset(self):
                pass
        class Microphone:
            def __init__(self, **kwargs):
                self.callback = kwargs["callback"]
            def __enter__(self):
                def send(seconds, level):
                    samples = np.full((round(seconds * 16000), 1), level, dtype="float32")
                    self.callback(samples, len(samples), None, None)
                send(.4, .2)  # old partial before playback must not merge with new speech
                clock[0] = 11
                gate.begin()
                clock[0] = 12
                send(.4, .3)  # speaker echo is excluded in capture callback
                clock[0] = 13
                gate.end()
                clock[0] = 14
                send(.4, .4)
                clock[0] = 14.8
                send(.8, 0)
                return self
            def __exit__(self, *args):
                pass
        chunks = []
        callback = Mock()
        captured, jobs = Captured(), queue.Queue()
        def transcribe(model, audio):
            chunks.append(audio.copy())
            return "새 발화"
        with patch.object(speech, "_load_transcription_dependencies", return_value=(SimpleNamespace(InputStream=Microphone), Mock())), \
                patch.object(speech, "prepare_microphone"), patch.object(speech, "SileroProbability", return_value=Probability()), \
                patch.object(speech, "_load_whisper_model", return_value=object()), \
                patch.object(speech.queue, "Queue", side_effect=[captured, jobs]), \
                patch.object(speech, "transcribe_utterance", side_effect=transcribe), \
                patch.object(speech.time, "monotonic", side_effect=lambda: clock[0]), \
                contextlib.redirect_stdout(io.StringIO()):
            speech.run_live_transcription(callback, options=speech.SpeechOptions(live_preview=False), playback_gate=gate)
        callback.assert_called_once_with("음성 입력", "새 발화")
        self.assertEqual(len(chunks), 1)
        self.assertFalse(np.any(chunks[0] == np.float32(.3)))
        self.assertFalse(np.any(chunks[0] == np.float32(.2)))
        self.assertTrue(np.any(chunks[0] == np.float32(.4)))


class SelectionTests(unittest.TestCase):
    def test_result_selection_does_not_claim_execution(self):
        result = CommandResult(CommandDecision("concrete_command", 1), plan=ActionPlan(
            status="plan", summary="앞으로 30cm 이동", steps=["전진", "정지"], error=""))
        self.assertEqual(local_tts.spoken_text(result), "행동 계획입니다. 앞으로 30cm 이동")
        result = CommandResult(CommandDecision("stop_command", 1), plan=result.plan)
        self.assertEqual(local_tts.spoken_text(result), "정지 계획을 접수했습니다.")
        result = CommandResult(CommandDecision("not_command", 1), message="고정 안내")
        self.assertEqual(local_tts.spoken_text(result), "")

    def test_long_reply_uses_complete_sentences(self):
        text = "첫 번째 문장입니다. " + "긴 문장입니다. " * 70
        result = CommandResult(CommandDecision("conversation", 1), message=text)
        spoken = local_tts.spoken_text(result)
        self.assertLessEqual(len(spoken), 300)
        self.assertTrue(spoken.endswith("."))
        self.assertTrue(text.startswith(spoken))
        result = CommandResult(CommandDecision("conversation", 1), message="가" * 301)
        self.assertIn("화면", local_tts.spoken_text(result))

    def test_recorder_audio_import_preserves_original(self):
        import av
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            reference = root / "recording.m4a"
            with av.open(str(reference), "w") as container:
                stream = container.add_stream("aac", rate=24000)
                stream.layout = "mono"
                samples = (np.sin(np.arange(6 * 24000) * .1) * .1).astype("float32").reshape(1, -1)
                frame = av.AudioFrame.from_ndarray(samples, format="fltp", layout="mono")
                frame.sample_rate = 24000
                for packet in stream.encode(frame):
                    container.mux(packet)
                for packet in stream.encode(None):
                    container.mux(packet)
            before = reference.read_bytes()
            result = local_tts.import_reference(reference, root / "imports")
            self.assertEqual(reference.read_bytes(), before)
            duration, rate = local_tts.wav_info(result)
            self.assertAlmostEqual(duration, 6, delta=.1)
            self.assertEqual(rate, 24000)

    def test_two_references_use_equal_excerpts_bounded_to_fifty_seconds(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            originals = []
            for index, seconds in enumerate([35, 57]):
                path = root / f"reference{index}.wav"
                with wave.open(str(path), "wb") as wav:
                    wav.setnchannels(1)
                    wav.setsampwidth(2)
                    wav.setframerate(24000)
                    wav.writeframes(np.full(seconds * 24000, 1000 + index, dtype="<i2").tobytes())
                originals.append(path)
            before = [p.read_bytes() for p in originals]
            combined = local_tts.combine_references(originals, root / "imports")
            self.assertEqual(local_tts.wav_info(combined)[0], 50)
            self.assertEqual([p.read_bytes() for p in originals], before)
            with wave.open(str(combined), "rb") as wav:
                audio = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2")
            self.assertTrue(np.all(audio[:25 * 24000] == 1000))
            self.assertTrue(np.all(audio[25 * 24000:] == 1001))


class PlaybackTests(unittest.TestCase):
    def create_service(self, root, worker=None):
        voice = root / "voice.json"
        voice.write_text(json.dumps({"voice_version": "voice1", "model_version": "model1"}), encoding="utf-8")
        options = local_tts.TTSOptions(voice=voice, cache=root / "cache")
        worker = worker or Mock()
        if worker.request.side_effect is None:
            worker.request.side_effect = lambda action, **data: make_wav(Path(data["output"]))
        sound = Mock()
        tts = local_tts.LocalTTS(options, worker=worker, sound=sound)
        self.addCleanup(tts.close)
        return tts, worker, sound

    def test_cache_audio_log_and_playback_gate(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            root = Path(folder)
            tts, worker, sound = self.create_service(root)
            checks = []
            sound.play.side_effect = lambda *a, **kw: checks.append(tts.gate.accepts(0, float("inf")))
            with session_log.capture_output(root / "logs") as log:
                tts.start()
                tts.submit("안녕하세요.", "first-command")
                tts.wait()
                tts.submit("안녕하세요.", "second-command")
                tts.wait()
                tts.close()
            self.assertEqual(worker.request.call_count, 1)
            self.assertEqual(sound.play.call_count, 2)
            self.assertEqual(checks, [False, False])
            events = [json.loads(line) for line in log.events_path.read_text(encoding="utf-8").splitlines()]
            ready = [e for e in events if e["event"] == "tts_audio_ready"]
            self.assertEqual([e["cache_hit"] for e in ready], [False, True])
            self.assertEqual([e["command_id"] for e in ready], ["first-command", "second-command"])
            self.assertEqual(len(list((root / "logs").glob("*_audio/*.wav"))), 2)
            for event in ready:
                self.assertTrue((root / "logs" / event["audio_path"]).is_file())

    def test_failed_job_does_not_stop_next_job(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            tts, worker, sound = self.create_service(Path(folder))
            calls = []
            def synth(action, **data):
                calls.append(data["text"])
                if data["text"] == "실패":
                    raise RuntimeError("timeout")
                make_wav(Path(data["output"]))
            worker.request.side_effect = synth
            tts.start()
            tts.submit("실패", "1")
            tts.submit("성공", "2")
            with self.assertRaisesRegex(RuntimeError, "1개"):
                tts.wait()
            self.assertEqual(calls, ["실패", "성공"])
            sound.play.assert_called_once()

    def test_queue_is_bounded_and_close_discards_pending(self):
        with tempfile.TemporaryDirectory() as folder:
            tts, worker, sound = self.create_service(Path(folder))
            self.assertTrue(all(tts.submit(str(i), str(i)) for i in range(4)))
            self.assertFalse(tts.submit("overflow", "5"))
            tts.close()
            self.assertEqual(tts._jobs.unfinished_tasks, 0)
            self.assertFalse(tts.submit("closed", "6"))

    def test_corrupt_cache_is_regenerated_and_voice_speed_change_cache_key(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            tts, worker, sound = self.create_service(Path(folder))
            tts.start()
            path, cached, _ = tts.synthesize("시험")
            self.assertFalse(cached)
            path.write_bytes(b"bad")
            self.assertFalse(tts.synthesize("시험")[1])
            tts._voice_version = "voice2"
            other = tts.synthesize("시험")[0]
            self.assertNotEqual(path, other)
            self.assertEqual(worker.request.call_count, 3)

    def test_replay_failure_releases_gate(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            tts, worker, sound = self.create_service(Path(folder))
            sound.play.side_effect = RuntimeError("device disconnected")
            tts.start()
            tts.submit("시험", "id")
            with self.assertRaises(RuntimeError):
                tts.wait()
            self.assertEqual(tts.gate.epoch, 2)
            self.assertTrue(tts.gate.accepts(1e30, 1e31))

    def test_missing_voice_has_actionable_error(self):
        with tempfile.TemporaryDirectory() as folder:
            tts = local_tts.LocalTTS(local_tts.TTSOptions(voice=Path(folder) / "missing.json"))
            with self.assertRaisesRegex(RuntimeError, "prepare-tts"):
                tts.start()


class CLITests(unittest.TestCase):
    def test_tts_restricted_to_supported_modes_and_prepare_needs_reference(self):
        for args in [["--tts"], ["--prepare-tts"], ["--tts-reference", "voice.wav"], ["--output-device", "1"]]:
            with self.subTest(args=args), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    main._run_main(args)

    def test_assist_options_are_forwarded(self):
        with patch.object(main, "run_voice_vision_assistant") as run:
            self.assertEqual(main._run_main(["--assist", "--tts", "--output-device", "1"]), 0)
        self.assertEqual(run.call_args.kwargs["tts_options"].output_device, 1)

    def test_tts_test_uses_no_ai(self):
        with patch.object(local_tts, "LocalTTS") as create, patch.object(main, "process_command") as ai:
            self.assertEqual(main._run_main(["--test-tts", "안녕하세요."]), 0)
        create.return_value.start.assert_called_once()
        create.return_value.wait.assert_called_once()
        ai.assert_not_called()

    def test_plan_and_audio_share_command_id(self):
        result = CommandResult(CommandDecision("conversation", 1), message="안녕하세요.")
        with patch.object(local_tts, "LocalTTS") as tts, \
                patch.object(main, "process_command", return_value=result) as process, \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main._run_main(["--plan", "--command", "안녕", "--tts"]), 0)
        command_id = process.call_args.kwargs["command_id"]
        tts.return_value.submit_result.assert_called_once_with(result, command_id)


class WorkerProtocolTests(unittest.TestCase):
    def test_worker_explicitly_blocks_network(self):
        from module import local_tts_worker
        import socket
        original = (socket.socket.connect, socket.socket.connect_ex, socket.create_connection)
        try:
            local_tts_worker.deny_network()
            with self.assertRaisesRegex(OSError, "네트워크"):
                socket.create_connection(("example.invalid", 443))
        finally:
            socket.socket.connect, socket.socket.connect_ex, socket.create_connection = original

    def test_real_subprocess_protocol_and_timeout_cleanup(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "manifest.json").write_text("{}", encoding="utf-8")
            (root / "local_tts_worker.py").write_text('''import json,sys,time,os
print(json.dumps({"ready": True}), flush=True)
for line in sys.stdin:
    item=json.loads(line)
    if item["action"] == "hang":
        time.sleep(10)
    print(json.dumps({"ok": True,"result": {"text": item.get("text"),"has_key": bool(os.getenv("OPENAI_API_KEY"))}}), flush=True)
''', encoding="utf-8")
            options = local_tts.TTSOptions(python=Path(sys.executable), models=root, timeout=.5)
            worker = local_tts.WorkerClient(options)
            try:
                with patch.object(local_tts, "__file__", str(root / "caller.py")), \
                        patch.dict("os.environ", {"OPENAI_API_KEY": "must-not-leak"}):
                    response = worker.request("echo", text="한국어")
                    self.assertEqual(response, {"text": "한국어", "has_key": False})
                    with self.assertRaisesRegex(RuntimeError, "제한 시간"):
                        worker.request("hang")
                    self.assertIsNone(worker.process)
                    # A later request starts a fresh worker instead of retrying the failed job.
                    self.assertEqual(worker.request("echo", text="다음 요청")["text"], "다음 요청")
            finally:
                worker.close()


if __name__ == "__main__":
    unittest.main()
