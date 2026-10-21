"""Bounded previews, final-priority scheduling, and no provisional AI commands."""

import contextlib
import io
import queue
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from module import cli, speech, transcription, utterance
from module import speech_quality as quality
from module.live_transcription import LiveTranscriptionQueue, PreviewTask


def preview(identity="a", text_value=0.1):
    return PreviewTask(identity, np.full(16000, text_value, dtype=np.float32), 1)


def final(identity="a", text_value=0.2):
    return utterance.TranscriptionTask(identity, -1, np.full(16000, text_value, dtype=np.float32), True, 1)


def result(text="임시 문장", accepted=True):
    return quality.TranscriptionResult(text, accepted, "" if accepted else "low_log_probability", {})


class QueueTests(unittest.TestCase):
    def test_preview_slot_replaces_old_tasks_without_accumulation(self):
        jobs = LiveTranscriptionQueue(speech._STOP)
        jobs.activate("a")
        for number in range(100):
            self.assertTrue(jobs.offer_preview(preview(text_value=number)))
        self.assertEqual(jobs.unfinished_tasks, 1)
        item = jobs.get()
        self.assertEqual(item.audio[0], 99)
        jobs.task_done()
        jobs.put(speech._STOP)
        self.assertIs(jobs.get(), speech._STOP)
        jobs.task_done()
        jobs.join()

    def test_final_wins_and_pending_preview_is_removed(self):
        jobs = LiveTranscriptionQueue(speech._STOP)
        jobs.activate("a")
        jobs.offer_preview(preview())
        task = final()
        jobs.put(task)
        self.assertFalse(jobs.offer_preview(preview()))
        self.assertEqual(jobs.unfinished_tasks, 1)
        self.assertIs(jobs.get(), task)
        jobs.task_done()

    def test_final_chunks_keep_fifo_and_previews_are_skipped_while_backlogged(self):
        jobs = LiveTranscriptionQueue(speech._STOP)
        first, second = final("first"), final("second")
        jobs.put(first)
        jobs.put(second)
        jobs.activate("new")
        self.assertFalse(jobs.offer_preview(preview("new")))
        self.assertIs(jobs.get(), first)
        jobs.task_done()
        self.assertIs(jobs.get(), second)
        jobs.task_done()

    def test_close_discards_preview_and_drains_final_before_stop(self):
        jobs = LiveTranscriptionQueue(speech._STOP)
        task = final()
        jobs.put(task)
        jobs.activate("b")
        jobs.put(speech._STOP)
        jobs.put(speech._STOP)  # Idempotent close.
        self.assertIs(jobs.get(), task)
        jobs.task_done()
        self.assertIs(jobs.get(), speech._STOP)
        jobs.task_done()
        self.assertEqual(jobs.unfinished_tasks, 0)
        with self.assertRaises(RuntimeError):
            jobs.put(final())

    def test_no_preview_is_admitted_while_final_inference_is_running(self):
        jobs = LiveTranscriptionQueue(speech._STOP)
        jobs.put(final("old"))
        jobs.activate("new")
        jobs.get()
        self.assertFalse(jobs.offer_preview(preview("new")))
        jobs.task_done()
        self.assertTrue(jobs.offer_preview(preview("new")))


class WorkerTests(unittest.TestCase):
    def test_live_microphone_path_displays_preview_before_final_callback(self):
        displayed, finished = threading.Event(), threading.Event()
        callbacks, observations = [], []

        class CapturedQueue(queue.Queue):
            def get(self, block=True, timeout=None):
                if not block:
                    return super().get(False)
                while True:
                    try:
                        return super().get(timeout=0.02)
                    except queue.Empty:
                        if finished.is_set():
                            raise KeyboardInterrupt

        captured = CapturedQueue()

        class Microphone:
            def __init__(self, **kwargs):
                self.callback = kwargs["callback"]

            def __enter__(self):
                def feed():
                    try:
                        for _ in range(20):
                            self.callback(np.full((1024, 1), 0.01, dtype=np.float32), 1024, None, None)
                        observations.append(displayed.wait(2))
                        for _ in range(10):
                            self.callback(np.zeros((1024, 1), dtype=np.float32), 1024, None, None)
                    finally:
                        finished.set()
                self.thread = threading.Thread(target=feed)
                self.thread.start()
                return self

            def __exit__(self, *args):
                displayed.set()
                self.thread.join(2)

        model = Mock()

        def transcribe(audio, **kwargs):
            text = "앞으로" if kwargs["beam_size"] == 1 else "앞으로 30cm 이동해"
            segment = SimpleNamespace(text=text, start=0, end=1, avg_logprob=-0.2,
                                      no_speech_prob=0, compression_ratio=1, tokens=[1, 2, 3])
            return iter([segment]), SimpleNamespace(language_probability=1)

        model.transcribe.side_effect = transcribe

        def log(event, **kwargs):
            if event == "stt_preview_displayed":
                displayed.set()

        output = io.StringIO()
        with patch.object(speech, "_load_transcription_dependencies", return_value=(SimpleNamespace(InputStream=Microphone), Mock())), \
                patch.object(speech, "prepare_microphone"), \
                patch.object(speech, "SileroProbability", return_value=lambda frame: float(np.max(frame) > 0)), \
                patch.object(transcription, "_load_whisper_model", return_value=model), \
                patch.object(speech.queue, "Queue", return_value=captured), \
                patch.object(transcription, "record_event", side_effect=log), \
                contextlib.redirect_stdout(output), contextlib.redirect_stderr(io.StringIO()):
            speech.run_live_transcription(lambda *args: callbacks.append(args))
        self.assertEqual(observations, [True])
        self.assertIn("[음성 감지]", output.getvalue())
        self.assertIn("[임시 인식] 앞으로", output.getvalue())
        self.assertIn("[전사 중]", output.getvalue())
        self.assertEqual(callbacks, [("음성 입력", "앞으로 30cm 이동해")])

    def test_provisional_text_is_displayed_but_never_calls_final_callback(self):
        jobs = LiveTranscriptionQueue(speech._STOP)
        jobs.activate("a")
        jobs.offer_preview(preview())
        transcriber = quality.SpeechTranscriber(None, quality.SpeechOptions())
        calls = []

        def infer(audio, *, preview=False):
            self.assertTrue(preview)
            return result()

        transcriber.transcribe_audio = infer
        output = io.StringIO()
        # Wait on an event rather than sleeping until the preview has been printed.
        displayed = threading.Event()

        def log(event, **kwargs):
            if event == "stt_preview_displayed":
                displayed.set()

        with patch.object(transcription, "record_event", side_effect=log), contextlib.redirect_stdout(output):
            worker = threading.Thread(target=speech._transcription_worker,
                args=(transcriber, None, jobs, lambda *args: calls.append(args)))
            worker.start()
            try:
                self.assertTrue(displayed.wait(2))
            finally:
                jobs.put(speech._STOP)
                worker.join(2)
        self.assertFalse(worker.is_alive())
        self.assertEqual(calls, [])
        self.assertIn("[임시 인식] 임시 문장", output.getvalue())
        self.assertEqual(jobs.unfinished_tasks, 0)

    def test_inflight_preview_is_hidden_after_final_arrives_and_final_runs_once(self):
        jobs = LiveTranscriptionQueue(speech._STOP)
        jobs.activate("a")
        jobs.offer_preview(preview())
        transcriber = quality.SpeechTranscriber(None, quality.SpeechOptions())
        started, proceed = threading.Event(), threading.Event()
        inference_order, callbacks = [], []

        def infer(audio, *, preview=False):
            inference_order.append("preview" if preview else "final")
            if preview:
                started.set()
                if not proceed.wait(2):
                    raise RuntimeError("test wait timeout")
                value = result("폐기할 임시 문장")
            else:
                value = result("최종 명령")
            transcriber.last_result = value
            return value

        transcriber.transcribe_audio = infer
        output = io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(io.StringIO()):
            worker = threading.Thread(target=speech._transcription_worker,
                args=(transcriber, None, jobs, lambda *args: callbacks.append(args)))
            worker.start()
            try:
                self.assertTrue(started.wait(2))
                jobs.put(final())
                jobs.put(speech._STOP)
            finally:
                proceed.set()
                worker.join(2)
        self.assertFalse(worker.is_alive())
        self.assertEqual(inference_order, ["preview", "final"])
        self.assertEqual(callbacks, [("음성 입력", "최종 명령")])
        self.assertNotIn("폐기할 임시 문장", output.getvalue())
        self.assertEqual(jobs.unfinished_tasks, 0)


class PreviewAndLimitTests(unittest.TestCase):
    def test_detector_emits_throttled_bounded_snapshots_without_final_tasks(self):
        options = quality.SpeechOptions(preview_interval=1, preview_window=2)
        detector = speech.MonoUtteranceDetector(probability=lambda block: 1.0, options=options)
        frames = np.full(16000, 0.01, dtype=np.float32)
        self.assertEqual(detector.process(frames, 0), [])
        first = detector.preview_snapshot()
        self.assertIsNotNone(first)
        self.assertIsNone(detector.preview_snapshot())
        detector.process(frames[:1600], 1)
        self.assertIsNone(detector.preview_snapshot())
        detector.process(np.full(48000, 0.01, dtype=np.float32), 1.1)
        latest = detector.preview_snapshot()
        self.assertLessEqual(len(latest.audio), 32000)
        self.assertEqual(latest.utterance_id, first.utterance_id)
        self.assertEqual(len(detector.flush()), 1)
        self.assertIsNone(detector.preview_snapshot())

    def test_local_decoder_limits_and_fast_preview_beam(self):
        model = Mock()
        segment = SimpleNamespace(text="앞으로", start=0, end=1, avg_logprob=-0.2,
                                  no_speech_prob=0.01, compression_ratio=1, tokens=[1, 2, 3])
        model.transcribe.return_value = (iter([segment]), SimpleNamespace(language_probability=1))
        transcriber = quality.SpeechTranscriber(model, quality.SpeechOptions(beam_size=5))
        self.assertTrue(transcriber.transcribe_audio(np.ones(16000), preview=True).accepted)
        kwargs = model.transcribe.call_args.kwargs
        self.assertEqual(kwargs["beam_size"], 1)
        self.assertLessEqual(kwargs["max_new_tokens"], 40)
        self.assertEqual(kwargs["no_repeat_ngram_size"], 6)
        self.assertGreater(kwargs["repetition_penalty"], 1)
        self.assertTrue(kwargs["without_timestamps"])

    def test_token_limit_rejects_possible_truncation_instead_of_command_execution(self):
        model = Mock()
        # One-second final budget is 32, and near-budget output is held for re-speech.
        segment = SimpleNamespace(text="앞으로 30cm", start=0, end=1, avg_logprob=-0.1,
                                  no_speech_prob=0, compression_ratio=1, tokens=list(range(32)))
        model.transcribe.return_value = (iter([segment]), SimpleNamespace(language_probability=1))
        transcriber = quality.SpeechTranscriber(model, quality.SpeechOptions())
        value = transcriber.transcribe_audio(np.ones(16000))
        self.assertFalse(value.accepted)
        self.assertEqual(value.reason, "output_token_limit")
        self.assertEqual(value.text, "앞으로 30cm")

    def test_api_preview_is_disabled_without_a_paid_request(self):
        client = Mock()
        transcriber = quality.SpeechTranscriber(None, quality.SpeechOptions(backend="openai"), client=client)
        with self.assertRaises(ValueError):
            transcriber.transcribe_audio(np.ones(16000), preview=True)
        client.audio.transcriptions.create.assert_not_called()

    def test_cli_exposes_preview_and_output_limits(self):
        with patch.object(speech, "run_live_transcription") as run:
            self.assertEqual(cli.run(["--no-live-preview", "--stt-max-tokens", "64"]), 0)
        options = run.call_args.kwargs["options"]
        self.assertFalse(options.live_preview)
        self.assertEqual(options.max_output_tokens, 64)
        self.assertEqual(options.silence, 0.45)


if __name__ == "__main__":
    unittest.main()
