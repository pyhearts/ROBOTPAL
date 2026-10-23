"""Offline execution, logical gait, protocol and concurrent input regression tests."""
import contextlib
import io
import json
import signal
import threading
import unittest
from unittest.mock import Mock, patch

import httpx2 as httpx
from openai import OpenAI
from pydantic import ValidationError

from module import cli
from module.decision_router import CommandDecision, CommandResult
from module.executable_plan import (ExecutablePlan, ExecutionStep, create_executable_plan,
                                    is_direct_stop, parse_timed_command, process_execution_command, stop_plan)
from module.execution import ExecutionManager
from module.execution_runtime import ExecutionRuntime, immediate_interrupt_stop
from module.gait import GaitModel, GaitTiming, GROUPS, PHASES
from module.robot_device import (HardwareProfile, JointProfile, SerialDevice, VirtualDevice,
                                SIMULATION_PROFILE)


def plan(action="forward", duration=200):
    return ExecutablePlan(status="plan", summary="가상 이동", steps=["이동한다.", "정지한다."],
                          error="", actions=[ExecutionStep(action=action, duration_ms=duration),
                                             ExecutionStep(action="stop", duration_ms=0)])


class Clock:
    def __init__(self):
        self.now = 0.0
    def __call__(self):
        return self.now
    def advance(self, amount=0.02):
        self.now += amount
        return self.now


class PlanTests(unittest.TestCase):
    def test_invalid_steps_and_whole_plan_rejected(self):
        for action, duration in [("jump", 2), ("forward", 0), ("forward", 10001),
                                 ("stop", 1), ("forward", True), ("forward", 1.2), ("forward", "100")]:
            with self.subTest(action=action, duration=duration), self.assertRaises(ValidationError):
                ExecutionStep(action=action, duration_ms=duration)
        base = plan().model_dump()
        for actions in [[], [{"action": "forward", "duration_ms": 1}],
                        [{"action": "forward", "duration_ms": 5000}] * 5 + [{"action": "stop", "duration_ms": 0}],
                        [{"action": "stop", "duration_ms": 0}] * 11]:
            with self.subTest(actions=actions), self.assertRaises(ValidationError):
                ExecutablePlan.model_validate({**base, "actions": actions})

    def test_invalid_plan_cannot_contain_actions_or_extra_motor_fields(self):
        with self.assertRaises(ValidationError):
            ExecutablePlan(status="invalid", summary="", steps=[], error="거절", actions=plan().actions)
        with self.assertRaises(ValidationError):
            ExecutionStep(action="forward", duration_ms=1, pwm=1500)

    def test_stop_detection_does_not_match_negation_question_or_mixed_request(self):
        for text in ["멈춰", " 정지! ", "그만."]:
            self.assertTrue(is_direct_stop(text))
        for text in ["멈추지 마", "멈춰?", "멈춰?!", "정지할까", "앞으로 가고 정지", "정지 그만"]:
            self.assertFalse(is_direct_stop(text))

    def test_local_stop_skips_all_api_calls(self):
        with patch("module.executable_plan.process_command") as process:
            result = process_execution_command("멈춰!")
        process.assert_not_called()
        self.assertEqual(result.plan.actions, stop_plan().actions)

    def test_ten_second_command_skips_low_confidence_api(self):
        client = Mock()
        client.decisions.create.return_value = {"answers": [{"type": "choice", "name": "robot_command",
            "choice": "concrete_command", "confidence": 0.77}]}
        with contextlib.redirect_stdout(io.StringIO()):
            result = process_execution_command("앞으로 10초 이동해", client=client)
        self.assertEqual(result.plan.actions[0], ExecutionStep(action="forward", duration_ms=10000))
        self.assertEqual(result.plan.actions[-1].action, "stop")
        client.decisions.create.assert_not_called()
        client.responses.create.assert_not_called()

    def test_exact_time_variants_and_directions(self):
        for text, action, duration in [("앞으로 10초 이동해", "forward", 10000),
                ("앞으로 0.5초 동안 걸어줘!", "forward", 500),
                ("뒤로 2초 이동해 주세요", "backward", 2000),
                ("왼쪽으로 1초 돌아", "turn_left", 1000),
                ("오른쪽으로 3초 회전해줘", "turn_right", 3000)]:
            with self.subTest(text=text):
                self.assertEqual(parse_timed_command(text).actions[0],
                                 ExecutionStep(action=action, duration_ms=duration))

    def test_local_rule_never_extracts_movement_from_ambiguous_text(self):
        for text in ["앞으로 10초 이동하지 마", "앞으로 10초 이동해?", "앞으로 이동해",
                     "앞으로 30cm 이동해", "앞으로 1초 이동해 그리고 점프해",
                     "'앞으로 10초 이동해'라고 말했어", "앞으로 1초 이동해 뒤로 1초 이동해",
                     "왼쪽으로 1초 이동해", "앞으로 1초 회전해", "앞으로 -1초 이동해"]:
            with self.subTest(text=text):
                self.assertIsNone(parse_timed_command(text))

    def test_local_rule_rejects_out_of_range_without_truncation_or_splitting(self):
        for text in ["앞으로 11초 이동해", "앞으로 0초 이동해", "앞으로 0.0005초 이동해"]:
            result = parse_timed_command(text)
            self.assertEqual(result.status, "invalid")
            self.assertFalse(result.actions)

    def test_distance_angle_count_and_missing_time_skip_planner_api(self):
        client = Mock()
        for text in ["앞으로 가", "앞으로 30cm", "90도 돌아", "2초 동안 30cm", "앞으로 2초씩 세 번"]:
            self.assertEqual(create_executable_plan(text, client=client).status, "invalid")
        client.responses.create.assert_not_called()

    def test_real_sdk_structured_response_and_no_image(self):
        requests = []
        def respond(request):
            requests.append(request)
            return httpx.Response(200, json={"id": "test", "object": "response", "created_at": 0,
                "model": "test", "status": "completed", "output": [{"type": "message", "id": "m",
                "role": "assistant", "status": "completed", "content": [{"type": "output_text",
                "text": plan().model_dump_json(), "annotations": []}]}]})
        with OpenAI(api_key="offline", max_retries=0,
                    http_client=httpx.Client(transport=httpx.MockTransport(respond))) as client:
            result = create_executable_plan("앞으로 0.2초", image=b"ignored", client=client)
        self.assertEqual(result.actions[0].duration_ms, 200)
        body = json.loads(requests[0].content)
        self.assertEqual(body["text"]["format"]["schema"], ExecutablePlan.model_json_schema())
        self.assertFalse(body["store"])
        self.assertNotIn("input_image", str(body))

    def test_bad_model_output_and_incomplete_response_rejected(self):
        from module.ai_api import PlannerResponseError
        client = Mock()
        for status, output in [("completed", "{}"), ("completed", plan().model_dump_json().replace('"forward"', '"jump"')),
                               ("incomplete", plan().model_dump_json())]:
            client.responses.create.return_value = {"status": status, "output_text": output}
            with self.assertRaises(PlannerResponseError):
                create_executable_plan("앞으로 1초", client=client)


class GaitTests(unittest.TestCase):
    def test_groups_are_disjoint_and_cover_eight_legs(self):
        self.assertEqual(set(GROUPS["A"]) & set(GROUPS["B"]), set())
        self.assertEqual(set(GROUPS["A"] + GROUPS["B"]), {f"{s}{i}" for s in "RL" for i in range(1, 5)})

    def test_all_directions_keep_the_same_phase_sequence(self):
        for direction in ["forward", "backward", "turn_left", "turn_right"]:
            gait = GaitModel()
            gait.start(direction, 5000, 0)
            phases = [gait.phase]
            now = 0
            for _ in range(8):
                now = gait.phase_end + 0.001
                gait.advance(now)
                phases.append(gait.phase)
                groups = gait.snapshot()["groups"]
                self.assertLessEqual(sum(v != "support" for v in groups.values()), 1)
            self.assertEqual(phases, list(PHASES) + ["support"])

    def test_stop_at_every_phase_never_lifts_next_group(self):
        for index in range(8):
            with self.subTest(index=index):
                gait = GaitModel()
                gait.start("forward", 5000, 0)
                now = 0
                for _ in range(index):
                    now = gait.phase_end + 0.001
                    gait.advance(now)
                old_group = gait.phase[0] if gait.phase != "support" else None
                gait.stop(now)
                observed = []
                for _ in range(5):
                    now += 0.2
                    gait.advance(now)
                    observed.append(gait.phase)
                self.assertFalse(gait.active)
                self.assertTrue(all(p == "support" or p == f"{old_group}_lower" for p in observed))

    def test_expiry_and_long_tick_gap_never_replay_lifts(self):
        gait = GaitModel()
        gait.start("forward", 100, 0)
        gait.advance(30)
        self.assertFalse(gait.active)
        self.assertEqual(gait.phase, "support")

    def test_simulation_timing_cannot_be_used_as_hardware_timing(self):
        for kwargs in [{"simulation_only": False}, {"lift": float("nan")}, {"lower": 0}]:
            with self.assertRaises(ValueError):
                GaitTiming(**kwargs)


class ManagerTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.events = []
        self.device = VirtualDevice()
        self.manager = ExecutionManager(self.device, clock=self.clock, emit=self.events.append)

    def drive(self, seconds=2):
        for _ in range(int(seconds / 0.02)):
            self.clock.advance()
            self.manager.tick()

    def test_whole_plan_completes_with_no_contact_claim(self):
        request = self.manager.begin()
        self.manager.offer(request, plan())
        self.drive()
        self.assertEqual(self.manager.outcome, "completed")
        self.assertEqual(self.manager.state, "ready")
        self.assertTrue(self.manager.settled)
        self.assertTrue(all(e.get("contact_confirmed") is False for e in self.events if e["event"] == "completed"))

    def test_user_ten_second_command_runs_both_groups_then_stops(self):
        with contextlib.redirect_stdout(io.StringIO()):
            result = process_execution_command("앞으로 10초 이동해")
        request = self.manager.begin()
        self.assertTrue(self.manager.offer(request, result.plan))
        self.drive(9.8)
        self.assertEqual(self.manager.state, "running")
        self.drive(1)
        self.assertTrue(self.manager.settled)
        self.assertEqual(self.manager.outcome, "completed")
        phases = {e["device_event"].get("phase") for e in self.events if e["event"] == "device"}
        self.assertTrue({"A_lift", "B_lift", "support"} <= phases)

    def test_new_request_lands_before_new_direction(self):
        old = self.manager.begin()
        self.manager.offer(old, plan(duration=2000))
        self.drive(0.2)
        self.assertEqual(self.device.gait.phase, "A_lift")
        new = self.manager.begin()
        self.manager.offer(new, plan("backward"))
        self.assertEqual(self.manager.state, "stopping")
        self.manager.tick()
        self.assertEqual(self.device.gait.direction, "forward")
        self.drive()
        steps = [e["action"] for e in self.events if e["event"] == "step"]
        self.assertEqual(steps, ["forward", "backward", "stop"])

    def test_invalid_replacement_leaves_robot_stopped(self):
        old = self.manager.begin()
        self.manager.offer(old, plan(duration=2000))
        self.drive(0.2)
        new = self.manager.begin()
        self.manager.reject(new, "시간 누락")
        self.drive()
        self.assertTrue(self.device.idle)
        self.assertEqual(self.manager.state, "ready")
        self.assertEqual(self.manager.outcome, "rejected")

    def test_late_old_response_and_post_stop_response_are_discarded(self):
        old = self.manager.begin()
        current = self.manager.begin()
        self.assertFalse(self.manager.offer(old, plan()))
        self.manager.stop()
        self.assertFalse(self.manager.offer(current, plan()))
        self.drive()
        self.assertFalse(any(e["event"] == "step" for e in self.events))

    def test_request_expires_before_plan_returns(self):
        request = self.manager.begin()
        self.drive(10.1)
        self.assertFalse(self.manager.offer(request, plan()))
        self.assertEqual(self.manager.outcome, "expired")

    def test_expired_replacement_does_not_start_after_long_landing(self):
        self.device.gait = GaitModel(GaitTiming(lower=11))
        old = self.manager.begin()
        self.manager.offer(old, plan(duration=2000))
        self.drive(0.2)
        new = self.manager.begin()
        self.manager.offer(new, plan("backward"))
        self.drive(12)
        self.assertEqual(self.manager.outcome, "expired")
        self.assertEqual([e["action"] for e in self.events if e["event"] == "step"], ["forward"])

    def test_mutated_plan_is_revalidated_as_a_whole(self):
        candidate = plan()
        candidate.actions.append({"action": "jump", "duration_ms": 100})
        request = self.manager.begin()
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertFalse(self.manager.offer(request, candidate))
        self.assertEqual(self.manager.outcome, "rejected")

    def test_device_fault_latches_and_never_resumes(self):
        request = self.manager.begin()
        self.manager.offer(request, plan(duration=2000))
        self.drive(0.2)
        self.device.inject_fault("관절 오류", self.clock())
        self.drive()
        self.assertEqual(self.manager.state, "fault")
        self.manager.stop()
        self.assertEqual(self.manager.state, "fault")
        with self.assertRaises(RuntimeError):
            self.manager.begin()

    def test_heartbeat_expiry_is_enforced_by_device_without_manager(self):
        self.device.move(ExecutionStep(action="forward", duration_ms=5000), "m", 0)
        self.device.tick(0)
        self.device.tick(0.2)
        events = self.device.tick(1.1)
        self.assertTrue(any(e["event"] == "error" for e in events))
        self.assertTrue(self.device.gait.stopping)
        self.device.tick(1.4)
        self.device.tick(1.6)
        self.assertTrue(self.device.idle)

    def test_disconnect_reconnect_does_not_resume(self):
        self.device.move(ExecutionStep(action="forward", duration_ms=1000), "m", 0)
        self.device.tick(0)
        old_session = self.device.session
        self.device.disconnect(0.01)
        self.device.tick(0.3)
        self.device.reconnect()
        self.assertNotEqual(self.device.session, old_session)
        self.assertFalse(self.device.prepared)
        self.assertTrue(self.device.idle)
        with self.assertRaises(RuntimeError):
            self.device.prepare(0.4)

    def test_late_heartbeat_cannot_hide_a_stalled_controller_loop(self):
        request = self.manager.begin()
        self.manager.offer(request, plan(duration=5000))
        self.manager.tick()
        self.clock.advance(1.1)
        self.manager.tick()
        self.assertEqual(self.manager.state, "fault")

    def test_delayed_acceptance_and_duplicate_move(self):
        device = VirtualDevice(accept_delay=0.3)
        device.prepare(0)
        device.move(ExecutionStep(action="forward", duration_ms=10), "x", 0)
        device.tick(0.2)
        self.assertFalse(device.gait.active)
        device.tick(0.3)
        self.assertTrue(device.gait.active)
        device.tick(0.4)
        self.assertTrue(device.idle)
        with self.assertRaises(RuntimeError):
            device.move(ExecutionStep(action="forward", duration_ms=10), "x", 0.5)


class Transport:
    def __init__(self):
        self.writes = []
        self.incoming = b""
    def write(self, data):
        self.writes.append(json.loads(data))
        return len(data)
    def read(self, maximum):
        data, self.incoming = self.incoming[:maximum], self.incoming[maximum:]
        return data
    def reply(self, packet, event="accepted", **overrides):
        reply = {k: packet[k] for k in ("version", "session", "profile", "request_id")}
        self.incoming += (json.dumps({**reply, "event": event, "simulation": True, **overrides}) + "\n").encode()


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.transport = Transport()
        self.device = SerialDevice(self.transport)

    def ready(self):
        self.device.prepare(0)
        self.transport.reply(self.transport.writes[-1])
        self.device.tick(0.01)
        self.assertEqual(self.transport.writes[-1]["op"], "ready")
        self.transport.reply(self.transport.writes[-1])
        self.device.tick(0.02)
        self.assertTrue(self.device.ready)

    def test_handshake_move_stop_and_late_cancelled_completion(self):
        self.ready()
        self.device.move(plan().actions[0], "move-1", 0.03)
        move = self.transport.writes[-1]
        self.transport.reply(move)
        self.device.tick(0.04)
        self.device.stop(0.05)
        stop = self.transport.writes[-1]
        self.transport.reply(stop)
        self.device.tick(0.06)
        self.transport.reply(stop, "completed")
        self.device.tick(0.07)
        self.assertTrue(self.device.idle)
        self.transport.reply(move, "completed")
        self.device.tick(0.08)
        self.assertIsNone(self.device.fault)

    def test_identity_mismatches_and_real_output_peer_are_rejected(self):
        for override in [{"version": 2}, {"session": "old"}, {"profile": "wrong"},
                         {"request_id": "unknown"}, {"simulation": False}]:
            with self.subTest(override=override):
                transport = Transport()
                device = SerialDevice(transport)
                device.prepare(0)
                transport.reply(transport.writes[-1], **override)
                device.tick(0.1)
                self.assertIsNotNone(device.fault)
                self.assertFalse(device.ready)

    def test_missing_ack_never_retransmits_move(self):
        self.ready()
        self.device.move(plan().actions[0], "m", 0.03)
        self.device.tick(0.6)
        self.assertIsNotNone(self.device.fault)
        self.device.tick(1)
        self.assertEqual(sum(p["op"] == "move" for p in self.transport.writes), 1)

    def test_move_completion_cannot_release_a_pending_stop(self):
        self.ready()
        self.device.move(plan().actions[0], "m", 0.03)
        move = self.transport.writes[-1]
        self.transport.reply(move)
        self.device.tick(0.04)
        self.device.stop(0.05)
        stop = self.transport.writes[-1]
        self.transport.reply(stop)
        self.transport.reply(move, "completed")
        self.device.tick(0.06)
        self.assertFalse(self.device.idle)
        self.transport.reply(stop, "completed")
        self.device.tick(0.07)
        self.assertTrue(self.device.idle)

    def test_bad_partial_packet_and_completion_timeout(self):
        self.ready()
        self.transport.incoming = b"{bad\n"
        self.device.tick(0.03)
        self.assertIsNotNone(self.device.fault)
        self.setUp()
        self.ready()
        self.device.move(plan().actions[0], "m", 0.03)
        self.transport.reply(self.transport.writes[-1])
        self.device.tick(0.04)
        self.device.tick(3)
        self.assertIn("완료", self.device.fault)

    def test_fragmented_line_is_buffered_until_newline(self):
        self.device.prepare(0)
        self.transport.reply(self.transport.writes[-1])
        packet = self.transport.incoming
        self.transport.incoming = packet[:8]
        self.device.tick(0.1)
        self.assertFalse(self.device.ready)
        self.transport.incoming = packet[8:]
        self.device.tick(0.2)
        self.assertEqual(self.transport.writes[-1]["op"], "ready")

    def test_common_manager_waits_for_handshake_then_completes_plan(self):
        clock = Clock()
        manager = ExecutionManager(self.device, clock=clock)
        request = manager.begin()
        manager.offer(request, plan())
        manager.tick()
        self.assertEqual(manager.state, "ready")
        self.assertEqual(len(self.transport.writes), 1)
        self.transport.reply(self.transport.writes[-1])  # hello
        clock.advance()
        manager.tick()
        self.transport.reply(self.transport.writes[-1])  # ready
        clock.advance()
        manager.tick()
        move = self.transport.writes[-1]
        self.assertEqual(move["op"], "move")
        self.transport.reply(move)
        self.transport.reply(move, "completed")
        clock.advance()
        manager.tick()
        stop = self.transport.writes[-1]
        self.assertEqual(stop["op"], "stop")
        self.transport.reply(stop)
        self.transport.reply(stop, "completed")
        clock.advance()
        manager.tick()
        self.assertTrue(manager.settled)
        self.assertEqual(manager.outcome, "completed")

    def test_repeated_stop_does_not_restart_landing_or_send_duplicate(self):
        self.ready()
        self.device.stop(0.03)
        self.device.stop(0.04)
        self.assertEqual(sum(p["op"] == "stop" for p in self.transport.writes), 1)

    def test_late_ack_is_not_accepted_and_status_query_is_correlated(self):
        self.device.prepare(0)
        self.transport.reply(self.transport.writes[-1])
        self.device.tick(0.6)
        self.assertIsNotNone(self.device.fault)
        self.setUp()
        self.ready()
        self.device.status(0.03)
        self.transport.reply(self.transport.writes[-1], "status", state="ready")
        self.device.tick(0.04)
        self.assertIsNone(self.device.fault)
        self.assertFalse(self.device.pending)


class ProfileTests(unittest.TestCase):
    def profile(self):
        joints = [JointProfile(leg=leg, role="hip", servo_model="unconfirmed", address=0x40,
                  channel=i, direction=1, minimum_us=1000, neutral_us=1500, maximum_us=2000,
                  max_change_us_per_second=100) for i, leg in enumerate(sum(GROUPS.values(), ()))]
        return HardwareProfile(name="example-only", revision="1", simulation_only=True,
                               calibrated=False, support_verified=False, fault_policy="unverified", joints=joints)

    def test_unverified_profile_blocked_and_fingerprint_changes(self):
        profile = self.profile()
        with self.assertRaises(ValueError):
            profile.require_hardware_ready()
        changed = HardwareProfile.model_validate({**profile.model_dump(), "revision": "2"})
        self.assertNotEqual(profile.fingerprint, changed.fingerprint)

    def test_missing_leg_duplicate_channel_and_invalid_ranges_rejected(self):
        profile = self.profile().model_dump()
        for mutation in ["leg", "channel", "range"]:
            data = json.loads(json.dumps(profile))
            if mutation == "leg":
                data["joints"].pop()
            elif mutation == "channel":
                data["joints"][1]["channel"] = data["joints"][0]["channel"]
            else:
                data["joints"][0]["neutral_us"] = 3000
            with self.assertRaises(ValidationError):
                HardwareProfile.model_validate(data)


class RuntimeTests(unittest.TestCase):
    def test_text_cli_routes_real_sdk_response_into_virtual_gait(self):
        events = []
        def respond(request):
            if request.url.path.endswith("/decisions"):
                return httpx.Response(200, json={"model": "test", "answers": [{"type": "choice",
                    "name": "robot_command", "choice": "concrete_command", "confidence": 1}]})
            return httpx.Response(200, json={"id": "test", "object": "response", "created_at": 0,
                "model": "test", "status": "completed", "output": [{"type": "message", "id": "m",
                "role": "assistant", "status": "completed", "content": [{"type": "output_text",
                "text": plan().model_dump_json(), "annotations": []}]}]})
        client = OpenAI(api_key="offline", max_retries=0,
                        http_client=httpx.Client(transport=httpx.MockTransport(respond)))
        self.addCleanup(client.close)
        with patch("module.decision_router.create_api_client", return_value=client), \
             patch("module.execution_runtime.console_event", side_effect=events.append), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(cli.run(["--execute", "--command", "로봇 기준 전방으로 0.2초 움직여"]), 0)
        phases = [e["device_event"]["phase"] for e in events if e["event"] == "device"
                  and e["device_event"].get("event") == "gait"]
        self.assertIn("A_lift", phases)
        self.assertIn("A_lower", phases)
        self.assertEqual(phases[-1], "support")
        self.assertTrue(any(e["event"] == "completed" for e in events))

    def test_slow_inference_does_not_block_stop_and_stale_result(self):
        entered, release, returned = threading.Event(), threading.Event(), threading.Event()
        def processor(text, **kwargs):
            entered.set()
            release.wait(2)
            returned.set()
            return CommandResult(CommandDecision("concrete_command", 1), plan=plan())
        manager = ExecutionManager()
        runtime = ExecutionRuntime(manager=manager, processor=processor)
        try:
            runtime.submit("앞으로 1초")
            self.assertTrue(entered.wait(1))
            runtime.on_transcription("speaker", "멈춰")
            self.assertFalse(manager.awaiting)
            release.set()
            self.assertTrue(returned.wait(1))
            self.assertTrue(runtime.wait(1))
            self.assertFalse(manager.steps)
        finally:
            release.set()
            runtime.close()

    def test_latest_input_can_complete_while_old_worker_is_blocked(self):
        entered, release = threading.Event(), threading.Event()
        def processor(text, **kwargs):
            if text == "old":
                entered.set()
                release.wait(2)
            return CommandResult(CommandDecision("concrete_command", 1), plan=plan(duration=20))
        runtime = ExecutionRuntime(manager=ExecutionManager(), processor=processor)
        try:
            runtime.submit("old")
            self.assertTrue(entered.wait(1))
            runtime.submit("new")
            self.assertTrue(runtime.wait(2))
            self.assertEqual(runtime.manager.outcome, "completed")
        finally:
            release.set()
            runtime.close()

    def test_ctrl_c_blocks_drained_final_callbacks_and_still_closes_loop(self):
        processor = Mock()
        runtime = ExecutionRuntime(manager=ExecutionManager(), processor=processor)
        try:
            with immediate_interrupt_stop(runtime):
                with self.assertRaises(KeyboardInterrupt):
                    signal.getsignal(signal.SIGINT)(signal.SIGINT, None)
            runtime.on_transcription("speaker", "앞으로 2초")
            processor.assert_not_called()
        finally:
            runtime.close()
        self.assertFalse(runtime.loop.is_alive())

    def test_execution_voice_passes_only_final_callback_without_camera(self):
        from module.execution_runtime import run_execution
        with patch("module.ai_api.create_api_client") as client, \
             patch("module.speech.run_live_transcription") as speech, \
             patch("module.execution_runtime.ExecutionRuntime") as runtime, \
             contextlib.redirect_stdout(io.StringIO()):
            runtime.return_value.manager.state = "ready"
            runtime.return_value.manager.outcome = None
            run_execution(voice=True, speech_kwargs={"offline": True})
        kwargs = speech.call_args.kwargs
        self.assertEqual(kwargs["on_transcription"], runtime.return_value.on_transcription)
        self.assertTrue(kwargs["offline"])
        self.assertFalse(kwargs["identify_speakers"])
        client.return_value.close.assert_called_once()
        runtime.return_value.close.assert_called_once()

    def test_cli_modes_are_explicit_and_reject_camera_or_invalid_command_usage(self):
        for flags in [["--execute", "--camera", "0"], ["--execute-voice", "--command", "x"],
                      ["--execute", "--plan"]]:
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                cli.run(flags)
        with patch("module.execution_runtime.run_execution", return_value=0) as run:
            self.assertEqual(cli.run(["--execute", "--command", "멈춰"]), 0)
            run.assert_called_once_with(command="멈춰")


if __name__ == "__main__":
    unittest.main()
