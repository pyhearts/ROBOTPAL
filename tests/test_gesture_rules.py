import math
import types
import unittest

import gesture
from gesture_rules import EXTENDED, FOLDED, UNCERTAIN, classify_landmarks


def landmark(point):
    return types.SimpleNamespace(x=point[0], y=point[1], z=point[2])


def hand_pose(extended=(), ok=False):
    """Create a synthetic but anatomically indexed 21-point hand."""
    points = [(0.0, 0.0, 0.0) for _ in range(21)]
    points[0] = (0.0, 0.0, 0.0)

    if "thumb" in extended:
        points[1:5] = [
            (-0.3, 0.3, 0.0),
            (-0.6, 0.6, 0.0),
            (-0.9, 0.9, 0.0),
            (-1.2, 1.2, 0.0),
        ]
    else:
        points[1:5] = [
            (-0.3, 0.3, 0.0),
            (-0.5, 0.5, 0.0),
            (-0.4, 0.65, 0.0),
            (-0.1, 0.5, 0.0),
        ]

    fingers = {
        "index": (5, (-0.35, 0.8, 0.0)),
        "middle": (9, (0.0, 1.0, 0.0)),
        "ring": (13, (0.32, 0.85, 0.0)),
        "pinky": (17, (0.58, 0.7, 0.0)),
    }
    for name, (mcp, base) in fingers.items():
        x, y, z = base
        points[mcp] = base
        if name in extended:
            points[mcp + 1] = (x, y + 0.45, z)
            points[mcp + 2] = (x, y + 0.85, z)
            points[mcp + 3] = (x, y + 1.2, z)
        else:
            points[mcp + 1] = (x, y + 0.4, z)
            points[mcp + 2] = (x + 0.3, y + 0.2, z)
            points[mcp + 3] = (x + 0.45, y, z)

    if ok:
        # Touch thumb and index tips without changing the three open fingers.
        points[4] = (-0.25, 1.15, 0.0)
        points[8] = (-0.25, 1.15, 0.0)
    return [landmark(point) for point in points]


def transform(landmarks, scale=1.0, angle_degrees=0.0, mirror=False):
    angle = math.radians(angle_degrees)
    cosine, sine = math.cos(angle), math.sin(angle)
    result = []
    for item in landmarks:
        x = -item.x if mirror else item.x
        y = item.y
        result.append(
            landmark(
                (
                    scale * (x * cosine - y * sine),
                    scale * (x * sine + y * cosine),
                    scale * item.z,
                )
            )
        )
    return result


class RuleClassifierTests(unittest.TestCase):
    CASES = {
        "Number_1": ("index",),
        "Number_2": ("index", "middle"),
        "Number_3": ("index", "middle", "ring"),
        "Number_4": ("index", "middle", "ring", "pinky"),
        "Number_5": ("thumb", "index", "middle", "ring", "pinky"),
    }

    def test_numbers_one_through_five(self):
        for expected, fingers in self.CASES.items():
            with self.subTest(expected=expected):
                result = classify_landmarks(hand_pose(fingers))
                self.assertIsNotNone(result)
                self.assertEqual(result.gesture, expected)
                self.assertEqual(result.confidence, 1.0)

    def test_both_number_three_forms(self):
        regular = classify_landmarks(hand_pose(("index", "middle", "ring")))
        alternate = classify_landmarks(hand_pose(("thumb", "index", "middle")))
        self.assertEqual(regular.gesture, "Number_3")
        self.assertEqual(alternate.gesture, "Number_3")

    def test_ok_has_priority(self):
        value = classify_landmarks(hand_pose(("middle", "ring", "pinky"), ok=True))
        self.assertEqual(value.gesture, "OK")

    def test_ok_distance_boundary_is_inclusive(self):
        pose = hand_pose(("middle", "ring", "pinky"))
        # Palm scale is exactly 1.0 in this fixture.
        pose[4] = landmark((-0.25, 1.15, 0.0))
        pose[8] = landmark((0.0, 1.15, 0.0))
        self.assertEqual(classify_landmarks(pose).gesture, "OK")
        pose[8] = landmark((0.001, 1.15, 0.0))
        self.assertNotEqual(classify_landmarks(pose).gesture, "OK")

    def test_rotation_scale_and_mirroring_do_not_change_result(self):
        original = hand_pose(("index", "middle"))
        for scale, angle, mirror in ((0.5, 70, False), (2.4, -125, True), (1.3, 180, False)):
            with self.subTest(scale=scale, angle=angle, mirror=mirror):
                value = classify_landmarks(transform(original, scale, angle, mirror))
                self.assertEqual(value.gesture, "Number_2")

    def test_invalid_and_zero_size_hands_are_rejected(self):
        self.assertIsNone(classify_landmarks(hand_pose(("index",))[:20]))
        zero = [landmark((0.0, 0.0, 0.0)) for _ in range(21)]
        self.assertIsNone(classify_landmarks(zero))

    def test_ambiguous_finger_blocks_number(self):
        pose = hand_pose(("index",))
        # Straight angles, but a radial ratio between folded and extended.
        pose[8] = landmark((pose[6].x, pose[6].y * 1.05, 0.0))
        value = classify_landmarks(pose)
        self.assertIsNotNone(value)
        self.assertIsNone(value.gesture)
        self.assertIn(UNCERTAIN, value.finger_states)


class RuleIntegrationTests(unittest.TestCase):
    @staticmethod
    def media_result(canned, landmarks, world_landmarks=None):
        category = types.SimpleNamespace(category_name=canned, score=0.95)
        hand = types.SimpleNamespace(category_name="Right", score=1.0)
        return types.SimpleNamespace(
            gestures=[[category]],
            handedness=[[hand]],
            hand_landmarks=[landmarks],
            hand_world_landmarks=[] if world_landmarks is None else [world_landmarks],
        )

    def test_basic_mode_is_unchanged(self):
        value = self.media_result("Victory", hand_pose(("index", "middle")))
        self.assertEqual(gesture.extract_observation(value, rule_set="basic").gesture, "Victory")

    def test_numbers_mode_prioritizes_landmark_rule(self):
        value = self.media_result("Victory", hand_pose(("index", "middle")))
        observation = gesture.extract_observation(value, rule_set="numbers")
        self.assertEqual(observation, gesture.GestureObservation("Right", "Number_2", 1.0))

    def test_numbers_mode_suppresses_overlapping_canned_fallback(self):
        ambiguous = hand_pose(("index",))
        ambiguous[8] = landmark((ambiguous[6].x, ambiguous[6].y * 1.05, 0.0))
        value = self.media_result("Pointing_Up", ambiguous)
        self.assertIsNone(gesture.extract_observation(value, rule_set="numbers"))

    def test_numbers_mode_keeps_non_overlapping_fallback(self):
        value = self.media_result("Thumb_Up", [])
        self.assertEqual(
            gesture.extract_observation(value, rule_set="numbers").gesture,
            "Thumb_Up",
        )

    def test_world_landmarks_take_priority(self):
        normalized = hand_pose(("index",))
        world = hand_pose(("index", "middle", "ring", "pinky"))
        value = self.media_result("None", normalized, world)
        self.assertEqual(
            gesture.extract_observation(value, rule_set="numbers").gesture,
            "Number_4",
        )

    def test_rule_result_uses_existing_stabilizer(self):
        value = self.media_result("Victory", hand_pose(("index", "middle")))
        state = gesture._ResultState("numbers")
        state.receive(value, None, 0)
        state.receive(value, None, 399)
        self.assertTrue(state.events.empty())
        state.receive(value, None, 400)
        event = state.events.get_nowait()
        self.assertEqual(event.gesture, "Number_2")


if __name__ == "__main__":
    unittest.main()
