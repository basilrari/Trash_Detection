"""Unit tests for one-side standing and groin distance in PeeingDetector."""

from __future__ import annotations

import unittest

import numpy as np

from models.peeing_detector import Coco17Landmark, PeeingDetector


def _blank_pose() -> tuple[np.ndarray, np.ndarray]:
    xyn = np.zeros((17, 2), dtype=np.float32)
    conf = np.zeros(17, dtype=np.float32)
    return xyn, conf


def _detector_stub() -> PeeingDetector:
    d = object.__new__(PeeingDetector)
    d.min_visibility = 0.45
    d.hand_groin_y_threshold = 0.1
    return d


class TestOneSideStanding(unittest.TestCase):
    def setUp(self) -> None:
        self.d = _detector_stub()
        self.L = Coco17Landmark

    def test_left_only_standing(self) -> None:
        xyn, conf = _blank_pose()
        conf[int(self.L.LEFT_HIP)] = 0.9
        conf[int(self.L.LEFT_KNEE)] = 0.9
        xyn[int(self.L.LEFT_HIP), 1] = 0.3
        xyn[int(self.L.LEFT_KNEE), 1] = 0.5
        self.assertTrue(self.d._is_standing_coco(xyn, conf))
        self.assertEqual(self.d._standing_sides_label(xyn, conf), "L")

    def test_right_only_standing(self) -> None:
        xyn, conf = _blank_pose()
        conf[int(self.L.RIGHT_HIP)] = 0.9
        conf[int(self.L.RIGHT_KNEE)] = 0.9
        xyn[int(self.L.RIGHT_HIP), 1] = 0.3
        xyn[int(self.L.RIGHT_KNEE), 1] = 0.5
        self.assertTrue(self.d._is_standing_coco(xyn, conf))
        self.assertEqual(self.d._standing_sides_label(xyn, conf), "R")

    def test_no_visible_knees(self) -> None:
        xyn, conf = _blank_pose()
        conf[int(self.L.LEFT_HIP)] = 0.9
        xyn[int(self.L.LEFT_HIP), 1] = 0.3
        self.assertFalse(self.d._is_standing_coco(xyn, conf))

    def test_squatting_left_not_standing(self) -> None:
        xyn, conf = _blank_pose()
        conf[int(self.L.LEFT_HIP)] = 0.9
        conf[int(self.L.LEFT_KNEE)] = 0.9
        xyn[int(self.L.LEFT_HIP), 1] = 0.6
        xyn[int(self.L.LEFT_KNEE), 1] = 0.4
        self.assertFalse(self.d._is_standing_coco(xyn, conf))


class TestOneSideGroin(unittest.TestCase):
    def setUp(self) -> None:
        self.d = _detector_stub()
        self.L = Coco17Landmark

    def test_left_hip_wrist_close(self) -> None:
        xyn, conf = _blank_pose()
        for i in (self.L.LEFT_HIP, self.L.LEFT_WRIST):
            conf[int(i)] = 0.9
        xyn[int(self.L.LEFT_HIP), 1] = 0.4
        xyn[int(self.L.LEFT_WRIST), 1] = 0.42
        self.assertLess(self.d._min_wrist_groin_y_dist(xyn, conf), 0.1)

    def test_only_left_hip_ignores_far_right_wrist(self) -> None:
        xyn, conf = _blank_pose()
        conf[int(self.L.LEFT_HIP)] = 0.9
        conf[int(self.L.LEFT_WRIST)] = 0.9
        conf[int(self.L.RIGHT_WRIST)] = 0.9
        xyn[int(self.L.LEFT_HIP), 1] = 0.4
        xyn[int(self.L.LEFT_WRIST), 1] = 0.42
        xyn[int(self.L.RIGHT_WRIST), 1] = 0.9
        self.assertAlmostEqual(
            self.d._min_wrist_groin_y_dist(xyn, conf), 0.02, places=5
        )

    def test_no_hips_returns_large_dist(self) -> None:
        xyn, conf = _blank_pose()
        self.assertGreaterEqual(self.d._min_wrist_groin_y_dist(xyn, conf), 1e8)

    def test_both_hips_uses_mid_groin(self) -> None:
        xyn, conf = _blank_pose()
        for i in (self.L.LEFT_HIP, self.L.RIGHT_HIP, self.L.LEFT_WRIST):
            conf[int(i)] = 0.9
        xyn[int(self.L.LEFT_HIP), 1] = 0.4
        xyn[int(self.L.RIGHT_HIP), 1] = 0.6
        xyn[int(self.L.LEFT_WRIST), 1] = 0.49
        dist = self.d._min_wrist_groin_y_dist(xyn, conf)
        self.assertAlmostEqual(dist, abs(0.49 - 0.5), places=5)


if __name__ == "__main__":
    unittest.main()
