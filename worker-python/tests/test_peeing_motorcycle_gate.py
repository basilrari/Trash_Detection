# worker-python/tests/test_peeing_motorcycle_gate.py
from __future__ import annotations

import unittest

from models.peeing_motorcycle_gate import person_seated_on_motorcycle


class TestPeeingMotorcycleGate(unittest.TestCase):
    def _params(self) -> dict:
        return {
            "expand_x": 0.15,
            "expand_y": 0.10,
            "lower_body_fraction": 0.60,
            "overlap_threshold": 0.10,
        }

    def test_rider_is_motorcycle_seated(self) -> None:
        motorcycle = (40.0, 300.0, 200.0, 400.0)
        person = (50.0, 150.0, 180.0, 350.0)
        self.assertTrue(
            person_seated_on_motorcycle(
                person,
                [motorcycle],
                **self._params(),
            )
        )

    def test_pedestrian_far_not_seated(self) -> None:
        motorcycle = (40.0, 300.0, 200.0, 400.0)
        person = (400.0, 100.0, 480.0, 360.0)
        self.assertFalse(
            person_seated_on_motorcycle(
                person,
                [motorcycle],
                **self._params(),
            )
        )

    def test_pedestrian_beside_bike_not_seated(self) -> None:
        motorcycle = (40.0, 300.0, 200.0, 400.0)
        person = (210.0, 50.0, 280.0, 380.0)
        self.assertFalse(
            person_seated_on_motorcycle(
                person,
                [motorcycle],
                **self._params(),
            )
        )

    def test_floating_above_bike_not_seated(self) -> None:
        motorcycle = (40.0, 300.0, 200.0, 400.0)
        person = (60.0, 50.0, 170.0, 220.0)
        self.assertFalse(
            person_seated_on_motorcycle(
                person,
                [motorcycle],
                **self._params(),
            )
        )

    def test_empty_motorcycle_list(self) -> None:
        person = (50.0, 150.0, 180.0, 350.0)
        self.assertFalse(
            person_seated_on_motorcycle(
                person,
                [],
                **self._params(),
            )
        )


if __name__ == "__main__":
    unittest.main()
