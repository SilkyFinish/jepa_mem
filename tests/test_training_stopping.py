from datetime import datetime, timezone
from pathlib import Path
import tempfile
import unittest
from vjepa_policy.stopping import deadline_timestamp, stop_requested


class TrainingStopping(unittest.TestCase):
    def test_deadline_requires_timezone_and_honors_boundary(self):
        with self.assertRaisesRegex(ValueError, "timezone"):
            deadline_timestamp("2026-10-21T09:55:00")
        point = deadline_timestamp("2026-10-21T09:55:00+08:00")
        self.assertEqual(point, datetime(2026, 10, 21, 1, 55, tzinfo=timezone.utc).timestamp())
        self.assertFalse(stop_requested(point, now=point - 0.01))
        self.assertTrue(stop_requested(point, now=point))

    def test_operator_stop_does_not_wait_for_deadline(self):
        with tempfile.TemporaryDirectory() as directory:
            flag = Path(directory) / "stop"
            self.assertFalse(stop_requested(100, flag, now=10))
            flag.touch()
            self.assertTrue(stop_requested(100, flag, now=10))
