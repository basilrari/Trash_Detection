# worker-python/core/writer.py
from __future__ import annotations
import csv
import os
import tempfile
from typing import Optional

FIELDS = ["video_name", "timestamp", "crime", "vehicle_number"]


class CsvWriter:
    """
    Minimal CSV writer for your final required schema:
    video_name, timestamp, crime, vehicle_number
    """

    def __init__(self, out_path: str, video_name: str):
        self.out_path = out_path
        self.video_name = video_name

        # Write into a temp file then atomically replace
        out_dir = os.path.dirname(os.path.abspath(out_path)) or "."
        os.makedirs(out_dir, exist_ok=True)

        fd, tmp = tempfile.mkstemp(prefix=".tmp_", suffix=".csv", dir=out_dir)
        os.close(fd)
        self._tmp_path = tmp

        self._f = open(self._tmp_path, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._f, fieldnames=FIELDS)
        self._writer.writeheader()

    def write_event(self, timestamp: float, crime: str, vehicle_number: Optional[str]):
        """
        Write one simplified CSV row.
        """
        self._writer.writerow({
            "video_name": self.video_name,
            "timestamp": timestamp,
            "crime": crime,
            "vehicle_number": vehicle_number or ""
        })

    def close(self):
        self._f.flush()
        self._f.close()
        # atomic replace
        os.replace(self._tmp_path, self.out_path)
