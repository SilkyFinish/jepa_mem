"""Wall-clock and operator stop requests checked at optimizer boundaries."""
from datetime import datetime
from pathlib import Path
import time


def deadline_timestamp(value):
    if not value:
        return None
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError("--stop-at must include a timezone offset")
    return parsed.timestamp()


def stop_requested(stop_at=None, stop_file=None, *, now=None):
    return bool(
        (stop_file and Path(stop_file).exists())
        or (stop_at is not None and (time.time() if now is None else now) >= stop_at)
    )
