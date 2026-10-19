"""Camera-specific, measured illumination timing used by the noise workspace."""

import json
import math
import os
from pathlib import Path


def calibration_path():
    return (
        Path(os.environ.get("XDG_STATE_HOME", str(Path.home() / ".local/state")))
        / "raw-rig/flicker.json"
    )


def measured_period(cameras):
    serials = {c.serial for c in cameras}
    if not serials or any(c.simulated for c in cameras):
        return 0.0
    try:
        data = json.loads(calibration_path().read_text())
        period = float(data["period_us"])
        if (
            serials != set(data["serials"])
            or not math.isfinite(period)
            or not 100 <= period <= 33333.334
        ):
            return 0.0
        return period
    except (OSError, ValueError, TypeError, KeyError):
        return 0.0


def measured_preview_exposure(cameras):
    period = measured_period(cameras)
    if not period:
        return None
    try:
        time = float(json.loads(calibration_path().read_text())["preview_exposure_us"])
        if (
            math.isfinite(time)
            and time > 0
            and abs(time - round(time / period) * period) <= max(10.0, period * 0.002)
        ):
            return time
    except (OSError, ValueError, TypeError, KeyError):
        pass
    return None
