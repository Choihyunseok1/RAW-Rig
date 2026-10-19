import numpy as np
import pytest
from rawrig.signal_coverage import signal_coverage


def test_missing_signal_is_not_replaced_by_curve_interpolation():
    centers = np.geomspace(0.1, 1e4, 41)
    remove = 17
    observed = np.delete(centers, remove)
    report = signal_coverage(observed * 1000, observed, exposure_limits_us=(24, 3e7))
    assert not report[remove]["covered"]
    assert report[remove]["status"] == "needs_measurement"
    assert report[remove]["suggested_exposure_us"] == pytest.approx(
        centers[remove] * 1000
    )
    assert sum(row["covered"] for row in report) == 40


def test_saturation_and_hardware_minimum_are_explicit():
    report = signal_coverage(
        [24, 100, 1000], [0.25, 1.0, 10.0], valid=[True, True, False]
    )
    assert report[0]["status"] == "needs_optical_attenuation_or_outside_sensor_range"
    assert not report[0]["reachable_by_exposure_time"]
    assert not any(row["covered"] and 10 in [row["target_signal_e"]] for row in report)
