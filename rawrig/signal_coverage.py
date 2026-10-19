"""Measured signal coverage, never filling a curve with interpolated measurements."""

import math
import numpy as np


def signal_coverage(
    exposure_us,
    signal_e,
    valid=None,
    minimum_e=0.1,
    maximum_e=1e4,
    bins_per_decade=8,
    exposure_limits_us=(24.0, 3e7),
):
    times = np.asarray(exposure_us, dtype=float)
    signal = np.asarray(signal_e, dtype=float)
    good = np.isfinite(signal) & np.isfinite(times) & (signal > 0) & (times > 0)
    if valid is not None:
        good &= np.asarray(valid, dtype=bool)
    if minimum_e <= 0 or maximum_e <= minimum_e or bins_per_decade < 1:
        raise ValueError("Positive increasing signal range and bin density required")
    centers = 10 ** np.linspace(
        math.log10(minimum_e),
        math.log10(maximum_e),
        round(math.log10(maximum_e / minimum_e) * bins_per_decade) + 1,
    )
    width = (math.log10(maximum_e) - math.log10(minimum_e)) / (len(centers) - 1)
    rows = []
    if good.any():
        order = np.argsort(signal[good])
        S = signal[good][order]
        T = times[good][order]
        rate = float(np.median(S / T))
    else:
        rate = None
    for center in centers:
        lo = center / 10 ** (width / 2)
        hi = center * 10 ** (width / 2)
        selected = np.flatnonzero(good & (signal >= lo) & (signal < hi))
        predicted = center / rate if rate else None
        reachable = (
            predicted is not None
            and exposure_limits_us[0] <= predicted <= exposure_limits_us[1]
        )
        rows.append(
            dict(
                target_signal_e=float(center),
                bin_low_e=float(lo),
                bin_high_e=float(hi),
                measured_point_indices=selected.tolist(),
                measured_count=int(len(selected)),
                covered=bool(len(selected)),
                suggested_exposure_us=float(predicted) if predicted else None,
                reachable_by_exposure_time=bool(reachable),
                status="measured"
                if len(selected)
                else "needs_measurement"
                if reachable
                else "needs_optical_attenuation_or_outside_sensor_range",
            )
        )
    return rows
