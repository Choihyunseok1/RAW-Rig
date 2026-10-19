"""Fit periodic illumination using disjoint guard photometry and camera clocks."""

import math
import numpy as np


def phase_for_exposure(model, reference_us, target_us, timestamp_position='exposure_start'):
    """Transfer integrated harmonics between exposure times without changing RAW."""
    if timestamp_position not in ('exposure_start','exposure_end'):
        raise ValueError('Camera timestamp position must be measured first')
    if min(reference_us,target_us)<=0:
        raise ValueError('Exposure times must be positive')
    omega=2*np.pi*model['frequency_hz']
    sign=1 if timestamp_position=='exposure_start' else -1
    coefficients=[]
    for k in range(1,4):
        ref=np.sinc(k*model['frequency_hz']*reference_us/1e6)
        if abs(ref)<.05:
            raise ValueError('Reference exposure suppresses a phase harmonic')
        scale=target_us/reference_us*np.sinc(k*model['frequency_hz']*target_us/1e6)/ref
        c,s=model['harmonic_coefficients'][2*(k-1):2*k]
        a=(c-1j*s)*scale*np.exp(sign*1j*k*omega*(target_us-reference_us)/2e6)
        coefficients.extend([float(a.real),float(-a.imag)])
    angles=np.linspace(0,2*np.pi,4096,endpoint=False)
    wave=sum(coefficients[2*(k-1)]*np.cos(k*angles)+coefficients[2*(k-1)+1]*np.sin(k*angles) for k in range(1,4))
    return {**model,'phase_ns':float(angles[np.argmax(wave)]/(2*np.pi)*model['period_ns']),
            'harmonic_coefficients':coefficients,'reference_exposure_us':reference_us,
            'target_exposure_us':target_us,'timestamp_position':timestamp_position,
            'phase_transfer':'rectangular integration sinc and measured timestamp convention'}


def fit_light_phase(descriptions, expected_frequency_hz=120.0, search_width_hz=0.5):
    if len(descriptions) < 32:
        return None
    epoch = int(descriptions[0]["camera_timestamp_ns"])
    t = np.array([(int(d["camera_timestamp_ns"]) - epoch) / 1e9 for d in descriptions])
    y = np.array([d["mean_dn"] for d in descriptions])
    if not np.all(np.isfinite(y)) or np.ptp(t) < 0.2 or np.var(y) < 1e-8:
        return None
    centered = y - y.mean()

    def evaluate(frequency):
        angle = t * (2 * np.pi * frequency)
        design = np.column_stack(
            [np.ones(len(t))]
            + [func(k * angle) for k in range(1, 4) for func in (np.cos, np.sin)]
        )
        coefficients = np.linalg.lstsq(design, centered, rcond=None)[0]
        error = float(np.mean((design @ coefficients - centered) ** 2))
        return error, coefficients

    grid = np.linspace(
        expected_frequency_hz - search_width_hz,
        expected_frequency_hz + search_width_hz,
        201,
    )
    best = min(grid, key=lambda f: evaluate(f)[0])
    fine = np.linspace(best - search_width_hz / 100, best + search_width_hz / 100, 201)
    frequency = min(fine, key=lambda f: evaluate(f)[0])
    error, coefficients = evaluate(frequency)
    r_squared = 1 - error / float(np.var(centered))
    angles = np.linspace(0, 2 * np.pi, 4096, endpoint=False)
    wave = (
        np.column_stack(
            [np.ones(len(angles))]
            + [func(k * angles) for k in range(1, 4) for func in (np.cos, np.sin)]
        )
        @ coefficients
    )
    contrast = float(np.ptp(wave))
    noise = float(
        np.median(
            [
                np.sqrt(np.mean(np.asarray(d["sem_dn"]) ** 2))
                + d.get("dark_common_mean_sigma_dn", 0)
                for d in descriptions
            ]
        )
    )
    if r_squared < 0.85 or contrast < max(8 * noise, abs(float(y.mean())) * 0.001):
        return None
    phase = float(angles[int(np.argmax(wave))])
    period = 1e9 / frequency
    return {
        "epoch_ns": epoch,
        "period_ns": period,
        "phase_ns": period * phase / (2 * np.pi),
        "frequency_hz": float(frequency),
        "fit_r_squared": r_squared,
        "guard_contrast_dn": contrast,
        "pilot_frames": len(descriptions),
        "harmonic_coefficients": coefficients[1:].tolist(),
        "last_observed_camera_timestamp_ns": int(
            descriptions[-1]["camera_timestamp_ns"]
        ),
        "method": "camera timestamp / disjoint guard / three harmonics; peak illumination",
    }


def fit_phase_servo(descriptions, model):
    """Track the light peak using independent off-peak guard probes.

    Actual camera timestamps, rather than nominal trigger times, account for
    host scheduling jitter. Amplitude, offset and slow intensity drift remain
    separate nuisance parameters; no sensor/ROI residual is used here.
    """
    coefficients = np.asarray(model.get("harmonic_coefficients", []))
    if len(descriptions) < 8 or coefficients.shape != (6,):
        return None
    stamps = [int(d["camera_timestamp_ns"]) for d in descriptions]
    epoch = (stamps[0] + stamps[-1]) // 2
    times = np.asarray([(t - epoch) / 1e9 for t in stamps])
    angles = np.asarray([(t - model["epoch_ns"]) / 1e9 for t in stamps]) * (
        2 * np.pi * model["frequency_hz"]
    )
    values = np.asarray([d["mean_dn"] for d in descriptions])
    if np.ptp(times) <= 0 or np.var(values) <= 1e-8:
        return None

    def evaluate(shift):
        angle = angles - shift
        waveform = (
            np.column_stack(
                [f(k * angle) for k in range(1, 4) for f in (np.cos, np.sin)]
            )
            @ coefficients
        )
        design = np.column_stack([waveform, np.ones(len(times)), times])
        fit = np.linalg.lstsq(design, values, rcond=None)[0]
        error = float(np.mean((design @ fit - values) ** 2))
        return error if 0.25 <= fit[0] <= 4 else float("inf"), fit

    grid = np.linspace(-np.pi, np.pi, 401)
    best = min(grid, key=lambda shift: evaluate(shift)[0])
    fine = np.linspace(best - np.pi / 200, best + np.pi / 200, 101)
    shift = min(fine, key=lambda phi: evaluate(phi)[0])
    error, fit = evaluate(shift)
    r_squared = 1 - error / float(np.var(values))
    if not np.isfinite(error) or r_squared < 0.85:
        return None
    # A first-order phase/frequency loop corrects changing mains/PWM phase.
    elapsed = (
        epoch - model.get("last_observed_camera_timestamp_ns", model["epoch_ns"])
    ) / 1e9
    change = -0.5 * shift / (2 * np.pi * elapsed) if elapsed > 0.2 else 0.0
    change = float(np.clip(change, -0.05, 0.05))
    frequency = model["frequency_hz"] + change
    period = 1e9 / frequency
    rotation = (epoch - model["epoch_ns"]) / 1e9 * (
        2 * np.pi * model["frequency_hz"]
    ) - shift
    rotated = []
    for k in range(1, 4):
        c, s = coefficients[(k - 1) * 2 : k * 2]
        angle = k * rotation
        rotated.extend(
            [
                float((c * np.cos(angle) + s * np.sin(angle)) * fit[0]),
                float((-c * np.sin(angle) + s * np.cos(angle)) * fit[0]),
            ]
        )
    peak = model["phase_ns"] / model["period_ns"] * 2 * np.pi - rotation
    return {
        **model,
        "epoch_ns": epoch,
        "frequency_hz": frequency,
        "period_ns": period,
        "phase_ns": float((peak % (2 * np.pi)) / (2 * np.pi) * period),
        "harmonic_coefficients": rotated,
        "last_observed_camera_timestamp_ns": epoch,
        "servo_phase_shift_ns": float(shift / (2 * np.pi) * model["period_ns"]),
        "servo_fit_r_squared": r_squared,
        "servo_frequency_change_hz": change,
        "servo_probe_frames": len(descriptions),
    }


def phase_target_ns(now_ns, model, lead_ns=2000000, phase_fraction=None):
    offset = (
        model["phase_ns"]
        if phase_fraction is None
        else model["period_ns"] * phase_fraction
    )
    remaining = now_ns + lead_ns - model["epoch_ns"] - offset
    cycles = math.ceil(remaining / model["period_ns"])
    return model["epoch_ns"] + round(cycles * model["period_ns"] + offset)
