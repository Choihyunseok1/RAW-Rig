"""Bounded, group-atomic illumination checks for noise stacks.

Pilot images never enter the sensor-noise statistics. Where possible the
illumination monitor uses pixels outside the analysis ROI, so accepting a frame
does not select the noise residuals of the pixels being characterized.
"""

from __future__ import annotations

import math
from copy import deepcopy
import random
import time

import numpy as np

from .models import CaptureError, check_cancel

PILOT_GROUPS = 32
MAX_PILOT_GROUPS = 96
MAX_ATTEMPT_FACTOR = 30
NEAR_WHITE_FRACTION = 0.995


def describe_light(frame, roi, dark_mean):
    from .noise import channel_slices

    pixels = frame.pixels
    x, y, w, h = roi
    height, width = pixels.shape
    # Disjoint, even-aligned rectangles retain CFA parity. No analyzed pixel is
    # used when any sufficiently large surrounding region is available.
    # Monitor the nearby flat field, not unrelated objects/window shadows at
    # the image edges. The ring remains disjoint from every analyzed pixel.
    margin = max(8, min(w, h) // 2 // 2 * 2)
    left, top = max(0, x - margin), max(0, y - margin)
    right, bottom = min(width, x + w + margin), min(height, y + h + margin)
    rectangles = [
        (left, top, right - left, y - top),
        (left, y + h, right - left, bottom - y - h),
        (left, y, x - left, h),
        (x + w, y, right - x - w, h),
    ]
    rectangles = [r for r in rectangles if min(r[2:]) >= 8]
    independent = bool(rectangles)
    if not independent:
        half_w, half_h = w // 4 * 2, h // 4 * 2
        rectangles = [
            (x, y, half_w, half_h),
            (x + half_w, y, w - half_w, half_h),
            (x, y + half_h, half_w, h - half_h),
            (x + half_w, y + half_h, w - half_w, h - half_h),
        ]
    means, sems, labels, counts = [], [], [], []
    planes = channel_slices(frame.pixel_format)
    for index, (left, top, rw, rh) in enumerate(rectangles):
        patch = pixels[top : top + rh, left : left + rw]
        for channel, slices in planes.items():
            plane = patch[slices]
            stride = max(1, math.ceil(math.sqrt(plane.size / 4096)))
            sampled = plane[::stride, ::stride].astype(float)
            # Use every guard pixel for photometry. Subsampling is only used
            # for the conservative spatial noise estimate below. Fixed texture
            # and gradients must not inflate uncertainty as if only 4096 pixels
            # had been measured.
            means.append(float(plane.mean(dtype=np.float64)))
            counts.append(int(plane.size))
            # Same-colour spatial differences estimate a conservative frame-mean
            # noise allowance without using pilot temporal flicker as 'noise'.
            differences = np.diff(sampled, axis=1).ravel()
            sigma = (
                float(np.std(differences) / math.sqrt(2)) if len(differences) else 0.0
            )
            sems.append(sigma / math.sqrt(plane.size))
            labels.append(f"region_{index}_{channel}")
    region = pixels[y : y + h, x : x + w]
    white = frame.settings.get(
        "sensor_white_level", frame.settings.get("pixel_max", 65535)
    )
    return {
        "frame_id": frame.frame_id,
        "camera_timestamp_ns": frame.camera_timestamp_ns,
        "host_received_ns": frame.host_received_ns,
        "monitor_outside_roi": independent,
        "monitor_scope": "nearby disjoint ring" if independent else "ROI fallback",
        "labels": labels,
        "means_dn": means,
        "sem_dn": sems,
        "monitor_pixel_counts": counts,
        "mean_dn": float(np.average(means, weights=counts)),
        "dark_mean_dn": float(dark_mean),
        "roi_mean_dn": float(region.mean()),
        "roi_near_white_fraction": float(
            np.mean(region >= white * NEAR_WHITE_FRACTION)
        ),
        "white_dn": float(white),
        "crc_passed": frame.transport.get("crc_passed"),
        "exposure_us": frame.chunk.get('exposure_us',frame.settings.get('exposure_us')),
    }


def fit_light_model(pilot, max_flicker_percent):
    """Find a populated, bright, unsaturated plateau; never trust frame #1."""
    if len(pilot) < PILOT_GROUPS:
        raise CaptureError("조명 사전 검사 프레임이 부족합니다.")
    means = np.array([d["means_dn"] for d in pilot])
    sems = np.median([d["sem_dn"] for d in pilot], axis=0)
    counts = np.asarray(
        pilot[0].get("monitor_pixel_counts", np.ones(means.shape[1])), dtype=float
    )
    weights = counts / counts.sum()
    scalar_sem = float(np.sqrt(np.sum((weights * sems) ** 2)))
    common_sigma = float(pilot[0].get("dark_common_mean_sigma_dn", 0.0))
    scalar_sem = math.sqrt(scalar_sem**2 + common_sigma**2)
    scalar = means @ weights
    dark = pilot[0]["dark_mean_dn"]
    near_white = np.array([d["roi_near_white_fraction"] for d in pilot])
    saturated = near_white >= 0.9
    min_support = 6
    fraction = max_flicker_percent / 100
    if len(pilot) >= 2 * PILOT_GROUPS:
        tail = scalar[-PILOT_GROUPS:]
        tail_signal = max(float(np.median(tail)) - dark, 1.0)
        tail_excess = math.sqrt(max(float(np.var(tail, ddof=1)) - scalar_sem**2, 0))
        if (
            not saturated[-PILOT_GROUPS:].any()
            and np.min(tail) > dark + max(1.0, 6 * scalar_sem)
            and tail_excess <= fraction * tail_signal
        ):
            # Exposure transitions/daylight transients must not freeze a rare
            # bright reference. These additional pilot frames are independent
            # of every analyzed sample and remain excluded from noise statistics.
            model = fit_light_model(pilot[-PILOT_GROUPS:], max_flicker_percent)
            model["pilot_groups"] = len(pilot)
            model["reference_pilot_start_index"] = len(pilot) - PILOT_GROUPS
            model["selection"] = "stable final pilot window; frozen before acquisition"
            return model
    eligible = (~saturated) & (scalar > dark + max(1.0, 6 * scalar_sem))
    state = "illuminated"
    if (
        int(eligible.sum()) < min_support
        and not saturated.any()
        and np.median(scalar) <= dark + max(1.0, 6 * scalar_sem)
    ):
        # A legitimate short exposure may have no resolvable signal. Keep its
        # noise samples; light presence is checked separately at the longest
        # matched exposure before acquiring the sequence.
        eligible = np.ones(len(pilot), dtype=bool)
        state = "below_detection"
    if not eligible.any() and saturated.sum() >= min_support:
        eligible = saturated
        state = "saturated"

    def candidate_model(candidate, state_mask, stable_rms=False):
        multiplier = 3.0 if stable_rms else 0.8
        allowance = max(
            0.001,
            6 * scalar_sem,
            multiplier * fraction * max(scalar[candidate] - dark, 1.0),
        )
        cluster = state_mask & (np.abs(scalar - scalar[candidate]) <= allowance)
        if cluster.sum() < min_support:
            return None
        center = np.median(means[cluster], axis=0)
        bound = np.maximum(
            1.0,
            np.maximum(
                6 * np.sqrt(sems**2 + common_sigma**2),
                (3 if stable_rms else 2) * fraction * np.maximum(center - dark, 1),
            ),
        )
        stable = cluster & np.all(np.abs(means - center) <= bound, axis=1)
        if stable.sum() < min_support:
            return None
        center = np.median(means[stable], axis=0)
        scalar_center = float(np.median(scalar[stable]))
        scalar_bound = max(
            0.001,
            6 * scalar_sem,
            multiplier * fraction * max(scalar_center - dark, 1.0),
        )
        pooled = np.abs(scalar - scalar_center) <= scalar_bound
        if int(np.count_nonzero(stable & pooled)) < min_support:
            return None
        return {
            "schema_version": 3,
            "acceptance_policy": "stable RMS; three-sigma outlier limit"
            if stable_rms
            else "unstable illumination; narrow plateau",
            "state": state,
            "monitor_outside_roi": pilot[0]["monitor_outside_roi"],
            "labels": pilot[0]["labels"],
            "center_dn": center.tolist(),
            "bound_dn": bound.tolist(),
            "monitor_pixel_counts": counts.astype(int).tolist(),
            "scalar_center_dn": scalar_center,
            "scalar_bound_dn": scalar_bound if state != "below_detection" else None,
            "scalar_sem_dn": scalar_sem,
            "definition": "full-resolution outside-ROI CFA means; pooled illumination and regional banding checks",
            "pilot_groups": len(pilot),
            "pilot_plateau_groups": int(np.count_nonzero(stable & pooled)),
            "pilot_saturated_groups": int(saturated.sum()),
            "max_flicker_percent": max_flicker_percent,
            "near_white_threshold_fraction": NEAR_WHITE_FRACTION,
            "selection": "brightest populated state; densest stable window; frozen before acquisition",
        }

    ordered = np.flatnonzero(eligible)[np.argsort(scalar[eligible])]
    splits = []
    for j in range(1, len(ordered)):
        low, high = scalar[ordered[j - 1]], scalar[ordered[j]]
        # Preserve actual on/off illumination states, while treating small
        # continuous LED ripple as one state. Within it choose the most
        # populated window rather than a rare bright tail that never fills.
        gap_limit = max(0.01, 12 * scalar_sem, 8 * fraction * max(high - dark, 1))
        if high - low > gap_limit:
            splits.append(j)
    for indices in reversed(np.split(ordered, splits)):
        if len(indices) < min_support:
            continue
        mask = np.zeros(len(pilot), dtype=bool)
        mask[indices] = True
        signal = max(float(np.median(scalar[indices])) - dark, 1.0)
        excess = math.sqrt(
            max(float(np.var(scalar[indices], ddof=1)) - scalar_sem**2, 0)
        )
        stable_rms = state == "illuminated" and excess <= fraction * signal
        candidates = [
            m
            for i in indices
            if (m := candidate_model(i, mask, stable_rms)) is not None
        ]
        if candidates:
            return max(
                candidates,
                key=lambda m: (m["pilot_plateau_groups"], m["scalar_center_dn"]),
            )
    raise CaptureError(
        "안정된 밝은 구간을 찾지 못했습니다. 장면 밝기와 LED 플리커를 확인하고 "
        "②를 다시 누르세요. 차광 자료는 유지됩니다."
    )


def fit_drift_light_model(reference, max_flicker_percent):
    """Predict slow guard-field drift from independent calibration images only."""
    model = fit_light_model(reference, max_flicker_percent)
    if model["state"] != "illuminated":
        return model
    epoch = int(reference[-1]["camera_timestamp_ns"])
    reference = [d for d in reference if not light_rejection(d, model)]
    if len(reference) < 6:
        return model
    times = np.array([(int(d["camera_timestamp_ns"]) - epoch) / 1e9 for d in reference])
    if np.ptp(times) <= 0:
        return model
    values = np.asarray([d["means_dn"] for d in reference])
    coefficients = np.linalg.lstsq(
        np.column_stack([np.ones(len(times)), times]), values, rcond=None
    )[0]
    weights = np.asarray(model["monitor_pixel_counts"], dtype=float)
    weights /= weights.sum()
    scalar = values @ weights
    residual = scalar - (np.column_stack([np.ones(len(times)), times]) @ coefficients) @ weights
    sigma = max(model['scalar_sem_dn'],float(np.std(residual,ddof=2)))
    slope_se = sigma / math.sqrt(float(np.sum((times-times.mean())**2)))
    scalar_slope = float(coefficients[1] @ weights)
    model['drift_fit'] = {'estimated_scalar_slope_dn_per_second':scalar_slope,
                          'slope_standard_error_dn_per_second':slope_se,
                          'significance_sigma':abs(scalar_slope)/max(slope_se,1e-12)}
    if abs(scalar_slope) <= 4*slope_se:
        # Extrapolating an insignificant near-dark slope turns sensor noise
        # into a fictitious light drift and then rejects legitimate frames.
        model['drift_fit']['applied'] = False
        return model
    model['drift_fit']['applied'] = True
    model.update(
        schema_version=4,
        center_dn=coefficients[0].tolist(),
        scalar_center_dn=float(coefficients[0] @ weights),
        acceptance_policy="independent local linear time reference; Eq.8 frame pairs",
        temporal_reference={
            "epoch_ns": epoch,
            "slopes_dn_per_second": coefficients[1].tolist(),
            "scalar_slope_dn_per_second": float(coefficients[1] @ weights),
            "max_prediction_seconds": 20.0,
        },
    )
    return model


def light_rejection(description, model):
    legacy = model.get("schema_version", 1) == 1
    if description["labels"] != model["labels"]:
        return "조명 검사 영역 변경"
    if (
        model["state"] == "illuminated"
        and description["roi_near_white_fraction"] >= 0.9
    ):
        return "포화 프레임"
    if model["state"] == "saturated" and description["roi_near_white_fraction"] < 0.9:
        return "포화 단계의 조명 변동"
    values = np.asarray(description["means_dn"])
    center, bound = np.asarray(model["center_dn"]), np.asarray(model["bound_dn"])
    scalar_center = model.get("scalar_center_dn")
    if "temporal_reference" in model:
        temporal = model["temporal_reference"]
        dt = (int(description["camera_timestamp_ns"]) - temporal["epoch_ns"]) / 1e9
        if dt < 0 or dt > temporal["max_prediction_seconds"]:
            return "조명 기준 시간 범위 밖"
        center = center + np.asarray(temporal["slopes_dn_per_second"]) * dt
        scalar_center += temporal["scalar_slope_dn_per_second"] * dt
    if np.any(values < center - bound):
        return (
            "LED 소등·밝기 저하·부분 밴딩"
            if legacy
            else "조명 소등·밝기 저하·부분 밴딩"
        )
    if np.any(values > center + bound):
        return "밝기 증가·부분 밴딩"
    if model.get("schema_version", 1) >= 2 and model.get("scalar_bound_dn") is not None:
        if description.get("monitor_pixel_counts") != model["monitor_pixel_counts"]:
            return "조명 검사 픽셀 수 변경"
        delta = description["mean_dn"] - scalar_center
        if delta < -model["scalar_bound_dn"]:
            return "전체 밝기 저하·조명 변동"
        if delta > model["scalar_bound_dn"]:
            return "전체 밝기 증가·조명 변동"
    return ""


def excess_flicker(stats, dark_mean, confidence_margin=False, dark_stats=None):
    signal = float(stats["mean"].mean()) - dark_mean
    expected = float(stats["variance"].mean()) / stats["mean"].size
    dark_variance = 0.0
    if dark_stats is not None:
        dark_variance = float(np.var(dark_stats["frame_means"], ddof=1))
        dark_independent = (
            float(dark_stats["variance"].mean()) / dark_stats["mean"].size
        )
        expected += max(dark_variance - dark_independent, 0.0)
    variance = float(np.var(stats["frame_means"], ddof=1))
    if confidence_margin:
        # A finite stack of ordinary shot noise also has a fluctuating estimate
        # of its frame-mean variance. Require a four-standard-error excess before
        # declaring a whole stack unstable; the unadjusted estimate is still saved.
        expected += 4 * expected * math.sqrt(2 / (int(stats["count"]) - 1))
        if dark_stats is not None:
            expected += (
                4 * dark_variance * math.sqrt(2 / (int(dark_stats["count"]) - 1))
            )
    return math.sqrt(max(variance - expected, 0)) / max(signal, 1.0)


def excess_pair_flicker(stats, dark_mean, confidence_margin=False, dark_stats=None):
    """Eq.8 pair differences test rapid illumination changes, not slow drift.

    No RAW sample is changed or detrended. The dark pair means account for real
    common-mode sensor noise, which is shared across independently read pixels.
    """
    means = np.asarray(stats["frame_means"])
    n = len(means) // 2
    if n < 2 or dark_stats is None:
        return excess_flicker(stats, dark_mean, confidence_margin, dark_stats)
    pairs = (means[: 2 * n : 2] - means[1 : 2 * n : 2]) / math.sqrt(2)
    dark = np.asarray(dark_stats["frame_means"])
    nd = len(dark) // 2
    if nd < 2:
        return excess_flicker(stats, dark_mean, confidence_margin, dark_stats)
    dark_pairs = (dark[: 2 * nd : 2] - dark[1 : 2 * nd : 2]) / math.sqrt(2)
    dark_variance = float(np.mean(dark_pairs**2))
    expected = float(stats["pair_variance"].mean()) / stats["mean"].size
    expected += max(
        dark_variance
        - float(dark_stats["pair_variance"].mean()) / dark_stats["mean"].size,
        0.0,
    )
    if confidence_margin:
        expected += 4 * expected * math.sqrt(2 / max(n - 1, 1))
        expected += 4 * dark_variance * math.sqrt(2 / max(nd - 1, 1))
    signal = float(stats["mean"].mean()) - dark_mean
    return math.sqrt(max(float(np.mean(pairs**2)) - expected, 0.0)) / max(signal, 1.0)


def pair_noise_bias_upper(stats, dark_stats, planes):
    """Conservative four-SEM bound on common LED variance / pixel noise variance.

    Uses untouched disjoint RAW pairs and matched dark. CFA scaling assumes the
    fixed LED spectrum; independent colour guards still reject spectral changes.
    """
    means = np.asarray(stats["frame_means"])
    dark = np.asarray(dark_stats["frame_means"])
    n, nd = len(means) // 2, len(dark) // 2
    if min(n, nd) < 2:
        return math.inf
    pairs = (means[:2*n:2] - means[1:2*n:2]) / math.sqrt(2)
    dp = (dark[:2*nd:2] - dark[1:2*nd:2]) / math.sqrt(2)
    observed, dark_var = float(np.mean(pairs**2)), float(np.mean(dp**2))
    expected = float(stats["pair_variance"].mean()) / stats["mean"].size
    expected += max(dark_var - float(dark_stats["pair_variance"].mean()) / dark_stats["mean"].size, 0)
    upper = max(observed - expected, 0) + 4 * observed * math.sqrt(2 / (n-1))
    upper += 4 * dark_var * math.sqrt(2 / (nd-1))
    signal = stats["mean"] - dark_stats["mean"]
    average = max(float(signal.mean()), 1)
    return max(
        upper * (float(signal[z].mean()) / average)**2
        / max(float(stats["pair_variance"][z].mean()), 1)
        for z in planes.values()
    )


def lit_groups(
    capture,
    describe,
    count,
    max_flicker_percent,
    cancel,
    audit,
    progress,
    preview=None,
    lowest_step=False,
    simulated=False,
    phase_control=None,
    phase_seed=None,
    local_pair_protocol=False,
    fast_stack=False,
    phase_reference_capture=None,
):
    """Yield only accepted whole-camera groups; retain a bounded audit of all trials."""
    audit.update(
        schema_version=3 if local_pair_protocol else 2 if phase_control else 1,
        required_groups=count,
        accepted_groups=0,
        rejected_groups=0,
        pilot_groups=PILOT_GROUPS,
        max_attempts=max(256, count * MAX_ATTEMPT_FACTOR),
        timeout_seconds=max(180, count * 2 + 400),
        models={},
        observations=[],
        complete=False,
        phase_probe_groups=0,
        phase_models=[],
        phase_seed=phase_seed,
        phase_reference_groups=0,
        model_segments=[],
        unpaired_groups=0,
        phase_servo_groups=0,
        quality_method="disjoint adjacent frame differences; Eq.8"
        if local_pair_protocol
        else "whole-stack temporal stability",
    )
    started = time.monotonic()
    rng = random.Random()
    reference_phase_model=phase_seed.get('reference_model') if phase_seed else None

    def transfer_phase(model):
        if reference_phase_model is None:
            return model
        from .light_phase import phase_for_exposure
        return phase_for_exposure(model,phase_seed['reference_exposure_us'],
                                  phase_seed['target_exposure_us'],phase_seed['timestamp_position'])

    def trial(kind):
        check_cancel(cancel)
        if (
            kind != "pilot"
            and sum(
                audit[k]
                for k in (
                    "accepted_groups",
                    "rejected_groups",
                    "phase_probe_groups",
                    "phase_reference_groups",
                    "unpaired_groups",
                    "phase_servo_groups",
                )
            )
            >= audit["max_attempts"]
        ):
            raise CaptureError(
                "조명 검사·위상 측정·보충 촬영 한도에 도달했습니다. 차광 자료는 유지됩니다."
            )
        if not simulated:
            # Random trigger spacing avoids repeatedly sampling the same mains/PWM phase.
            # ExposureTime, Gain, RAW processing and the dark match stay unchanged.
            cancel.wait(rng.uniform(0, 0.019))
            check_cancel(cancel)
        if time.monotonic() - started > audit["timeout_seconds"]:
            raise CaptureError(
                "조명 검사·자동 보충 촬영 시간 초과. 광원을 안정시키고 ②를 다시 누르세요. 차광 자료는 유지됩니다."
            )
        frames, trigger = (phase_reference_capture() if phase_reference_capture and
                           reference_phase_model is not None and kind in ('phase_probe','phase_servo') else capture())
        descriptions = describe(frames)
        entry = {"kind": kind, "cameras": descriptions, "reasons": {}}
        if local_pair_protocol:
            entry["model_segment"] = max(0, len(audit["model_segments"]) - 1)
        audit["observations"].append(entry)
        if preview and (len(audit["observations"]) % 8 == 1):
            preview(frames)
        return frames, trigger, entry

    for target in range(PILOT_GROUPS, MAX_PILOT_GROUPS + 1, PILOT_GROUPS):
        while len(audit["observations"]) < target:
            trial("pilot")
            audit["pilot_groups"] = len(audit["observations"])
            progress(
                f"조명 사전 검사 {audit['pilot_groups']}/{target} · 분석용 장수에 포함하지 않음"
            )
        models = {}
        try:
            for serial in audit["observations"][0]["cameras"]:
                pilot = [entry["cameras"][serial] for entry in audit["observations"]]
                models[serial] = fit_light_model(pilot, max_flicker_percent)
        except CaptureError:
            if target == MAX_PILOT_GROUPS:
                raise
            progress("안정된 밝은 구간을 더 찾습니다 · 조명 검사 자동 연장")
            continue
        audit["models"] = models
        if not (fast_stack and phase_seed) and target < MAX_PILOT_GROUPS and any(
            m.get("state") == "illuminated"
            and m.get("acceptance_policy") == "unstable illumination; narrow plateau"
            for m in models.values()
        ):
            progress("노출시간 변경 후 조명 안정 확인 · 사전 검사 자동 연장")
            continue
        for serial, model in models.items():
            if lowest_step and model["state"] == "saturated":
                raise CaptureError(
                    f"{serial}: 가장 짧은 노출시간에서도 화면이 포화됩니다. 조명을 약하게 하거나 렌즈 조리개를 조이고 ②를 다시 누르세요. 차광 자료는 유지됩니다."
                )
        break
    phase_model = phase_seed["model"] if phase_seed else None
    phase_descriptions = phase_seed["descriptions"] if phase_seed else []
    if phase_model:
        from .light_phase import fit_light_phase

        serial = phase_seed["serial"]
        audit["phase_models"].append(phase_model)
    if phase_control and not phase_model and len(audit["models"]) == 1:
        from .light_phase import fit_light_phase

        serial = next(iter(audit["models"]))
        phase_descriptions = [
            e["cameras"][serial] for e in audit["observations"] if e["kind"] == "pilot"
        ]
        phase_model = fit_light_phase(phase_descriptions)
        if phase_model and phase_control(phase_model):
            audit["phase_models"].append(phase_model)
            progress("LED 위상 확인 · 안정된 시점으로 소프트웨어 트리거 조절")
            reference = []
            for _ in range(PILOT_GROUPS):
                _, _, entry = trial("phase_probe")
                audit["phase_probe_groups"] += 1
                reference.append(entry["cameras"][serial])
            audit["models"][serial] = fit_light_model(reference, max_flicker_percent)
            audit["models"][serial]["reference_kind"] = (
                "separate phase-locked pilot; excluded from analysis"
            )
        else:
            phase_model = None
    active_models = audit["models"]
    if local_pair_protocol:
        audit["model_segments"].append(
            {"models": active_models, "reference_kind": "pilot"}
        )
    probe_interval = 200 if fast_stack else (80 if local_pair_protocol else 200)
    audit["probe_interval"] = probe_interval
    next_probe = probe_interval
    next_servo = 16
    pending_pair = None

    def discard_pending(reason):
        nonlocal pending_pair
        if pending_pair is not None:
            pending_pair[2]["kind"] = "unpaired"
            pending_pair[2]["sampling_reason"] = reason
            audit["unpaired_groups"] += 1
            pending_pair = None

    consecutive_rejected = 0
    for attempt in range(audit["max_attempts"]):
        if (phase_model or local_pair_protocol) and attempt >= next_probe:
            discard_pending("independent calibration boundary")
            if phase_model:
                probes = []
                for j in range(PILOT_GROUPS):
                    phase_control(reference_phase_model or phase_model, j / PILOT_GROUPS)
                    _, _, entry = trial("phase_probe")
                    audit["phase_probe_groups"] += 1
                    probes.append(entry["cameras"][serial])
                updated = fit_light_phase(
                    probes
                    if local_pair_protocol
                    else phase_descriptions[-32:] + probes,
                    phase_model["frequency_hz"],
                    0.1 if local_pair_protocol else 0.025,
                )
                if updated:
                    if reference_phase_model is not None:
                        reference_phase_model=updated
                    phase_model = transfer_phase(updated)
                    audit["phase_models"].append(updated)
                phase_descriptions = probes
                phase_control(phase_model)
            if local_pair_protocol:
                reference = []
                for _ in range(PILOT_GROUPS + 8):
                    _, _, entry = trial("phase_reference")
                    audit["phase_reference_groups"] += 1
                    reference.append(entry["cameras"])
                for reference_retry in range(3):
                    try:
                        refreshed_models = {
                            s: fit_drift_light_model(
                                [d[s] for d in reference[-PILOT_GROUPS:]], max_flicker_percent
                            )
                            for s in active_models
                        }
                        break
                    except CaptureError:
                        if reference_retry == 2:
                            raise
                        progress("독립 조명 기준 재측정 · 정상 RAW와 완료 단계는 유지")
                        for _ in range(PILOT_GROUPS):
                            _, _, entry = trial("phase_reference")
                            audit["phase_reference_groups"] += 1
                            reference.append(entry["cameras"])
                active_models = refreshed_models
                audit["model_segments"].append(
                    {
                        "models": active_models,
                        "reference_kind": "independent phase-locked reference; excluded from RAW statistics",
                        "reference_frame_ids": {
                            s: [d[s]["frame_id"] for d in reference[-PILOT_GROUPS:]]
                            for s in active_models
                        },
                    }
                )
            next_probe = attempt + probe_interval
        if (
            local_pair_protocol
            and phase_model
            and "harmonic_coefficients" in phase_model
            and attempt >= next_servo
        ):
            from .light_phase import fit_phase_servo

            discard_pending("independent calibration boundary")
            probe_descriptions = []
            servo_model=reference_phase_model or phase_model
            peak_fraction = servo_model["phase_ns"] / servo_model["period_ns"]
            for offset in (-0.1, 0, 0.1, 0, 0.1, 0, -0.1, 0):
                phase_control(servo_model, (peak_fraction + offset) % 1)
                _, _, entry = trial("phase_servo")
                audit["phase_servo_groups"] += 1
                probe_descriptions.append(entry["cameras"][serial])
            updated = fit_phase_servo(probe_descriptions, servo_model)
            if updated:
                if reference_phase_model is not None:
                    reference_phase_model=updated
                phase_model = transfer_phase(updated)
                audit["phase_models"].append(updated)
            phase_control(phase_model)
            center_reference = []
            for _ in range(4):
                _, _, entry = trial("phase_reference")
                audit["phase_reference_groups"] += 1
                center_reference.append(entry["cameras"][serial])
            model = deepcopy(active_models[serial])
            center = np.mean([d["means_dn"] for d in center_reference], axis=0)
            epoch = sum(int(d["camera_timestamp_ns"]) for d in center_reference) // 4
            slopes = np.zeros_like(center)
            previous_peak = model.get("peak_reference")
            if previous_peak and epoch > previous_peak["epoch_ns"]:
                dt = (epoch - previous_peak["epoch_ns"]) / 1e9
                slopes = (center - np.asarray(previous_peak["means_dn"])) / dt
            weights = np.asarray(model["monitor_pixel_counts"], dtype=float)
            weights /= weights.sum()
            if previous_peak and epoch > previous_peak['epoch_ns'] and abs(float(slopes @ weights)) <= 4*math.sqrt(2)*model['scalar_sem_dn']/2/dt:
                slopes=np.zeros_like(center)
            presence_descriptions = probe_descriptions if reference_phase_model is not None else center_reference
            presence = np.median([float(np.asarray(d["means_dn"]) @ weights) - d["dark_mean_dn"] for d in presence_descriptions])
            # Short target samples may intentionally be below read-noise detection.
            # Verify light using the independent longer reference, never discard
            # such targets merely because their mean is close to the dark mean.
            if presence <= max(1.0, 6 * model["scalar_sem_dn"]):
                raise CaptureError(
                    "위상 추적 중 암영상 수준의 조명이 확인됐습니다. 차광·통과한 단계는 유지됩니다."
                )
            model.update(
                schema_version=4,
                center_dn=center.tolist(),
                scalar_center_dn=float(center @ weights),
                peak_reference={"epoch_ns": epoch, "means_dn": center.tolist()},
                temporal_reference={
                    "epoch_ns": epoch,
                    "slopes_dn_per_second": slopes.tolist(),
                    "scalar_slope_dn_per_second": float(slopes @ weights),
                    "max_prediction_seconds": 20.0,
                },
            )
            active_models = {serial: model}
            audit["model_segments"].append(
                {
                    "models": active_models,
                    "reference_kind": "independent servo center; excluded from RAW statistics",
                    "reference_frame_ids": {
                        serial: [d["frame_id"] for d in center_reference]
                    },
                }
            )
            next_servo = attempt + 16
        frames, trigger, entry = trial("candidate")
        reasons = {
            s: reason
            for s, model in active_models.items()
            if (reason := light_rejection(entry["cameras"][s], model))
        }
        entry["reasons"] = reasons
        if reasons:
            discard_pending("partner photometry rejected")
            entry["kind"] = "rejected"
            audit["rejected_groups"] += 1
            consecutive_rejected += 1
            progress(
                f"조명 이상 제외 {audit['rejected_groups']}회 ({'; '.join(sorted(set(reasons.values())))}) · 정상 {audit['accepted_groups']}/{count} · 자동 보충 중"
            )
            if consecutive_rejected >= 256:
                break
            continue
        consecutive_rejected = 0
        if local_pair_protocol and count - audit["accepted_groups"] >= 2:
            if pending_pair is None:
                entry["kind"] = "pending"
                pending_pair = (frames, trigger, entry)
                continue
            first_frames, first_trigger, first_entry = pending_pair
            pending_pair = None
            first_entry["kind"] = "accepted"
            first_entry["accepted_index"] = audit["accepted_groups"]
            audit["accepted_groups"] += 1
            entry["kind"] = "accepted"
            entry["accepted_index"] = audit["accepted_groups"]
            audit["accepted_groups"] += 1
            yield first_frames, first_trigger, first_entry["cameras"]
            yield frames, trigger, entry["cameras"]
            if audit["accepted_groups"] == count:
                return
            continue
        entry["kind"] = "accepted"
        entry["accepted_index"] = audit["accepted_groups"]
        audit["accepted_groups"] += 1
        yield frames, trigger, entry["cameras"]
        if audit["accepted_groups"] == count:
            return
    discard_pending("acquisition ended before partner")
    raise CaptureError(
        f"조명 밝기가 안정되지 않아 정상 {audit['accepted_groups']}/{count}장만 확보했습니다. 자동 보충 한도에 도달했습니다. 차광 자료는 유지됩니다."
    )
