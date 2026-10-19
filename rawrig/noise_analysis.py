"""Streaming-statistics PTC, pixel maps and reproducible scientific plots."""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
from threading import Event
from tempfile import TemporaryDirectory
import uuid

import numpy as np

from .models import CaptureError, check_cancel
from .noise import (
    NoiseConfig,
    PAPER_URL,
    channel_slices,
    save_npz,
    verify_phase,
    verify_noise_session,
)
from .storage import _safe_child, sha256_file, utc_now, write_json, fsync_dir
from .illumination import NEAR_WHITE_FRACTION, excess_pair_flicker, pair_noise_bias_upper

COLORS = {
    "R": "#cf5367",
    "G1": "#359b72",
    "G2": "#7eae50",
    "B": "#5975c8",
    "Mono": "#7762ad",
}


def _clean(value):
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _load(root, record, cache=None):
    with np.load(_safe_child(root, record["stats"]), allow_pickle=False) as archive:
        stats = {}
        for key in archive.files:
            values = archive[key]
            if cache is not None and values.nbytes > 1024**2:
                path = cache / (uuid.uuid4().hex + ".npy")
                np.save(path, values, allow_pickle=False)
                values = np.load(path, mmap_mode="r", allow_pickle=False)
            stats[key] = values
    stats["metadata"] = json.loads(_safe_child(root, record["metadata"]).read_text())
    stats["metadata"].pop("frames", None)
    meta = stats["metadata"]
    raw = np.memmap(
        _safe_child(root, record["raw"]),
        mode="r",
        dtype="<u2",
        shape=(meta["count"], *meta["stored_shape"]),
    )
    x, y, w, h = meta["roi"]
    stats["single_pixel_values"] = {}
    stats["single_pixel_roi_coordinates"] = {}
    for channel, slices in channel_slices(meta["settings"]["pixel_format"]).items():
        rows = np.arange(h)[slices[0]]
        columns = np.arange(w)[slices[1]]
        row, column = int(rows[len(rows) // 2]), int(columns[len(columns) // 2])
        stats["single_pixel_values"][channel] = np.array(
            raw[
                :,
                row + (y if meta["full_raw"] else 0),
                column + (x if meta["full_raw"] else 0),
            ],
            dtype=float,
        )
        stats["single_pixel_roi_coordinates"][channel] = [column, row]
    # Independent halves of whole frame pairs support the Eq.8 gain-map
    # precision estimate. Derive these from immutable RAW, including old dark
    # archives which did not save pair-half statistics.
    if "pair_half_variance" not in stats:
        pair_count = meta["count"] // 2
        half_count = np.array([(pair_count + 1) // 2, pair_count // 2])
        sums = np.zeros((2, h, w))
        means = np.zeros((2, h, w))
        block = max(1, min(16, 8 * 1024**2 // (h * w * 16)))
        for start in range(0, pair_count, block):
            stop = min(start + block, pair_count)
            pixels = raw[2 * start : 2 * stop]
            if meta["full_raw"]:
                pixels = pixels[:, y : y + h, x : x + w]
            pixels = pixels.astype(float)
            difference = pixels[::2] - pixels[1::2]
            average = (pixels[::2] + pixels[1::2]) / 2
            for half in (0, 1):
                offset = (half - start) % 2
                sums[half] += np.sum(difference[offset::2] ** 2, axis=0) / 2
                means[half] += np.sum(average[offset::2], axis=0)
        stats["pair_half_count"] = half_count
        stats["pair_half_variance"] = sums / np.maximum(half_count[:, None, None], 1)
        stats["pair_half_mean"] = means / np.maximum(half_count[:, None, None], 1)
    del raw
    if stats["metadata"]["phase"] == "bright":
        # The camera can clamp one or two ADC codes below its advertised white
        # level. Inspect original RAW, including old sessions whose exact-white
        # clipping counter was zero despite an entirely saturated image.
        meta = stats["metadata"]
        white = meta["settings"].get(
            "sensor_white_level", meta["settings"].get("pixel_max", 65535)
        )
        counts = np.zeros(stats["mean"].shape, dtype=np.int64)
        raw = np.memmap(
            _safe_child(root, record["raw"]),
            mode="r",
            dtype="<u2",
            shape=(meta["count"], *meta["stored_shape"]),
        )
        block = max(1, min(32, 8 * 1024**2 // (math.prod(meta["stored_shape"]) * 2)))
        x, y, w, h = meta["roi"]
        for start in range(0, meta["count"], block):
            pixels = raw[start : start + block]
            if meta["full_raw"]:
                pixels = pixels[:, y : y + h, x : x + w]
            counts += np.count_nonzero(pixels >= white * NEAR_WHITE_FRACTION, axis=0)
        del raw
        stats["near_white"] = counts
    return stats


def fit_gain(signal, shot_variance, valid):
    x, y = np.asarray(signal)[valid], np.asarray(shot_variance)[valid]
    if len(x) < 3 or np.min(x) <= 0 or np.max(x) / np.min(x) < 3:
        return {
            "valid": False,
            "reason": "포화·조명 변동을 제외한 단계가 3개 미만이거나 신호 범위가 3배 미만입니다.",
            "points": len(x),
            "gain_dn_per_e": None,
            "e_per_dn": None,
            "r_squared": None,
        }
    gain = float(np.dot(x, y) / np.dot(x, x))
    error = float(np.sum((y - gain * x) ** 2))
    r2 = 1 - error / max(float(np.sum((y - y.mean()) ** 2)), 1e-30)
    valid_fit = gain > 0 and r2 >= 0.9
    return {
        "valid": valid_fit,
        "reason": ""
        if valid_fit
        else "선형 shot-noise 모델의 R²가 0.9 미만이거나 기울기가 양수가 아닙니다.",
        "points": len(x),
        "gain_dn_per_e": gain if valid_fit else None,
        "e_per_dn": 1 / gain if valid_fit else None,
        "r_squared": r2,
        "candidate_slope_dn_per_e": gain,
    }


def _noise_hist(values):
    values = np.asarray(values).ravel()
    values = values[np.isfinite(values)]
    if not len(values):
        return {"count": 0}
    mean, std = (
        float(values.mean()),
        float(values.std(ddof=1)) if len(values) > 1 else 0.0,
    )
    z = (values - mean) / max(std, 1e-12)
    return {
        "count": len(values),
        "mean": mean,
        "std": std,
        "p01": float(np.quantile(values, 0.01)),
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "skewness": float(np.mean(z**3)),
        "excess_kurtosis": float(np.mean(z**4) - 3),
        "outside_3sigma_fraction": float(np.mean(np.abs(z) > 3)),
    }


def _flicker(bright, dark):
    mean_signal = float((bright["mean"] - dark["mean"]).mean())
    trace = bright["frame_means"]
    expected_variance = float(bright["variance"].mean()) / bright["mean"].size
    dark_mean_variance = float(np.var(dark["frame_means"], ddof=1))
    dark_independent = float(dark["variance"].mean()) / dark["mean"].size
    expected_variance += max(dark_mean_variance - dark_independent, 0.0)
    excess = max(float(np.var(trace, ddof=1)) - expected_variance, 0.0)
    return math.sqrt(excess) / max(mean_signal, 1.0), float(
        np.std(trace, ddof=1)
    ) / max(mean_signal, 1.0)


def _dark_lower_rail(bias):
    settings = bias["metadata"]["settings"]
    samples = bias["sample_values"]
    floor = int(samples.min())
    fraction = float(np.mean(samples == floor))
    clamping = str(settings.get("black_level_clamping", "")).lower() in (
        "1",
        "true",
        "on",
    )
    suspect = clamping and settings.get("black_level_percent") == 0 and fraction > 0.5
    return {
        "suspected_clamping": suspect,
        "sampled_floor_dn": floor,
        "sampled_floor_fraction": fraction,
        "scope": "camera-delivered RAW after black-level processing and ADC quantization",
    }


def _pixel_gain_maps(brights, darks, point_valid, white, pair_flags=None):
    pair_flags = pair_flags or [False] * len(brights)
    shape = brights[0]["mean"].shape
    sums = np.zeros((2, *shape))
    half_sums = np.zeros((2, 2, *shape))
    support = np.zeros(shape, dtype=np.uint16)
    y_sum, y_square_sum = np.zeros(shape), np.zeros(shape)
    for b, d, usable, paired in zip(brights, darks, point_valid, pair_flags):
        x = b["mean"] - d["mean"]
        variance_key = "pair_variance" if paired else "variance"
        y = b[variance_key] - d[variance_key]
        mask = (
            usable
            & (x > white * 0.01)
            & (x < white * 0.8)
            & (b["clipped"] == 0)
            & (x > 3 * np.sqrt(d["variance"]))
        )
        sums[0] += np.where(mask, x * y, 0)
        sums[1] += np.where(mask, x * x, 0)
        support += mask
        y_sum += np.where(mask, y, 0)
        y_square_sum += np.where(mask, y * y, 0)
        for half in (0, 1):
            mk = "pair_half_mean" if paired else "half_mean"
            vk = "pair_half_variance" if paired else "half_variance"
            hx = b[mk][half] - d[mk][half]
            hy = b[vk][half] - d[vk][half]
            half_sums[half, 0] += np.where(mask, hx * hy, 0)
            half_sums[half, 1] += np.where(mask, hx * hx, 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        gain = sums[0] / sums[1]
        half_gain = half_sums[:, 0] / half_sums[:, 1]
        residual = y_square_sum - 2 * gain * sums[0] + gain * gain * sums[1]
        total_variation = y_square_sum - y_sum * y_sum / support
        r_squared = 1 - residual / total_variation
    invalid = (support < 3) | (gain <= 0) | ~np.isfinite(r_squared) | (r_squared < 0.9)
    gain[invalid] = np.nan
    half_gain[:, invalid] = np.nan
    propagated = np.zeros(shape)
    for b, d, usable, paired in zip(brights, darks, point_valid, pair_flags):
        x = b["mean"] - d["mean"]
        variance_key = "pair_variance" if paired else "variance"
        y = b[variance_key] - d[variance_key]
        mask = (
            usable
            & (x > white * 0.01)
            & (x < white * 0.8)
            & (b["clipped"] == 0)
            & (x > 3 * np.sqrt(d["variance"]))
        )
        divisor = 2 if paired else 1
        variance_error = 2 * b[variance_key] ** 2 / max(
            int(b["count"]) // divisor - 1, 1
        ) + 2 * d[variance_key] ** 2 / max(int(d["count"]) // divisor - 1, 1)
        mean_error = b["variance"] / int(b["count"]) + d["variance"] / int(d["count"])
        propagated += np.where(
            mask,
            x * x * variance_error
            + (y - 2 * np.nan_to_num(gain) * x) ** 2 * mean_error,
            0,
        )
    with np.errstate(divide="ignore", invalid="ignore"):
        se = np.sqrt(propagated) / sums[1]
    se[~np.isfinite(gain)] = np.nan
    return gain, se, half_gain, support, r_squared


def characterize(
    root,
    camera,
    gain_index,
    noise,
    dark_records,
    bright_records,
    cancel,
    cache=None,
    acquisition_plan=None,
):
    points = [
        p for p in (acquisition_plan or noise.plan()) if p["gain_index"] == gain_index
    ]
    darks = [
        _load(root, dark_records[p["index"], camera["serial"]], cache) for p in points
    ]
    brights = [
        _load(root, bright_records[p["index"], camera["serial"]], cache) for p in points
    ]
    check_cancel(cancel)
    settings = darks[0]["metadata"]["settings"]
    white = settings.get("sensor_white_level", settings.get("pixel_max", 65535))
    planes = channel_slices(settings["pixel_format"])
    quality = [_flicker(b, d) for b, d in zip(brights, darks)]
    pair_flags = []
    variance_budget_flags = []
    for point in points:
        record = bright_records[point["index"], camera["serial"]]
        audit = (
            json.loads(_safe_child(root, record["illumination_audit"]).read_text())
            if record.get("illumination_audit")
            else {}
        )
        variance_budget_flags.append(audit.get("noise_variance_bias_budget_fraction") == 0.001)
        pair_flags.append(
            audit.get("quality_method") == "disjoint adjacent frame differences; Eq.8"
        )
    pair_quality = [
        excess_pair_flicker(
            b, float(d["mean"].mean()), confidence_margin=True, dark_stats=d
        )
        for b, d in zip(brights, darks)
    ]
    point_valid = [
        ((p if paired else f) <= noise.max_flicker_percent / 100)
        or (budget and pair_noise_bias_upper(b, d, planes) <= 0.001)
        for (f, _), p, paired, budget, b, d in zip(
            quality, pair_quality, pair_flags, variance_budget_flags, brights, darks
        )
    ]
    variance_bias_bounds = [
        pair_noise_bias_upper(b, d, planes) if budget else None
        for b, d, budget in zip(brights, darks, variance_budget_flags)
    ]
    gain_map, gain_se, half_gain, support, pixel_r2 = _pixel_gain_maps(
        brights, darks, point_valid, white, pair_flags
    )
    bias = darks[0]
    lower_rail = _dark_lower_rail(bias)
    n = int(bias["count"])
    rn = np.sqrt(bias["variance"])
    maps = {
        "offset_dn": bias["mean"],
        "read_noise_dn": rn,
        "dark_frame_means": bias["frame_means"],
        "dark_timestamps_ns": bias["timestamps_ns"],
        "read_variance_dn2": bias["variance"],
        "gain_dn_per_e": gain_map,
        "gain_standard_error": gain_se,
        "gain_fit_points": support,
        "gain_fit_r_squared": pixel_r2,
        "high_noise_pixel_mask": rn > 3 * np.median(rn),
        "offset_standard_error_dn": rn / math.sqrt(n),
        "read_noise_standard_error_dn": rn / math.sqrt(2 * (n - 1)),
        "read_variance_standard_error_dn2": bias["variance"] * math.sqrt(2 / (n - 1)),
        "offset_half_difference_dn": bias["half_mean"][0] - bias["half_mean"][1],
        "read_noise_half_difference_dn": np.sqrt(bias["half_variance"][0])
        - np.sqrt(bias["half_variance"][1]),
        "gain_half_difference": half_gain[0] - half_gain[1],
    }
    result = {
        "serial": camera["serial"],
        "model": camera["model"],
        "simulated": camera["simulated"],
        "protocol": noise.protocol,
        "gain_db": settings["gain_db"],
        "roi": bias["metadata"]["roi"],
        "settings": settings,
        "channels": {},
        "warnings": [],
        "wavelength_nm": noise.wavelength_nm,
        "qe": noise.quantum_efficiency.get(camera["serial"]),
        "illumination_filter": [],
        "dark_lower_rail": lower_rail,
    }
    if any(pair_flags):
        result["warnings"].append(
            "노출시간 순회에서는 논문 Eq.8 프레임 차분으로 빠른 노이즈·플리커를 검사합니다. 독립 기준 프레임으로 위상·느린 광량 변화를 추적하며 RAW는 수정하지 않습니다. 단일 픽셀 밝은 RAW 시간 분포에는 느린 조명 변화도 포함됩니다."
        )
    for p, b in zip(points, brights):
        record = bright_records[p["index"], camera["serial"]]
        if "illumination_audit" in record:
            audit = json.loads(
                _safe_child(root, record["illumination_audit"]).read_text()
            )
            result["illumination_filter"].append(
                {
                    "point": p["index"],
                    "accepted_groups": audit["accepted_groups"],
                    "rejected_groups": audit["rejected_groups"],
                    "pilot_groups": audit["pilot_groups"],
                    "model": audit["models"][camera["serial"]],
                    "audit": record["illumination_audit"],
                }
            )
    if result["illumination_filter"]:
        rejected = sum(r["rejected_groups"] for r in result["illumination_filter"])
        result["warnings"].append(
            f"조명 사전 검사·고정 판정 후 플리커 {rejected}회 제외·자동 보충했습니다. 그래프는 선택된 안정 조명 구간의 조건부 측정입니다."
        )
        if any(
            not r["model"]["monitor_outside_roi"] for r in result["illumination_filter"]
        ):
            result["warnings"].append(
                "전체 센서/작은 영상에서는 분석 영역 자체의 평균으로 조명을 검사했습니다. 선택에 따른 통계 영향이 있을 수 있습니다."
            )
    if any(float(b["near_white"].mean()) / int(b["count"]) >= 0.005 for b in brights):
        result["warnings"].append(
            "RAW 상한의 99.5% 이상에 몰린 포화 의심 픽셀을 검출했습니다. 해당 밝기 단계는 게인 추정에서 제외합니다. 조명을 약하게 하거나 조리개를 조이고 ②로 밝은 촬영만 다시 할 수 있습니다."
        )
    if noise.frames < 100 or noise.bias_frames < 1000:
        result["warnings"].append(
            "반복 수가 논문의 PTC 100장 / 픽셀 맵 1000장보다 작아 정밀도가 낮습니다."
        )
    if noise.frames < 1000:
        result["warnings"].append(
            f"픽셀 게인 맵은 단계별 {noise.frames}장으로 추정했습니다. 논문 그림 6의 1000장 조건보다 불확도가 크며 정밀도 맵을 함께 확인하세요."
        )
    if float(bias["zero_fraction"]) > 0.01:
        result["warnings"].append(
            "암영상 0 DN 클리핑이 1%를 초과합니다. 읽기 노이즈가 과소평가될 수 있습니다."
        )
    if lower_rail["suspected_clamping"]:
        result["warnings"].append(
            f"차광 표본의 {lower_rail['sampled_floor_fraction'] * 100:.1f}%가 {lower_rail['sampled_floor_dn']} DN 하한에 몰려 있습니다. 블랙 레벨 클램핑으로 센서 읽기 노이즈가 과소평가될 수 있습니다. 표시 값은 카메라 출력 RAW의 노이즈입니다."
        )
    if not all(point_valid):
        result["warnings"].append(
            "허용치를 넘는 조명 변동이 관측된 밝기 단계는 변환 게인 추정에서 제외했습니다."
        )
    for channel, slices in planes.items():
        check_cancel(cancel)
        offset = bias["mean"][slices]
        read_var = float(bias["variance"][slices].mean())
        rows = []
        for p, b, d, (flicker, raw_fluctuation), usable, paired, bias_upper in zip(
            points, brights, darks, quality, point_valid, pair_flags, variance_bias_bounds
        ):
            signal_map = b["mean"][slices] - d["mean"][slices]
            signal = float(signal_map.mean())
            temporal = float(b["variance"][slices].mean())
            pair = float(b["pair_variance"][slices].mean())
            dark_pair = float(d["pair_variance"][slices].mean())
            finite_mean_noise = b["variance"][slices] / int(b["count"]) + d["variance"][
                slices
            ] / int(d["count"])
            fpn_variance = max(
                float(np.var(signal_map, ddof=1)) - float(finite_mean_noise.mean()), 0.0
            )
            column_variance = max(
                float(np.var(signal_map.mean(axis=0), ddof=1))
                - float(finite_mean_noise.mean()) / signal_map.shape[0],
                0.0,
            )
            row_variance = max(
                float(np.var(signal_map.mean(axis=1), ddof=1))
                - float(finite_mean_noise.mean()) / signal_map.shape[1],
                0.0,
            )
            clipping = float(np.mean(b["clipped"][slices]) / int(b["count"]))
            near_white = float(np.mean(b["near_white"][slices]) / int(b["count"]))
            shot_var = pair - dark_pair
            usable = (
                usable
                and signal > max(3 * math.sqrt(read_var), white * 0.01)
                and signal < white * 0.8
                and clipping < 0.001
                and near_white < 0.001
                and shot_var > 0
            )
            rows.append(
                {
                    "exposure_us": d["metadata"]["settings"]["exposure_us"],
                    "illumination_noise_variance_bias_upper_fraction": bias_upper,
                    "bright_frames": int(b["count"]),
                    "dark_frames": int(d["count"]),
                    "signal_dn": signal,
                    "read_noise_dn": math.sqrt(read_var),
                    "dark_noise_dn": math.sqrt(float(d["variance"][slices].mean())),
                    "temporal_noise_dn": math.sqrt(pair if paired else temporal),
                    "raw_temporal_noise_dn": math.sqrt(temporal),
                    "pair_noise_dn": math.sqrt(pair),
                    "shot_variance_dn2": shot_var,
                    "shot_noise_dn": math.sqrt(max(shot_var, 0)),
                    "fpn_dn": math.sqrt(fpn_variance),
                    "column_fpn_dn": math.sqrt(min(column_variance, fpn_variance)),
                    "row_fpn_dn": math.sqrt(row_variance),
                    "pixel_fpn_dn": math.sqrt(max(fpn_variance - column_variance, 0)),
                    "total_noise_dn": math.sqrt(
                        (pair if paired else temporal) + fpn_variance
                    ),
                    "uniformity_min_over_mean": float(signal_map.min())
                    / max(signal, 1.0),
                    "flat_cv": float(signal_map.std()) / max(signal, 1.0),
                    "clipped_fraction": clipping,
                    "near_white_fraction": near_white,
                    "excess_flicker_fraction": excess_pair_flicker(
                        b, float(d["mean"].mean()), dark_stats=d
                    )
                    if paired
                    else flicker,
                    "whole_stack_excess_flicker_fraction": flicker,
                    "illumination_quality_method": "Eq.8 pair differences; slow drift retained in RAW"
                    if paired
                    else "whole-stack temporal stability",
                    "frame_fluctuation_fraction": raw_fluctuation,
                    "fit_used": bool(usable),
                    "signal_standard_error_dn": math.sqrt(
                        temporal / (int(b["count"]) * signal_map.size)
                    ),
                }
            )
        x = np.array([r["signal_dn"] for r in rows])
        y = np.array([r["shot_variance_dn2"] for r in rows])
        fit = fit_gain(x, y, np.array([r["fit_used"] for r in rows]))
        usable_rows = np.array([r["fit_used"] for r in rows])
        time_us = np.array([r["exposure_us"] for r in rows])
        response_slope = (
            float(
                np.dot(time_us[usable_rows], x[usable_rows])
                / np.dot(time_us[usable_rows], time_us[usable_rows])
            )
            if usable_rows.any() and noise.protocol != "two_condition"
            else None
        )
        response_r2 = None
        if usable_rows.sum() >= 3:
            errors = x[usable_rows] - response_slope * time_us[usable_rows]
            response_r2 = 1 - float(np.sum(errors**2)) / max(
                float(np.sum((x[usable_rows] - x[usable_rows].mean()) ** 2)), 1e-30
            )
            if response_r2 < 0.99:
                fit.update(
                    valid=False,
                    reason="고정 광원에서 노출시간 응답이 비선형입니다. HDR·ISP·조명 변화를 확인하세요.",
                    gain_dn_per_e=None,
                    e_per_dn=None,
                )
        e_per_dn = fit["e_per_dn"]
        for row in rows:
            row["snr_temporal"] = row["signal_dn"] / max(
                row["temporal_noise_dn"], 1e-20
            )
            row["snr_total"] = row["signal_dn"] / max(row["total_noise_dn"], 1e-20)
            row["signal_e"] = row["signal_dn"] * e_per_dn if e_per_dn else None
            qe = result["qe"]
            row["signal_photons"] = row["signal_e"] / qe if e_per_dn and qe else None
            row["relative_snr_photon"] = (
                row["snr_total"] / math.sqrt(row["signal_photons"])
                if row["signal_photons"] and row["signal_photons"] > 0
                else None
            )
            row["relative_snr_electron"] = (
                row["snr_total"] / math.sqrt(row["signal_e"])
                if row["signal_e"] and row["signal_e"] > 0
                else None
            )
        saturation = next(
            (i for i, r in enumerate(rows) if r["clipped_fraction"] >= 0.005), None
        )
        effective_fwc = (
            x[saturation] * e_per_dn if saturation is not None and e_per_dn else None
        )
        rn_e = math.sqrt(read_var) * e_per_dn if e_per_dn else None
        prnu_rows = [r for r in rows if r["fit_used"]]
        prnu = (
            float(np.median([r["fpn_dn"] / r["signal_dn"] for r in prnu_rows]))
            if prnu_rows
            else None
        )
        dark_levels = np.array([d["mean"][slices].mean() for d in darks])
        exposures = time_us / 1e6
        dark_slope = dark_se = None
        dark_rate_valid = False
        if len(exposures) >= 3 and np.ptp(exposures) > 0:
            dark_slope, dark_intercept = np.polyfit(exposures, dark_levels, 1)
            dark_residual = dark_levels - (dark_slope * exposures + dark_intercept)
            dark_se = float(
                np.sqrt(
                    np.sum(dark_residual**2)
                    / (len(exposures) - 2)
                    / np.sum((exposures - exposures.mean()) ** 2)
                )
            )
            dark_rate_valid = dark_slope > 3 * dark_se and dark_slope > 0
        selected = next(
            (i for i in reversed(range(len(rows))) if rows[i]["fit_used"]),
            len(rows) // 2,
        )
        bright = brights[selected]
        indices = bias["sample_indices"]
        channel_mask = np.zeros(bias["mean"].shape, bool)
        channel_mask[slices] = True
        sampled = channel_mask.ravel()[indices]
        bright_indices = bright["sample_indices"]
        bright_sampled = channel_mask.ravel()[bright_indices]
        dark_samples = bias["sample_values"][:, sampled].astype(float)
        bright_samples = bright["sample_values"][:, bright_sampled].astype(float)
        residuals = {
            "dark_single_pixel": bias["single_pixel_values"][channel]
            - bias["single_pixel_values"][channel].mean(),
            "bright_single_pixel": bright["single_pixel_values"][channel]
            - bright["single_pixel_values"][channel].mean(),
            "dark_centered": dark_samples - bias["mean"].ravel()[indices[sampled]],
            "dark_pair": (
                dark_samples[0 : len(dark_samples) // 2 * 2 : 2]
                - dark_samples[1 : len(dark_samples) // 2 * 2 : 2]
            )
            / math.sqrt(2),
            "bright_centered": bright_samples
            - bright["mean"].ravel()[bright_indices[bright_sampled]],
            "bright_pair": (
                bright_samples[0 : len(bright_samples) // 2 * 2 : 2]
                - bright_samples[1 : len(bright_samples) // 2 * 2 : 2]
            )
            / math.sqrt(2),
        }
        precision = {}
        for label, se_key, difference_key in (
            ("offset", "offset_standard_error_dn", "offset_half_difference_dn"),
            (
                "read_noise",
                "read_noise_standard_error_dn",
                "read_noise_half_difference_dn",
            ),
            ("gain", "gain_standard_error", "gain_half_difference"),
        ):
            theoretical = maps[se_key][slices]
            difference = maps[difference_key][slices]
            values = difference[np.isfinite(difference)]
            precision[label] = {
                "theoretical_rms_full": float(np.sqrt(np.nanmean(theoretical**2)))
                if np.isfinite(theoretical).any()
                else None,
                "empirical_rms_half": float(np.std(values, ddof=1) / math.sqrt(2))
                if values.size > 1
                else None,
                "empirical_estimated_rms_full": float(np.std(values, ddof=1) / 2)
                if values.size > 1
                else None,
                "method": "odd/even independent half maps: SD(A-B)/sqrt(2); full-sample estimate SD(A-B)/2",
            }
        summary = {
            "offset_dn": float(offset.mean()),
            "offset_map_spatial_std_dn": float(np.std(offset, ddof=1)),
            "dsnu_dn": math.sqrt(max(float(np.var(offset, ddof=1)) - read_var / n, 0)),
            "response_slope_dn_per_us": response_slope,
            "response_linearity_r_squared": response_r2,
            "read_noise_rms_dn": math.sqrt(read_var),
            "sensor_read_noise_validated": not lower_rail["suspected_clamping"],
            "read_noise_rms_e": rn_e,
            "gain_dn_per_e": fit["gain_dn_per_e"],
            "e_per_dn": e_per_dn,
            "effective_saturation_capacity_e": effective_fwc,
            "fwc_note": "effective saturation charge; ADC and physical full-well limits are not distinguished"
            if effective_fwc
            else "saturation not observed or conversion gain invalid; full-well capacity is unmeasured",
            "capacity_lower_bound_e": float(np.max(x) * e_per_dn)
            if e_per_dn and effective_fwc is None
            else None,
            "dynamic_range_db": 20 * math.log10(effective_fwc / rn_e)
            if effective_fwc and rn_e and effective_fwc > rn_e
            else None,
            "prnu_fraction_including_illumination": prnu,
            "dark_signal_slope_dn_per_s": float(dark_slope)
            if dark_slope is not None
            else None,
            "dark_signal_slope_standard_error": dark_se,
            "dark_current_e_per_s": float(dark_slope * e_per_dn)
            if e_per_dn and dark_rate_valid
            else None,
            "dark_current_valid": bool(dark_rate_valid),
            "gain_valid_pixel_fraction": float(np.isfinite(gain_map[slices]).mean()),
            "dark_bias_frames": n,
            "bright_frames_per_level": noise.frames,
            "bright_frame_counts_by_level": [int(b["count"]) for b in brights],
        }
        data = {
            "curve": rows,
            "fit": fit,
            "summary": summary,
            "precision": precision,
            "residual_statistics": {k: _noise_hist(v) for k, v in residuals.items()},
            "pixel_distributions": {
                k: _noise_hist(maps[k][slices])
                for k in (
                    "offset_dn",
                    "read_noise_dn",
                    "read_variance_dn2",
                    "gain_dn_per_e",
                )
            },
        }
        result["channels"][channel] = _clean(data)
        maps[f"residual_dark_{channel}"] = residuals["dark_pair"].ravel()
        maps[f"residual_bright_{channel}"] = residuals["bright_pair"].ravel()
        maps[f"centered_dark_{channel}"] = residuals["dark_centered"].ravel()
        maps[f"centered_bright_{channel}"] = residuals["bright_centered"].ravel()
        for kind, stack in (("dark", bias), ("bright", bright)):
            values = stack["single_pixel_values"][channel]
            maps[f"single_pixel_{kind}_{channel}"] = values
            maps[f"single_pixel_{kind}_coordinates_{channel}"] = np.asarray(
                stack["single_pixel_roi_coordinates"][channel]
            )
        maps[f"column_profile_{channel}"] = (
            brights[selected]["mean"][slices] - darks[selected]["mean"][slices]
        ).mean(axis=0)
        maps[f"row_profile_{channel}"] = (
            brights[selected]["mean"][slices] - darks[selected]["mean"][slices]
        ).mean(axis=1)
        maps[f"dark_mean_vs_exposure_{channel}"] = dark_levels
        maps[f"bright_trace_{channel}"] = bright["frame_means"]
        maps[f"bright_timestamps_{channel}"] = bright["timestamps_ns"]
        if not fit["valid"]:
            result["warnings"].append(f"{channel}: {fit['reason']}")
        if prnu_rows and min(r["uniformity_min_over_mean"] for r in prnu_rows) < 0.99:
            result["warnings"].append(
                f"{channel}: 공간 균일도가 99% 미만입니다. FPN·PRNU에는 조명/렌즈 불균일도 포함됩니다."
            )
    result["warnings"].append(
        "차광·밝은 두 조건만 측정했습니다. 밝기별 PTC 곡선·변환 게인·전자 단위·암전류는 미측정입니다."
        if noise.protocol == "two_condition"
        else "노출시간 순회 방식입니다. 논문의 고정 노출시간·광량 조절 실험과 구별하며 각 노출시간의 암영상을 차감합니다."
    )
    return _clean(result), maps


def _figure(rows=1, columns=1, size=None):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    fig = Figure(
        figsize=size or (11, 3.6 * rows), layout="constrained", facecolor="#fafbfe"
    )
    FigureCanvasAgg(fig)
    axes = np.asarray(fig.subplots(rows, columns, squeeze=False)).ravel()
    for ax in axes:
        ax.grid(alpha=0.18)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=9)
    return fig, axes


def _save_plot(folder, name, label, figure, plots):
    for extension in ("png", "svg"):
        figure.savefig(folder / f"{name}.{extension}", dpi=160)
    plots.append(
        {
            "label": label,
            "path": f"{folder.name}/{name}.png",
            "svg": f"{folder.name}/{name}.svg",
        }
    )
    for ax in figure.axes:
        ax.set_xscale("linear")
        ax.set_yscale("linear")
    figure.clear()


def create_camera_plots(folder, result, maps, plots, cancel):
    if result.get("protocol") == "two_condition":
        create_fixed_plots(folder, result, maps, plots, cancel)
        return
    planes = channel_slices(result["settings"]["pixel_format"])
    prefix = f"{result['serial']} · {result['gain_db']:g} dB"
    title = f"{result['serial']} | gain {result['gain_db']:g} dB" + (
        " | SIMULATED" if result["simulated"] else ""
    )
    channels = result["channels"]

    def save(name, label, fig):
        check_cancel(cancel)
        fig.suptitle(title, fontsize=13, fontweight="bold")
        _save_plot(folder, name, prefix + " · " + label, fig, plots)

    for name, plot_label, scope in (
        ("ptc_dn", "PTC · 노이즈 분리 (DN)", 1.0),
        ("ptc_e", "PTC · 전자 기준", None),
    ):
        fig, axes = _figure(2, 2)
        for ax, (ch, data) in zip(axes, channels.items()):
            factor = scope if scope is not None else data["fit"]["e_per_dn"]
            if factor is None:
                ax.text(
                    0.5,
                    0.5,
                    "Conversion gain not validated",
                    ha="center",
                    transform=ax.transAxes,
                )
                continue
            x = np.array([r["signal_dn"] for r in data["curve"]]) * factor
            for key, label in (
                ("total_noise_dn", "Total (temporal + spatial)"),
                ("temporal_noise_dn", "Temporal"),
                ("shot_noise_dn", "Shot"),
                ("dark_noise_dn", "Matched dark"),
                ("read_noise_dn", "Read"),
                ("fpn_dn", "FPN + illumination"),
                ("column_fpn_dn", "Column FPN"),
                ("pixel_fpn_dn", "Pixel FPN"),
            ):
                y = np.array([r[key] for r in data["curve"]]) * factor
                keep = (x > 0) & (y > 0)
                ax.loglog(
                    x[keep], y[keep], ".-", label=label, linewidth=1.2, markersize=4
                )
            ax.set(
                title=ch,
                xlabel="Signal [stored RAW DN]" if scope else "Signal [e-]",
                ylabel="Noise RMS [DN]" if scope else "Noise RMS [e-]",
            )
            ax.legend(fontsize=6.5)
        save(name, plot_label, fig)
    fig, axes = _figure(2, 2)
    for ax, (ch, data) in zip(axes, channels.items()):
        x = np.array([r["signal_dn"] for r in data["curve"]])
        y = np.array([r["shot_variance_dn2"] for r in data["curve"]])
        valid = np.array([r["fit_used"] for r in data["curve"]])
        ax.scatter(x[valid], y[valid], color=COLORS[ch], label="Fit points")
        ax.scatter(x[~valid], y[~valid], marker="x", color="#9399a5", label="Excluded")
        if data["fit"]["valid"]:
            ax.plot(
                x,
                x * data["fit"]["gain_dn_per_e"],
                label=f"R²={data['fit']['r_squared']:.4f}",
                color="#26334b",
            )
        ax.set(title=ch, xlabel="Signal [DN]", ylabel="Shot variance [DN²]")
        ax.legend(fontsize=8)
    save("variance_fit", "분산–평균 · 변환 게인 피팅", fig)
    fig, axes = _figure(2, 2)
    for ax, (ch, _) in zip(axes, channels.items()):
        for kind, color in (("dark", "#7160a8"), ("bright", COLORS[ch])):
            values = maps[f"residual_{kind}_{ch}"]
            sigma = max(float(values.std()), 1e-9)
            limit = max(float(np.quantile(np.abs(values), 0.999)), sigma * 5)
            ax.hist(
                values,
                bins=100,
                range=(-limit, limit),
                density=True,
                histtype="step",
                color=color,
                label=f"{kind}: σ={sigma:.2f}",
            )
            x = np.linspace(-limit, limit, 300)
            ax.plot(
                x,
                np.exp(-x * x / (2 * sigma * sigma)) / (sigma * math.sqrt(2 * math.pi)),
                "--",
                color=color,
                alpha=0.6,
            )
        ax.set(
            title=ch + " | (frame A - B) / sqrt(2)",
            xlabel="Noise residual [DN]",
            ylabel="Probability density",
        )
        ax.legend(fontsize=8)
    save("residual_histogram", "노이즈 잔차 히스토그램 · Gaussian 비교", fig)
    fig, axes = _figure(2, 2)
    for ax, (ch, _) in zip(axes, channels.items()):
        for kind, color in (("dark", "#7160a8"), ("bright", COLORS[ch])):
            values = maps[f"single_pixel_{kind}_{ch}"]
            unique = np.unique(values.astype(np.int64))
            # Integer RAW codes retain their native spacing. Bin-aligned Gaussian
            # probabilities compare discrete observations without smoothing data.
            step = int(np.gcd.reduce(np.diff(unique))) if len(unique) > 1 else 1
            step = max(step, 1)
            mean = float(values.mean())
            sigma = float(values.std(ddof=1))
            edges = np.arange(unique[0] - step / 2, unique[-1] + step, step)
            counts, _ = np.histogram(values, bins=edges)
            centers = (edges[:-1] + edges[1:]) / 2 - mean
            ax.step(
                centers,
                counts / len(values),
                where="mid",
                color=color,
                label=f"{kind}: one pixel, N={len(values)}, SD={sigma:.3g} DN",
            )
            if sigma > 0:
                cdf = np.array(
                    [
                        0.5 * (1 + math.erf((edge - mean) / (sigma * math.sqrt(2))))
                        for edge in edges
                    ]
                )
                ax.plot(centers, np.diff(cdf), "--", color=color, alpha=0.7)
        ax.set(
            title=ch,
            xlabel="One pixel: RAW - its temporal mean [DN]",
            ylabel="Probability per native RAW bin",
        )
        ax.legend(fontsize=6.5)
    save(
        "single_pixel_histogram",
        "단일 픽셀 시간 잔차 · 논문 그림 5(a) · RAW 양자화 유지",
        fig,
    )
    fig, axes = _figure(2, 2)
    for ax, (ch, _) in zip(axes, channels.items()):
        for kind, color in (("dark", "#7160a8"), ("bright", COLORS[ch])):
            values = maps[f"residual_{kind}_{ch}"]
            normalized = (values - values.mean()) / max(float(values.std()), 1e-12)
            ax.hist(
                normalized,
                bins=120,
                range=(-6, 6),
                density=True,
                histtype="step",
                color=color,
                label=kind,
            )
        x = np.linspace(-6, 6, 300)
        ax.plot(
            x,
            np.exp(-x * x / 2) / math.sqrt(2 * math.pi),
            "--",
            color="#26334b",
            label="Standard Gaussian",
        )
        ax.axvline(-3, color="#bbb", linewidth=0.8)
        ax.axvline(3, color="#bbb", linewidth=0.8)
        ax.set(title=ch, xlabel="Normalized residual z", ylabel="Probability density")
        ax.legend(fontsize=8)
    save("normalized_residual", "표준화 잔차 · Gaussian 형태·3σ 꼬리", fig)
    fig, axes = _figure(2, 2)
    for ax, (key, label) in zip(
        axes,
        (
            ("offset_dn", "Offset [DN]"),
            ("read_noise_dn", "Pixel read noise [DN RMS]"),
            ("read_variance_dn2", "Read variance [DN²]"),
            ("gain_dn_per_e", "Conversion gain [DN/e-]"),
        ),
    ):
        for ch, slices in planes.items():
            values = maps[key][slices].ravel()
            values = values[np.isfinite(values)]
            if values.size:
                ax.hist(
                    values,
                    bins=80,
                    histtype="step",
                    density=True,
                    label=ch,
                    color=COLORS[ch],
                )
        ax.set(xlabel=label, ylabel="Pixel density")
        ax.legend(fontsize=8)
    save("pixel_distributions", "픽셀별 오프셋·노이즈·분산·게인 분포", fig)
    fig, axes = _figure(2, 3, (12, 8))
    for ax, (key, label) in zip(
        axes,
        (
            ("offset_dn", "Offset [DN]"),
            ("read_noise_dn", "Read noise [DN]"),
            ("read_variance_dn2", "Read variance [DN²]"),
            ("gain_dn_per_e", "Gain [DN/e-]"),
            ("gain_standard_error", "Gain standard error [DN/e-]"),
            ("offset_standard_error_dn", "Offset standard error [DN]"),
        ),
    ):
        values = maps[key]
        finite = values[np.isfinite(values)]
        if finite.size:
            low, high = np.quantile(finite, [0.01, 0.99])
            im = ax.imshow(
                values,
                cmap="magma",
                vmin=low,
                vmax=max(high, low + 1e-9),
                interpolation="nearest",
            )
            fig.colorbar(im, ax=ax, shrink=0.7)
        ax.set(title=label, xlabel="ROI x [pixel]", ylabel="ROI y [pixel]")
        ax.grid(False)
    save("pixel_maps", "픽셀 맵 · 오프셋·읽기 노이즈·게인·정밀도", fig)
    fig, axes = _figure(2, 2)
    for ax, (ch, data) in zip(axes, channels.items()):
        rows = data["curve"]
        x = [r["signal_dn"] for r in rows]
        ax.semilogx(x, [r["snr_temporal"] for r in rows], ".-", label="Temporal SNR")
        ax.semilogx(x, [r["snr_total"] for r in rows], ".-", label="Total SNR")
        right = ax.twinx()
        key = "relative_snr_photon" if result["qe"] else "relative_snr_electron"
        valid = [
            (r["signal_dn"], r[key])
            for r in rows
            if r[key] is not None and r["signal_dn"] > 0
        ]
        if valid:
            xx, yy = zip(*valid)
            right.semilogx(
                xx,
                yy,
                "--",
                color=COLORS[ch],
                label="rSNR (photon)" if result["qe"] else "rSNR (electron reference)",
            )
        right.set_ylabel("Relative SNR", fontsize=8)
        ax.set(
            title=ch
            + (" | QE unknown" if not result["qe"] else f" | QE={result['qe']:g}"),
            xlabel="Signal [DN]",
            ylabel="SNR",
        )
        ax.legend(fontsize=7)
    save("snr", "SNR·상대 SNR · QE 조건 표시", fig)
    fig, axes = _figure(1, 3, (12, 4))
    for ax, kind in zip(axes, ("offset", "read_noise", "gain")):
        names = list(channels)
        theory = [
            channels[c]["precision"][kind]["theoretical_rms_full"] or 0 for c in names
        ]
        empirical = [
            channels[c]["precision"][kind]["empirical_estimated_rms_full"] or 0
            for c in names
        ]
        xx = np.arange(len(names))
        ax.bar(xx - 0.18, theory, 0.36, label="Theory full sample")
        ax.bar(xx + 0.18, empirical, 0.36, label="Independent-half estimate")
        ax.set(title=kind, xticks=xx, xticklabels=names, ylabel="RMS standard error")
        ax.legend(fontsize=7)
    save("precision", "픽셀 맵 측정 정밀도 · 이론/독립 표본", fig)
    fig, axes = _figure(2, 2)
    for ch, data in channels.items():
        exposures = [r["exposure_us"] / 1000 for r in data["curve"]]
        color = COLORS[ch]
        axes[0].plot(
            exposures,
            [r["signal_dn"] for r in data["curve"]],
            ".-",
            label=ch,
            color=color,
        )
        slope = data["summary"]["response_slope_dn_per_us"]
        if slope is not None:
            axes[0].plot(
                exposures,
                np.asarray(exposures) * 1000 * slope,
                "--",
                color=color,
                alpha=0.5,
            )
        axes[1].plot(
            exposures, maps[f"dark_mean_vs_exposure_{ch}"], ".-", label=ch, color=color
        )
        axes[2].plot(maps[f"column_profile_{ch}"], label=ch, color=color)
        axes[3].plot(maps[f"row_profile_{ch}"], label=ch, color=color)
    for ax, xlabel, ylabel in zip(
        axes,
        (
            "Exposure time [ms]",
            "Exposure time [ms]",
            "ROI column / CFA plane",
            "ROI row / CFA plane",
        ),
        (
            "Signal [DN] / linearity",
            "Dark mean [DN] / dark current",
            "Column mean [DN]",
            "Row mean [DN]",
        ),
    ):
        ax.set(xlabel=xlabel, ylabel=ylabel)
        ax.legend(fontsize=8)
    save("linearity_and_patterns", "노출시간 응답·암전류·행/열 패턴", fig)
    fig, axes = _figure(2, 2)
    first = next(iter(channels))
    trace = maps[f"bright_trace_{first}"]
    ts = maps[f"bright_timestamps_{first}"]
    times = (ts - ts[0]) / 1e9
    axes[0].plot(times, trace, color="#7762ad")
    axes[0].set(xlabel="Time [s]", ylabel="ROI frame mean [DN]")
    centered = trace - trace.mean()
    dt = float(np.median(np.diff(times)))
    if dt > 0 and len(trace) >= 4:
        spectrum = np.abs(np.fft.rfft(centered * np.hanning(len(trace)))) ** 2
        axes[1].plot(np.fft.rfftfreq(len(trace), dt), spectrum)
        axes[1].set(
            xlabel="Frequency [Hz] (median sampling interval)",
            ylabel="Frame-mean power",
        )
    if np.sum(centered**2) > 0:
        size = 1 << (2 * len(centered) - 1).bit_length()
        transformed = np.fft.rfft(centered, size)
        corr = np.fft.irfft(transformed * transformed.conj(), size)[
            : len(centered)
        ] / np.sum(centered**2)
        axes[2].plot(np.arange(min(100, len(corr))), corr[:100])
        axes[2].set(xlabel="Frame lag", ylabel="Normalized autocorrelation")
    rows = channels[first]["curve"]
    axes[3].semilogx(
        [r["signal_dn"] for r in rows],
        [r["excess_flicker_fraction"] * 100 for r in rows],
        ".-",
    )
    axes[3].set(xlabel="Signal [DN]", ylabel="Excess illumination fluctuation [%]")
    save("temporal_quality", "시간별 흔들림·자기상관·주파수·조명 검증", fig)


def _comparison_plot(folder, cameras, plots):
    if cameras and all(c.get("protocol") == "two_condition" for c in cameras):
        fig, axes = _figure(1, 3, (13, 4))
        rows = [(c, ch, d) for c in cameras for ch, d in c["channels"].items()]
        labels = [f"{c['serial']} {ch}" for c, ch, _ in rows]
        for ax, key, label in zip(
            axes,
            ("read_noise_dn", "temporal_noise_dn", "signal_dn"),
            (
                "Dark noise [stored DN RMS]",
                "Bright temporal noise [stored DN RMS]",
                "Bright signal [stored DN]",
            ),
        ):
            ax.bar(labels, [d["curve"][0][key] for _, _, d in rows])
            ax.tick_params(axis="x", rotation=75, labelsize=6)
            ax.set(ylabel=label)
        fig.suptitle(
            "Two conditions | DN scales and camera operating settings may differ"
        )
        _save_plot(
            folder,
            "comparison_dn",
            "전체 카메라 · 명암 측정 비교 (저장 DN 단위)",
            fig,
            plots,
        )
        return
    fig, axes = _figure(2, 2)
    for camera in cameras:
        for ch, data in camera["channels"].items():
            label = f"{camera['serial']} {camera['gain_db']:g}dB {ch}"
            rows = [r for r in data["curve"] if r["signal_e"] and r["signal_e"] > 0]
            if not rows:
                continue
            x = [r["signal_e"] for r in rows]
            axes[0].loglog(
                x,
                [r["total_noise_dn"] * data["fit"]["e_per_dn"] for r in rows],
                ".-",
                label=label,
            )
            axes[1].semilogx(x, [r["snr_total"] for r in rows], ".-", label=label)
            axes[2].semilogx(
                x, [r["relative_snr_electron"] for r in rows], ".-", label=label
            )
            photon = [r for r in rows if r["signal_photons"]]
            if photon:
                axes[3].semilogx(
                    [r["signal_photons"] for r in photon],
                    [r["relative_snr_photon"] for r in photon],
                    ".-",
                    label=label,
                )
    for ax, ylabel in zip(
        axes,
        (
            "Total noise [e- RMS]",
            "SNR",
            "Relative SNR (ideal electron reference)",
            "Relative SNR (ideal photon reference, QE required)",
        ),
    ):
        ax.set(
            xlabel="Signal [e-]" if ax is not axes[3] else "Incident photons / pixel",
            ylabel=ylabel,
        )
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=5.5, ncol=2)
        else:
            ax.text(
                0.5,
                0.5,
                "No validated conversion gain / QE",
                ha="center",
                transform=ax.transAxes,
            )
    _save_plot(folder, "comparison", "전체 카메라·게인 비교 · PTC/SNR/rSNR", fig, plots)


def analyze_noise_session(path, cancel=None, progress=None):
    root = Path(path).resolve()
    cancel = cancel or Event()
    progress = progress or (lambda *args: None)
    session, dark = verify_phase(root, "dark", cancel)
    _, bright = verify_phase(root, "bright", cancel)
    if session["status"] == "complete":
        verify_noise_session(root, cancel)
        report = json.loads((root / "analysis/report.json").read_text())
        # Enrich the display of legacy results from verified source RAW, while
        # retaining the original analysis files and their completion checksums.
        for camera in report["cameras"]:
            check_cancel(cancel)
            if "dark_lower_rail" not in camera:
                record = next(
                    (
                        r
                        for r in dark["stacks"]
                        if r["serial"] == camera["serial"]
                        and r["point"]["level"] == 0
                        and abs(
                            json.loads(_safe_child(root, r["metadata"]).read_text())[
                                "settings"
                            ]["gain_db"]
                            - camera["gain_db"]
                        )
                        <= 0.05
                    ),
                    None,
                )
                if record is None:
                    raise CaptureError(
                        "기존 결과와 대응하는 차광 RAW를 찾지 못했습니다."
                    )
                rail = _dark_lower_rail(_load(root, record))
                camera["dark_lower_rail"] = rail
                if rail["suspected_clamping"]:
                    camera["warnings"].append(
                        f"기존 RAW 재검사: 차광 표본의 {rail['sampled_floor_fraction'] * 100:.1f}%가 {rail['sampled_floor_dn']} DN 하한에 집중되었습니다. 블랙 레벨 클램핑으로 센서 읽기 노이즈가 과소평가될 수 있습니다."
                    )
                    for data in camera["channels"].values():
                        data["summary"]["sensor_read_noise_validated"] = False
            if any(
                "near_white_fraction" not in row
                for data in camera["channels"].values()
                for row in data["curve"]
            ):
                diagnostics = []
                for record in bright["stacks"]:
                    if record["serial"] != camera["serial"]:
                        continue
                    check_cancel(cancel)
                    stats = _load(root, record)
                    if (
                        abs(
                            stats["metadata"]["settings"]["gain_db"] - camera["gain_db"]
                        )
                        > 0.05
                    ):
                        continue
                    diagnostics.append(
                        {
                            "point": record["point"]["index"],
                            "near_white_fraction": float(stats["near_white"].mean())
                            / int(stats["count"]),
                        }
                    )
                camera["legacy_quality_recheck"] = diagnostics
                saturated = sum(d["near_white_fraction"] >= 0.99 for d in diagnostics)
                if saturated:
                    camera["warnings"].append(
                        f"기존 RAW 재검사: {saturated}개 단계가 거의 전체 포화입니다. 명목 상한보다 낮은 카메라 클램핑 값도 검출했습니다. 조명을 약하게 하고 ②로 밝은 촬영만 다시 하세요."
                    )
        return {
            "phase": "analysis",
            "path": str(root),
            "report": report,
        }
    noise = NoiseConfig.from_dict(session["noise_config"])
    output = root / ("analysis_" + uuid.uuid4().hex[:8] + ".partial")
    output.mkdir()
    dark_records = {(r["point"]["index"], r["serial"]): r for r in dark["stacks"]}
    bright_records = {(r["point"]["index"], r["serial"]): r for r in bright["stacks"]}
    report = {
        "schema_version": 1,
        "paper": PAPER_URL,
        "created_at_utc": utc_now(),
        "simulated": session["simulated"],
        "method": session["method"],
        "noise_config": session["noise_config"],
        "cameras": [],
        "plots": [],
        "method_notes": [
            "Unbiased pixel temporal variance (N-1); disjoint frame differences divided by sqrt(2).",
            "Fast 100-frame stacks retain legacy 0.1% pair photometric stability acceptance; alternatively a four-SEM upper bound of 0.1% common illumination variance relative to CFA pair noise variance can validate a stack, with a fixed LED spectrum assumption and independent colour guards. RAW samples are never detrended or adjusted. Recorded frame counts can differ across reused levels.",
            "Spatial FPN: variance of dark-subtracted mean image minus finite bright/dark mean-estimation variances. Total variance = temporal + spatial FPN.",
            "Column FPN is a component of FPN; pixel FPN variance = FPN variance - column FPN variance. This corrects the sign in the printed Eq.18.",
            "Printed Eq.14 centers each pixel over time and therefore excludes fixed spatial patterns. Both temporal and spatial-inclusive total noise are reported explicitly.",
            "Gain maps use Eq.28-30, with matched-exposure dark variance subtraction; negative/insufficient/saturated pixel fits remain NaN.",
            "Precision uses Eq.31-36 error propagation and independent odd/even half maps. Half-map difference SD/sqrt(2); full-sample estimate SD/2.",
            "QE and absolute photon flux are not measured. Photon rSNR requires externally supplied per-camera QE at the recorded wavelength.",
            "Saturation charge is effective ADC/sensor saturation; it is not an independently measured physical full-well capacity.",
            "Frame-mean FFT uses median sampling interval; nonuniform timing limits spectral accuracy. PTC acquisition uses a fixed light exposure sweep rather than the paper's fixed-time intensity sweep.",
            "Illumination filtering uses a separate pilot and frozen spatial/CFA mean bounds, preferably on pixels outside the noise ROI. Rejected whole-camera groups are replaced at unchanged exposure/gain; all decisions are SHA-256 audited. Results describe the selected stable illumination state, not an unfiltered camera-noise distribution under arbitrary LED modulation.",
            "Near-white occupancy is measured from unmodified RAW at >=99.5% of the nominal sensor white level. It is a conservative saturation warning, distinct from exact-white clipping and not a physical full-well measurement.",
        ],
    }
    try:
        total = len(session["cameras"]) * len(noise.gains_db)
        for index, (camera, gain_index) in enumerate(
            (c, g) for c in session["cameras"] for g in range(len(noise.gains_db))
        ):
            check_cancel(cancel)
            progress(
                f"노이즈 계산 · {camera['serial']} · 게인 {noise.gains_db[gain_index]:g} dB",
                index,
                total,
            )
            with TemporaryDirectory(prefix="calculation_", dir=output) as cache:
                result, maps = characterize(
                    root,
                    camera,
                    gain_index,
                    noise,
                    dark_records,
                    bright_records,
                    cancel,
                    cache=Path(cache),
                    acquisition_plan=session.get("acquisition_plan"),
                )
                folder = output / f"{camera['serial']}_gain_{gain_index:02d}"
                folder.mkdir()
                save_npz(folder / "pixel_maps_and_residuals.npz", maps)
                result["maps"] = f"{folder.name}/pixel_maps_and_residuals.npz"
                create_camera_plots(folder, result, maps, report["plots"], cancel)
                with (folder / "curves.csv").open(
                    "w", newline="", encoding="utf-8"
                ) as stream:
                    first = next(iter(result["channels"].values()))["curve"][0]
                    writer = csv.DictWriter(stream, fieldnames=["channel", *first])
                    writer.writeheader()
                    for channel, data in result["channels"].items():
                        for row in data["curve"]:
                            writer.writerow({"channel": channel, **row})
                report["cameras"].append(result)
                del maps
        comparison = output / "comparison"
        comparison.mkdir()
        _comparison_plot(comparison, report["cameras"], report["plots"])
        report = _clean(report)
        write_json(output / "report.json", report)
        _write_html(output, report)
        manifest = {
            "schema_version": 1,
            "files": [
                {"path": str(p.relative_to(output)), "sha256": sha256_file(p)}
                for p in sorted(output.rglob("*"))
                if p.is_file()
            ],
        }
        write_json(output / "manifest.json", manifest)
        for record in manifest["files"]:
            check_cancel(cancel)
            if sha256_file(_safe_child(output, record["path"])) != record["sha256"]:
                raise CaptureError("분석 결과 저장 후 체크섬 불일치")
        if (root / "analysis").exists():
            (root / "analysis").rename(
                root / ("previous_analysis_" + uuid.uuid4().hex[:8])
            )
        output.rename(root / "analysis")
        session.update(status="complete", completed_at_utc=utc_now())
        write_json(root / "session.json", session)
        write_json(
            root / "COMPLETE.json",
            {
                "sha256": {
                    name: sha256_file(_safe_child(root, name))
                    for name in (
                        "session.json",
                        "dark.json",
                        "bright.json",
                        "analysis/manifest.json",
                    )
                }
            },
        )
        fsync_dir(root)
        verify_noise_session(root, cancel)
        if root.name.endswith(".partial"):
            final = root.with_name(root.name.removesuffix(".partial"))
            os.rename(root, final)
            fsync_dir(final.parent)
            root = final
        progress("노이즈 분석·그래프·RAW 무결성 검사 완료", total, total)
        return {"phase": "analysis", "path": str(root), "report": report}
    except Exception as error:
        write_json(
            root / ("FAILED_ANALYSIS_" + uuid.uuid4().hex[:8] + ".json"),
            {"error": str(error), "time_utc": utc_now()},
        )
        raise


def _write_html(output, report):
    import html

    parts = [
        '<!doctype html><html lang="ko"><meta charset="utf-8"><title>RAW Rig noise report</title>',
        "<style>body{max-width:1280px;margin:40px auto;padding:0 24px;font:15px system-ui;color:#26334b;background:#fafbfe}img{width:100%;background:white;border:1px solid #dde1ee;border-radius:12px}table{border-collapse:collapse;width:100%}td,th{border:1px solid #dde1ee;padding:10px;text-align:left}section{margin:32px 0}</style>",
        "<h1>RAW Rig · 카메라 노이즈 분석</h1>",
        "<p>"
        + (
            "SIMULATED · 가상 데이터 / 실제 센서 측정 아님"
            if report["simulated"]
            else "원본 RAW 기반 측정"
        )
        + "</p>",
        '<p><a href="'
        + PAPER_URL
        + '">Li et al. (2016), Assessing low-light cameras with photon transfer curve method</a></p>',
        "<table><tr><th>카메라 / 게인 / 채널</th><th>읽기 노이즈 DN / e−</th><th>게인 DN/e−</th><th>유효 포화용량 e− / DR dB</th><th>피팅 R²</th></tr>",
    ]

    def value(v):
        return "측정 불가" if v is None else f"{v:.4g}"

    for camera in report["cameras"]:
        for ch, data in camera["channels"].items():
            s = data["summary"]
            parts.append(
                "<tr><td>"
                + html.escape(f"{camera['serial']} / {camera['gain_db']:g} dB / {ch}")
                + "</td>"
                + "".join(
                    "<td>" + v + "</td>"
                    for v in (
                        value(s["read_noise_rms_dn"])
                        + " / "
                        + value(s["read_noise_rms_e"]),
                        value(s["gain_dn_per_e"]),
                        value(s["effective_saturation_capacity_e"])
                        + " / "
                        + value(s["dynamic_range_db"]),
                        value(data["fit"]["r_squared"]),
                    )
                )
                + "</tr>"
            )
    parts.append("</table><h2>측정 조건·유효성</h2>")
    for camera in report["cameras"]:
        parts.append(
            "<p>"
            + html.escape(camera["serial"] + ": " + " / ".join(camera["warnings"]))
            + "</p>"
        )
    parts.append(
        "<h2>계산 방법</h2><ul>"
        + "".join("<li>" + html.escape(n) + "</li>" for n in report["method_notes"])
        + "</ul>"
    )
    for plot in report["plots"]:
        parts.append(
            "<section><h2>"
            + html.escape(plot["label"])
            + '</h2><a href="'
            + html.escape(plot["svg"])
            + '">SVG 원본</a><img loading="lazy" src="'
            + html.escape(plot["path"])
            + '" alt="'
            + html.escape(plot["label"])
            + '"></section>'
        )
    (output / "report.html").write_text("\n".join(parts) + "</html>", encoding="utf-8")


def compare_noise_sessions(paths, output, cancel=None):
    cancel = cancel or Event()
    cameras = []
    for index, path in enumerate(paths):
        check_cancel(cancel)
        verify_noise_session(path, cancel)
        report = json.loads((Path(path) / "analysis/report.json").read_text())
        for camera in report["cameras"]:
            cameras.append({**camera, "serial": f"{index + 1}:{camera['serial']}"})
    folder = Path(output) / ("noise_comparison_" + uuid.uuid4().hex[:8])
    folder.mkdir(parents=True)
    plots = []
    _comparison_plot(folder, cameras, plots)
    # The viewer resolves plots relative to the comparison folder's parent.
    write_json(
        folder / "sources.json",
        {"paths": [str(Path(p).resolve()) for p in paths], "cameras": cameras},
    )
    return {
        "phase": "comparison",
        "path": str(folder.parent),
        "report": {
            "cameras": cameras,
            "plots": plots,
            "simulated": any(c["simulated"] for c in cameras),
        },
    }


def create_fixed_plots(folder, result, maps, plots, cancel, dark_only=False):
    """Only measured DN distributions for the fixed two-condition experiment."""
    channels = result["channels"]
    n = next(iter(channels.values()))["summary"]["dark_bias_frames"]
    planes = channel_slices(result["settings"]["pixel_format"])
    prefix = result["serial"] + " · " + ("차광" if dark_only else "고정 명암")

    def save(name, label, fig):
        check_cancel(cancel)
        fig.suptitle(
            result["serial"]
            + (" | SIMULATED" if result["simulated"] else " | RAW measurement"),
            fontweight="bold",
        )
        _save_plot(folder, name, prefix + " · " + label, fig, plots)

    fig, axes = _figure(2, 2)
    for ax, (ch, data) in zip(axes, channels.items()):
        values = maps["residual_dark_" + ch]
        sigma = float(values.std())
        bound = max(6 * sigma, 16.0)
        ax.hist(
            values,
            bins=120,
            range=(-bound, bound),
            density=True,
            histtype="step",
            color=COLORS[ch],
            label=f"Pair residual: SD={sigma:.3g} DN",
        )
        centered = maps["centered_dark_" + ch]
        ax.hist(
            centered,
            bins=120,
            range=(-bound, bound),
            density=True,
            histtype="step",
            color="#7762ad",
            label="Pixel-mean residual",
        )
        if sigma > 0:
            xx = np.linspace(-bound, bound, 300)
            ax.plot(
                xx,
                np.exp(-xx * xx / (2 * sigma * sigma)) / (sigma * np.sqrt(2 * np.pi)),
                "--",
                color="#26334b",
                label="Gaussian reference",
            )
        ax.set(
            title=ch,
            xlabel="Dark residual [stored RAW DN]",
            ylabel="Probability density",
        )
        ax.legend(fontsize=7)
    save("dark_residual", "차광 잔차 히스토그램 · 픽셀 평균/프레임 차분", fig)
    fig, axes = _figure(1, 3, (12, 4))
    for ax, (key, label) in zip(
        axes,
        (
            ("offset_dn", "Offset [DN]"),
            ("read_noise_dn", "Pixel dark noise RMS [DN]"),
            ("read_variance_dn2", "Pixel variance [DN²]"),
        ),
    ):
        for ch, slices in planes.items():
            values = maps[key][slices].ravel()
            ax.hist(
                values,
                bins=80,
                density=True,
                histtype="step",
                label=ch,
                color=COLORS[ch],
            )
        ax.set(xlabel=label, ylabel="Pixel density")
        ax.legend(fontsize=8)
    save("dark_distributions", "픽셀별 오프셋·노이즈 크기·분산 분포", fig)
    fig, axes = _figure(2, 2)
    for ax, (key, label) in zip(
        axes,
        (
            ("offset_dn", "Offset [DN]"),
            ("read_noise_dn", "Dark temporal noise [DN RMS]"),
            ("read_variance_dn2", "Dark variance [DN²]"),
            ("offset_standard_error_dn", "Offset standard error [DN]"),
        ),
    ):
        values = maps[key]
        lo, hi = np.quantile(values, [0.01, 0.99])
        im = ax.imshow(
            values,
            vmin=lo,
            vmax=max(hi, lo + 1e-9),
            cmap="magma",
            interpolation="nearest",
        )
        fig.colorbar(im, ax=ax, shrink=0.75)
        ax.set(title=label, xlabel="ROI x [pixel]", ylabel="ROI y [pixel]")
        ax.grid(False)
    save("dark_maps", "픽셀별 오프셋·읽기 노이즈·분산·정밀도 맵", fig)
    fig, axes = _figure(1, 2, (11, 4))
    for ax, kind in zip(axes, ("offset", "read_noise")):
        names = list(channels)
        xx = np.arange(len(names))
        theory = [channels[c]["precision"][kind]["theoretical_rms_full"] for c in names]
        empirical = [
            channels[c]["precision"][kind]["empirical_estimated_rms_full"]
            for c in names
        ]
        ax.bar(xx - 0.18, theory, 0.36, label=f"Gaussian theory ({n} frames)")
        ax.bar(xx + 0.18, empirical, 0.36, label="Independent halves estimate")
        ax.set(title=kind, xticks=xx, xticklabels=names, ylabel="Standard error [DN]")
        ax.legend(fontsize=7)
    save("dark_precision", f"{n}장 측정 정밀도 · 이론/독립 표본", fig)
    fig, axes = _figure(1, 2, (11, 4))
    times = maps["dark_timestamps_ns"]
    seconds = (times - times[0]) / 1e9
    axes[0].plot(seconds, maps["dark_frame_means"])
    axes[0].set(xlabel="Time [s]", ylabel="Dark ROI mean [DN]")
    for ch, data in channels.items():
        values = maps["read_noise_dn"][planes[ch]].ravel()
        values = np.sort(values)
        axes[1].plot(
            values,
            np.arange(1, len(values) + 1) / len(values),
            label=ch,
            color=COLORS[ch],
        )
    axes[1].set(xlabel="Pixel dark noise [DN RMS]", ylabel="Cumulative pixel fraction")
    axes[1].legend(fontsize=8)
    save("dark_trace_cdf", "차광 시간 추이·픽셀 노이즈 누적 분포", fig)
    if not dark_only:
        fig, axes = _figure(2, 2)
        for ax, (ch, data) in zip(axes, channels.items()):
            row = data["curve"][0]
            keys = (
                "read_noise_dn",
                "temporal_noise_dn",
                "shot_noise_dn",
                "fpn_dn",
                "total_noise_dn",
            )
            ax.bar(
                ["Dark", "Bright temporal", "Shot estimate", "Spatial FPN", "Total"],
                [row[k] for k in keys],
                color=["#7762ad", COLORS[ch], "#66b586", "#c580b9", "#536683"],
            )
            ax.tick_params(axis="x", rotation=25)
            ax.set(title=ch, ylabel="Noise RMS [DN]")
        save(
            "two_condition_noise",
            "고정된 명암 두 조건의 노이즈 성분 · PTC 곡선 아님",
            fig,
        )
        fig, axes = _figure(2, 2)
        for ax, (ch, data) in zip(axes, channels.items()):
            for kind, color in (("dark", "#7762ad"), ("bright", COLORS[ch])):
                values = maps[f"residual_{kind}_{ch}"]
                std = max(float(values.std()), 1e-9)
                z = (values - values.mean()) / std
                ax.hist(
                    z,
                    bins=120,
                    range=(-6, 6),
                    density=True,
                    histtype="step",
                    label=kind,
                    color=color,
                )
            xx = np.linspace(-6, 6, 300)
            ax.plot(
                xx, np.exp(-xx * xx / 2) / np.sqrt(2 * np.pi), "--", color="#26334b"
            )
            ax.set(
                title=ch, xlabel="Normalized residual z", ylabel="Probability density"
            )
            ax.legend(fontsize=8)
        save("two_condition_residual", "명암 표준화 잔차 · Gaussian 꼬리 비교", fig)


def analyze_dark_session(path, cancel=None, progress=None):
    """Publish measured dark graphs without claiming a completed bright/PTC test."""
    root = Path(path).resolve()
    cancel = cancel or Event()
    progress = progress or (lambda *args: None)
    session, manifest = verify_phase(root, "dark", cancel)
    NoiseConfig.from_dict(session["noise_config"])
    output = root / ("dark_analysis_" + uuid.uuid4().hex[:8] + ".partial")
    output.mkdir()
    report = {
        "schema_version": 1,
        "paper": PAPER_URL,
        "simulated": session["simulated"],
        "method": "dark-only temporal statistics at the measured exposure/gain; minimum exposure of a sweep or frozen camera setting",
        "noise_config": session["noise_config"],
        "cameras": [],
        "plots": [],
        "method_notes": [
            "Dark-only result: bright statistics, conversion gain, electrons, PTC and dark current are unmeasured.",
            "All original RAW DN retained; unbiased temporal variance (N-1).",
            "Dark temporal noise may include dark-current noise at this exposure time. Stored RAW DN are not necessarily native ADC codes.",
            "Read-noise standard-error theory assumes Gaussian noise; independent-half precision and measured residual distributions are also reported.",
        ],
    }
    records = [r for r in manifest["stacks"] if r["point"]["level"] == 0]
    for index, record in enumerate(records):
        check_cancel(cancel)
        progress(
            "차광 분포·노이즈 맵 계산 · " + record["serial"],
            index,
            len(records),
        )
        stats = _load(root, record)
        n = int(stats["count"])
        std = np.sqrt(stats["variance"])
        settings = stats["metadata"]["settings"]
        lower_rail = _dark_lower_rail(stats)
        maps = {
            "offset_dn": stats["mean"],
            "read_noise_dn": std,
            "read_variance_dn2": stats["variance"],
            "offset_standard_error_dn": std / np.sqrt(n),
            "read_noise_standard_error_dn": std / np.sqrt(2 * (n - 1)),
            "dark_frame_means": stats["frame_means"],
            "dark_timestamps_ns": stats["timestamps_ns"],
        }
        camera = next(c for c in session["cameras"] if c["serial"] == record["serial"])
        result = {
            "serial": record["serial"],
            "model": camera["model"],
            "simulated": camera["simulated"],
            "gain_db": settings["gain_db"],
            "roi": stats["metadata"]["roi"],
            "settings": settings,
            "channels": {},
            "dark_lower_rail": lower_rail,
            "warnings": [
                "차광 단독 측정입니다. 밝은 장면·게인·전자 단위·PTC 곡선은 아직 미측정입니다."
            ],
        }
        if float(stats["zero_fraction"]) > 0.01:
            result["warnings"].append(
                "암영상 0 DN 클리핑이 1%를 초과해 노이즈가 과소평가될 수 있습니다."
            )
        if lower_rail["suspected_clamping"]:
            result["warnings"].append(
                f"차광 표본의 {lower_rail['sampled_floor_fraction'] * 100:.1f}%가 {lower_rail['sampled_floor_dn']} DN 하한에 몰려 있습니다. 블랙 레벨 클램핑으로 센서 읽기 노이즈가 과소평가될 수 있습니다. 표시 값은 카메라 출력 RAW의 노이즈입니다."
            )
        for ch, slices in channel_slices(settings["pixel_format"]).items():
            indices = stats["sample_indices"]
            mask = np.zeros(stats["mean"].shape, bool)
            mask[slices] = True
            select = mask.ravel()[indices]
            samples = stats["sample_values"][:, select].astype(float)
            centered = samples - stats["mean"].ravel()[indices[select]]
            pairs = samples[: len(samples) // 2 * 2]
            pair = (pairs[::2] - pairs[1::2]) / np.sqrt(2)
            maps["residual_dark_" + ch] = pair.ravel()
            maps["centered_dark_" + ch] = centered.ravel()
            variance = float(stats["variance"][slices].mean())
            precision = {}
            for kind, se, difference in (
                (
                    "offset",
                    std / np.sqrt(n),
                    stats["half_mean"][0] - stats["half_mean"][1],
                ),
                (
                    "read_noise",
                    std / np.sqrt(2 * (n - 1)),
                    np.sqrt(stats["half_variance"][0])
                    - np.sqrt(stats["half_variance"][1]),
                ),
            ):
                precision[kind] = {
                    "theoretical_rms_full": float(np.sqrt(np.mean(se[slices] ** 2))),
                    "empirical_estimated_rms_full": float(
                        np.std(difference[slices], ddof=1) / 2
                    ),
                }
            result["channels"][ch] = {
                "summary": {
                    "offset_dn": float(stats["mean"][slices].mean()),
                    "read_noise_rms_dn": math.sqrt(variance),
                    "sensor_read_noise_validated": not lower_rail["suspected_clamping"],
                    "read_noise_rms_e": None,
                    "gain_dn_per_e": None,
                    "effective_saturation_capacity_e": None,
                    "dynamic_range_db": None,
                    "dark_bias_frames": n,
                },
                "fit": {
                    "valid": False,
                    "reason": "차광 단독: 변환 게인 미측정",
                    "r_squared": None,
                    "e_per_dn": None,
                },
                "curve": [],
                "precision": precision,
                "residual_statistics": {
                    "dark_pair": _noise_hist(pair),
                    "dark_centered": _noise_hist(centered),
                },
            }
        folder = output / (
            record["serial"] + f"_gain_{record['point']['gain_index']:02d}"
        )
        folder.mkdir()
        save_npz(folder / "dark_maps_and_residuals.npz", maps)
        result["maps"] = folder.name + "/dark_maps_and_residuals.npz"
        create_fixed_plots(
            folder, result, maps, report["plots"], cancel, dark_only=True
        )
        report["cameras"].append(result)
    write_json(output / "report.json", _clean(report))
    _write_html(output, report)
    files = [
        {"path": str(f.relative_to(output)), "sha256": sha256_file(f)}
        for f in sorted(output.rglob("*"))
        if f.is_file()
    ]
    write_json(output / "manifest.json", {"files": files})
    for entry in files:
        check_cancel(cancel)
        if sha256_file(_safe_child(output, entry["path"])) != entry["sha256"]:
            raise CaptureError("차광 분석 결과 체크섬 불일치")
    if (root / "dark_analysis").exists():
        (root / "dark_analysis").rename(
            root / ("previous_dark_analysis_" + uuid.uuid4().hex[:8])
        )
    output.rename(root / "dark_analysis")
    progress("차광 분석·그래프·RAW 무결성 검사 완료", len(records), len(records))
    return {"phase": "dark_analysis", "path": str(root), "report": report}
