"""Two-stage, lossless RAW photon-transfer acquisition (Li et al., 2016)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import random
import shutil
import sys
from threading import Event
import time
import uuid

import numpy as np

from .models import CaptureError, Cancelled, Preset, check_cancel
from .recovery import capture_checked
from .illumination import (
    describe_light,
    excess_flicker,
    excess_pair_flicker,
    pair_noise_bias_upper,
    light_rejection,
    lit_groups,
    PILOT_GROUPS,
    MAX_PILOT_GROUPS,
)
from .storage import (
    fsync_dir,
    sha256_file,
    utc_now,
    write_bytes,
    write_json,
    _safe_child,
)

PAPER_URL = "https://doi.org/10.1142/S1793545816300081"


@dataclass
class NoiseConfig:
    frames: int = 100
    bias_frames: int = 1000
    dark_frames: int = 0
    dark_frame_counts: dict = field(default_factory=dict)
    bright_frame_counts: dict = field(default_factory=dict)
    levels: int = 16
    exposure_min_us: float = 100.0
    exposure_max_us: float = 100000.0
    gains_db: list[float] = field(default_factory=lambda: [0.0])
    roi_size: int = 256
    save_full_raw: bool = False
    sync_mode: str = "software"
    max_flicker_percent: float = 0.1
    wavelength_nm: float = 0.0
    quantum_efficiency: dict = field(default_factory=dict)
    protocol: str = "exposure_sweep"
    anti_flicker_period_us: float = 0.0

    @classmethod
    def fixed_curve(cls):
        """Locked practical PTC protocol; the light stays constant, time changes."""
        return cls(protocol="automatic_ptc")

    @classmethod
    def precision(cls):
        """Dense fixed-light sweep, with matched 1000-frame dark at every time."""
        return cls(
            frames=1000,
            bias_frames=1000,
            levels=32,
            exposure_min_us=24.0,
            exposure_max_us=100000.0,
            protocol="exposure_sweep",
        )

    @classmethod
    def fixed(cls):
        return cls(
            frames=1000,
            bias_frames=1000,
            levels=1,
            exposure_min_us=10000.0,
            exposure_max_us=10000.0,
            roi_size=256,
            protocol="two_condition",
        )

    def validate(self):
        if not isinstance(self.dark_frame_counts, dict) or any(
            not isinstance(k, str) or not k.isdigit() or str(int(k)) != k
            or int(k) >= self.levels * len(self.gains_db)
            or type(v) is not int or not 4 <= v <= 100000
            for k, v in self.dark_frame_counts.items()
        ):
            raise CaptureError("차광 촬영 단계별 반복 수가 잘못되었습니다.")
        if type(self.dark_frames) is not int or (self.dark_frames != 0 and not 4 <= self.dark_frames <= 100000):
            raise CaptureError("차광 반복 촬영 수는 4~100000이어야 합니다.")
        if not isinstance(self.bright_frame_counts, dict) or any(
            not isinstance(k, str) or not k.isdigit() or str(int(k)) != k
            or int(k) >= self.levels * len(self.gains_db)
            or type(v) is not int or not 4 <= v <= 100000
            for k, v in self.bright_frame_counts.items()
        ):
            raise CaptureError("밝은 촬영 단계별 반복 수가 잘못되었습니다.")
        if self.protocol not in ("exposure_sweep", "two_condition", "automatic_ptc"):
            raise CaptureError("지원하지 않는 노이즈 실험 절차입니다.")
        if (
            not math.isfinite(self.anti_flicker_period_us)
            or not 0 <= self.anti_flicker_period_us <= 33333.334
        ):
            raise CaptureError(
                "플리커 회피 주기는 0(미사용) 또는 33333.334 µs 이하여야 합니다."
            )
        if self.protocol == "automatic_ptc" and asdict(
            replace(self, anti_flicker_period_us=0.0)
        ) != asdict(type(self).fixed_curve()):
            raise CaptureError(
                "자동 PTC 조건은 16단계 · 단계별 100장 · 최단 차광 1000장 · 0 dB · 256×256으로 고정됩니다."
            )
        if self.protocol == "two_condition" and (
            self.frames != 1000
            or self.bias_frames != 1000
            or self.levels != 1
            or self.roi_size != 256
            or self.gains_db != [0.0]
            or self.exposure_min_us != 10000.0
            or self.exposure_max_us != 10000.0
            or self.save_full_raw
            or self.sync_mode != "software"
            or self.max_flicker_percent != 0.1
            or self.wavelength_nm != 0.0
            or self.quantum_efficiency
        ):
            raise CaptureError(
                "노이즈 실험은 256×256 · 차광/밝은 각 1000장 · 카메라 시작값 고정입니다."
            )
        if any(
            type(v) is not int or not 4 <= v <= 100000
            for v in (self.frames, self.bias_frames)
        ):
            raise CaptureError("노이즈 반복 촬영 수는 4~100000이어야 합니다.")
        if type(self.levels) is not int or (
            self.protocol != "two_condition" and not 3 <= self.levels <= 64
        ):
            raise CaptureError("밝기 단계는 3~64개여야 합니다.")
        if not all(
            math.isfinite(v) and v > 0
            for v in (self.exposure_min_us, self.exposure_max_us)
        ) or (
            self.protocol != "two_condition"
            and self.exposure_min_us >= self.exposure_max_us
        ):
            raise CaptureError("최소·최대 노출시간은 0 < 최소 < 최대여야 합니다.")
        if (
            not self.gains_db
            or len(self.gains_db) > 16
            or len(set(self.gains_db)) != len(self.gains_db)
            or not all(math.isfinite(g) and 0 <= g <= 100 for g in self.gains_db)
        ):
            raise CaptureError("게인은 중복 없이 0~100 dB, 최대 16개를 지정하세요.")
        if (
            type(self.roi_size) is not int
            or (self.roi_size != 0 and not 16 <= self.roi_size <= 4096)
            or self.roi_size % 2
        ):
            raise CaptureError(
                "분석 ROI는 0(전체 센서) 또는 16~4096의 짝수 크기여야 합니다."
            )
        if type(self.save_full_raw) is not bool or self.sync_mode not in (
            "software",
            "ptp",
        ):
            raise CaptureError(
                "RAW 저장 범위 또는 노이즈 촬영 동기 방식이 잘못되었습니다."
            )
        if (
            not math.isfinite(self.max_flicker_percent)
            or not 0.01 <= self.max_flicker_percent <= 10
        ):
            raise CaptureError("허용 조명 변동은 0.01~10%여야 합니다.")
        if not math.isfinite(self.wavelength_nm) or not 0 <= self.wavelength_nm <= 2000:
            raise CaptureError("파장은 0(미지정)~2000 nm여야 합니다.")
        if not isinstance(self.quantum_efficiency, dict) or any(
            not re.fullmatch(r"[A-Za-z0-9_-]+", k)
            or not math.isfinite(v)
            or not 0 < v <= 1
            for k, v in self.quantum_efficiency.items()
        ):
            raise CaptureError(
                "QE는 카메라 시리얼별로 0보다 크고 1 이하인 비율을 입력하세요."
            )

    @classmethod
    def from_dict(cls, data):
        result = cls(**data)
        result.validate()
        return result

    def plan(self, phase="bright"):
        self.validate()
        if self.protocol == "two_condition":
            return [
                {
                    "index": i,
                    "gain_index": 0,
                    "level": i,
                    "gain_db": 0.0,
                    "exposure_us": 10000.0,
                    "illumination_target_fraction": 0.0 if phase == "dark" else 1.0,
                }
                for i in range(1)
            ]
        return [
            {
                "index": gi * self.levels + li,
                "gain_index": gi,
                "level": li,
                "gain_db": gain,
                "exposure_us": float(exposure),
            }
            for gi, gain in enumerate(self.gains_db)
            for li, exposure in enumerate(
                curve_times(self, self.exposure_min_us, self.exposure_max_us)
            )
        ]

    def stack_count(self, phase, point):
        if phase == "dark":
            count = self.dark_frame_counts.get(str(point["index"]), self.dark_frames or self.frames)
            return max(count, self.bias_frames) if point["level"] == 0 else count
        return self.bright_frame_counts.get(str(point["index"]), self.frames)

    def count(self, phase):
        return sum(self.stack_count(phase, p) for p in self.plan(phase))


def roi_bounds(shape, size):
    h, w = shape
    if min(h, w) < 16:
        raise CaptureError("RAW가 16×16 ROI보다 작습니다.")
    if size == 0:
        return (0, 0, w, h)
    size = min(size, h - h % 2, w - w % 2)
    if size < 16:
        raise CaptureError("RAW가 16×16 ROI보다 작습니다.")
    x, y = ((w - size) // 2) // 2 * 2, ((h - size) // 2) // 2 * 2
    return (x, y, size, size)


def curve_times(noise, lower, upper):
    period = noise.anti_flicker_period_us
    if not period:
        return np.geomspace(lower, upper, noise.levels)
    first = max(1, math.ceil(lower / period - 1e-9))
    last = math.floor(upper / period + 1e-9)
    if last - first + 1 < 3:
        raise CaptureError(
            "지원 범위 안에 플리커 회피 노출시간이 3개 미만입니다. 밝기별 곡선을 측정할 수 없습니다."
        )
    multiples = np.unique(
        np.rint(np.linspace(first, last, min(noise.levels, last - first + 1))).astype(
            int
        )
    )
    return multiples * period


def supported_curve_plan(noise, limits):
    """Fit the locked sequence to the cameras' common writable range."""
    lower = max(noise.exposure_min_us, *(r["exposure_us"][0] for r in limits))
    upper = min(noise.exposure_max_us, *(r["exposure_us"][1] for r in limits))
    gain = max(0.0, *(r["gain_db"][0] for r in limits))
    if lower >= upper or gain > min(r["gain_db"][1] for r in limits):
        raise CaptureError(
            "연결한 카메라 사이에 노이즈 순회를 위한 공통 노출시간·게인 범위가 없습니다."
        )
    return [
        {
            "index": i,
            "gain_index": 0,
            "level": i,
            "gain_db": float(gain),
            "exposure_us": float(t),
        }
        for i, t in enumerate(curve_times(noise, lower, upper))
    ]


def noise_step_preset(point, frozen=None):
    if frozen is not None:
        return frozen
    if "camera_presets" in point:
        return {
            serial: Preset(**values)
            for serial, values in point["camera_presets"].items()
        }
    return Preset(f"noise_{point['index']:03d}", point["exposure_us"], point["gain_db"])


def channel_slices(pixel_format):
    pattern = next(
        (p for p in ("RG", "GR", "GB", "BG") if pixel_format.startswith("Bayer" + p)),
        None,
    )
    if pattern is None:
        if pixel_format.startswith("Mono"):
            return {"Mono": (slice(None), slice(None))}
        raise CaptureError(
            f"노이즈 분석용 Bayer/Mono RAW 형식이 아닙니다: {pixel_format}"
        )
    labels = {
        "RG": ("R", "G1", "G2", "B"),
        "GR": ("G1", "R", "B", "G2"),
        "GB": ("G1", "B", "R", "G2"),
        "BG": ("B", "G1", "G2", "R"),
    }[pattern]
    return {
        label: (slice(y, None, 2), slice(x, None, 2))
        for label, (y, x) in zip(labels, ((0, 0), (0, 1), (1, 0), (1, 1)))
    }


class Moments:
    """Bounded-memory pixel moments, independent halves and disjoint frame pairs."""

    def __init__(self, shape, pixel_format, expected_count=100):
        self.count = 0
        self.mean, self.m2 = np.zeros(shape), np.zeros(shape)
        self.half_mean = np.zeros((2, *shape))
        self.half_m2 = np.zeros((2, *shape))
        self.half_count = np.zeros(2, dtype=np.int64)
        self.pair_sum = np.zeros(shape)
        self.previous = None
        self.clipped = np.zeros(shape, dtype=np.uint32)
        indices = np.arange(np.prod(shape)).reshape(shape)
        chosen = []
        channels = channel_slices(pixel_format)
        sample_count = min(512, max(4, 2000000 // expected_count // len(channels)))
        for slices in channels.values():
            values = indices[slices].ravel()
            chosen.extend(
                values[
                    np.linspace(
                        0, len(values) - 1, min(sample_count, len(values)), dtype=int
                    )
                ]
            )
        self.indices = np.array(chosen, dtype=np.int64)
        self.samples, self.frame_means, self.timestamps = [], [], []
        self.zero_count = 0

    def add(self, pixels, white, timestamp):
        values = pixels.astype(np.float64)
        parity = self.count % 2
        self.count += 1
        delta = values - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (values - self.mean)
        self.half_count[parity] += 1
        delta = values - self.half_mean[parity]
        self.half_mean[parity] += delta / self.half_count[parity]
        self.half_m2[parity] += delta * (values - self.half_mean[parity])
        if parity == 0:
            self.previous = values
        else:
            self.pair_sum += (values - self.previous) ** 2
            self.previous = None
        self.clipped += pixels >= white
        self.zero_count += np.count_nonzero(pixels == 0)
        self.samples.append(pixels.ravel()[self.indices].copy())
        self.frame_means.append(float(values.mean()))
        self.timestamps.append(timestamp)

    def arrays(self):
        return {
            "mean": self.mean,
            "variance": self.m2 / (self.count - 1),
            "pair_variance": self.pair_sum / (2 * (self.count // 2)),
            "half_mean": self.half_mean,
            "half_variance": self.half_m2 / (self.half_count[:, None, None] - 1),
            "half_count": self.half_count,
            "clipped": self.clipped,
            "sample_indices": self.indices,
            "sample_values": np.asarray(self.samples),
            "frame_means": np.asarray(self.frame_means),
            "timestamps_ns": np.asarray(self.timestamps, dtype=np.int64),
            "zero_fraction": np.array(self.zero_count / (self.count * self.mean.size)),
            "count": np.array(self.count),
        }


def save_npz(path, arrays):
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    write_bytes(path, buffer.getvalue())


def _phase_records(root, phase):
    path = root / f"{phase}.json"
    marker = json.loads((root / f"{phase.upper()}_COMPLETE.json").read_text())
    if sha256_file(path) != marker["sha256"]:
        raise CaptureError(f"{phase} 촬영 목록 체크섬이 다릅니다.")
    return json.loads(path.read_text())


def _session_identity(session):
    value = {
        key: session[key]
        for key in (
            "schema_version",
            "kind",
            "cameras",
            "noise_config",
            "capture_config",
        )
    }
    if "operating_settings" in session:
        value["operating_settings"] = session["operating_settings"]
    if "acquisition_plan" in session:
        value["acquisition_plan"] = session["acquisition_plan"]
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def _verify_illumination(root, record, meta, count, camera_serials):
    name = record.get("illumination_audit")
    if not name:
        if any("illumination" in f for f in meta["frames"]):
            raise CaptureError("조명 검사 기록이 촬영 목록에 없습니다.")
        return
    path = _safe_child(root, name)
    if sha256_file(path) != record["illumination_audit_sha256"]:
        raise CaptureError("조명 검사 기록 체크섬 불일치")
    audit = json.loads(path.read_text())
    observations = audit["observations"]
    accepted = [e for e in observations if e["kind"] == "accepted"]
    rejected = [e for e in observations if e["kind"] == "rejected"]
    pilot = [e for e in observations if e["kind"] == "pilot"]
    probes = [e for e in observations if e["kind"] == "phase_probe"]
    references = [e for e in observations if e["kind"] == "phase_reference"]
    unpaired = [e for e in observations if e["kind"] == "unpaired"]
    servo = [e for e in observations if e["kind"] == "phase_servo"]
    if (
        not audit["complete"]
        or audit["schema_version"] not in (1, 2, 3)
        or audit["required_groups"] != count
        or audit["accepted_groups"] != count
        or len(accepted) != count
        or not PILOT_GROUPS <= len(pilot) <= MAX_PILOT_GROUPS
        or len(pilot) != audit["pilot_groups"]
        or len(rejected) != audit["rejected_groups"]
        or len(probes) != audit.get("phase_probe_groups", 0)
        or (probes and audit["schema_version"] not in (2, 3))
        or len(references) != audit.get("phase_reference_groups", 0)
        or (references and audit["schema_version"] != 3)
        or len(unpaired) != audit.get("unpaired_groups", 0)
        or (unpaired and audit["schema_version"] != 3)
        or len(servo) != audit.get("phase_servo_groups", 0)
        or (servo and audit["schema_version"] != 3)
        or len(observations)
        != len(accepted)
        + len(rejected)
        + len(pilot)
        + len(probes)
        + len(references)
        + len(unpaired)
        + len(servo)
        or len(accepted)
        + len(rejected)
        + len(probes)
        + len(references)
        + len(unpaired)
        + len(servo)
        > audit["max_attempts"]
        or set(audit["models"]) != camera_serials
    ):
        raise CaptureError("조명 검사·제외·보충 장수 또는 카메라 목록이 다릅니다.")
    for entry in observations:
        if set(entry["cameras"]) != camera_serials:
            raise CaptureError("조명 검사 중 카메라 그룹이 누락되었습니다.")
        if entry["kind"] in ("pilot", "phase_probe", "phase_reference", "phase_servo"):
            continue
        models = audit["models"]
        if audit["schema_version"] == 3:
            index = entry["model_segment"]
            if not isinstance(index, int) or not 0 <= index < len(
                audit["model_segments"]
            ):
                raise CaptureError("조명 검사 구간이 다릅니다.")
            models = audit["model_segments"][index]["models"]
            if set(models) != camera_serials:
                raise CaptureError("조명 검사 구간의 카메라 목록이 다릅니다.")
        reasons = {
            s: reason
            for s, model in models.items()
            if (reason := light_rejection(entry["cameras"][s], model))
        }
        if reasons != entry["reasons"] or bool(reasons) != (
            entry["kind"] == "rejected"
        ):
            raise CaptureError("조명 검사 판정과 저장·제외 기록이 다릅니다.")
        if entry["kind"] == "unpaired" and entry.get("sampling_reason") not in (
            "independent calibration boundary",
            "partner photometry rejected",
            "acquisition ended before partner",
        ):
            raise CaptureError("차분 쌍에 포함하지 않은 프레임의 사유가 없습니다.")
    serial = record["serial"]
    ids = [e["cameras"][serial]["frame_id"] for e in observations]
    seed = audit.get("phase_seed")
    if seed:
        if (
            audit["schema_version"] not in (2, 3)
            or seed["serial"] != serial
            or len(seed["descriptions"]) < PILOT_GROUPS
        ):
            raise CaptureError("LED 위상 사전 검사 기록이 다릅니다.")
        ids = [e["frame_id"] for e in seed["descriptions"]] + ids
    if len(ids) != len(set(ids)) or any(b <= a for a, b in zip(ids, ids[1:])):
        raise CaptureError("조명 검사 프레임이 중복·역순입니다.")
    if audit["schema_version"] == 3:
        positions = {id(entry): i for i, entry in enumerate(observations)}
        for a, b in zip(accepted[::2], accepted[1::2]):
            if (
                a["model_segment"] != b["model_segment"]
                or positions[id(b)] != positions[id(a)] + 1
            ):
                raise CaptureError("차분 프레임 쌍 사이에 조명 기준이 바뀌었습니다.")
        accepted_ids = {e["cameras"][serial]["frame_id"] for e in accepted}
        reference_ids = {e["cameras"][serial]["frame_id"] for e in references}
        for segment in audit["model_segments"][1:]:
            ref = segment["reference_frame_ids"][serial]
            reference_count = (
                4
                if segment["reference_kind"]
                == "independent servo center; excluded from RAW statistics"
                else PILOT_GROUPS
            )
            if (
                len(ref) != reference_count
                or not set(ref) <= reference_ids
                or set(ref) & accepted_ids
            ):
                raise CaptureError("독립 조명 기준 프레임이 분석 RAW와 섞였습니다.")
    for index, (entry, frame) in enumerate(zip(accepted, meta["frames"])):
        if (
            entry["accepted_index"] != index
            or entry["cameras"][serial] != frame["illumination"]
            or frame["frame_id"] != entry["cameras"][serial]["frame_id"]
        ):
            raise CaptureError("정상 RAW와 조명 검사 프레임이 일치하지 않습니다.")


def verify_phase(path, phase, cancel=None):
    root = Path(path).resolve()
    cancel = cancel or Event()
    try:
        session = json.loads((root / "session.json").read_text())
        noise = NoiseConfig.from_dict(session["noise_config"])
        manifest = _phase_records(root, phase)
        if (
            session["kind"] != "rawrig_noise"
            or session["schema_version"] != 1
            or manifest["phase"] != phase
            or manifest["schema_version"] != 1
            or manifest["session_identity"] != _session_identity(session)
        ):
            raise CaptureError(
                "노이즈 세션의 카메라·설정·촬영 목록이 일치하지 않습니다."
            )
        if "light_preflight" in manifest:
            p = manifest["light_preflight"]
            preflight_path = _safe_child(root, p["path"])
            if sha256_file(preflight_path) != p["sha256"]:
                raise CaptureError("밝은 사전 검사 체크섬 불일치")
            audit = json.loads(preflight_path.read_text())
            if not audit.get("passed") or set(audit["illuminated_serials"]) != {
                c["serial"] for c in session["cameras"]
            }:
                raise CaptureError(
                    "밝은 사전 검사에서 일부 카메라의 빛을 확인하지 못했습니다."
                )
        plan = session.get("acquisition_plan", noise.plan(phase))
        point_plan = {p["index"]: p for p in plan}
        expected = {(p["index"], c["serial"]) for p in plan for c in session["cameras"]}
        keys = set()
        paths = set()
        for record in manifest["stacks"]:
            check_cancel(cancel)
            key = record["point"]["index"], record["serial"]
            if key in keys or key not in expected:
                raise CaptureError(
                    "노이즈 촬영 단계·카메라가 누락 또는 중복되었습니다."
                )
            keys.add(key)
            if record["point"] != point_plan[key[0]]:
                raise CaptureError("노이즈 노출시간·게인 단계가 촬영 계획과 다릅니다.")
            for name in ("raw", "stats", "metadata"):
                child = _safe_child(root, record[name])
                if child in paths or sha256_file(child) != record[name + "_sha256"]:
                    raise CaptureError(f"노이즈 파일 중복·체크섬 불일치: {child.name}")
                paths.add(child)
            meta = json.loads(_safe_child(root, record["metadata"]).read_text())
            expected_count = noise.stack_count(phase, record["point"])
            shape = meta["stored_shape"]
            if (
                meta["serial"] != record["serial"]
                or meta["point"] != record["point"]
                or meta["phase"] != phase
                or meta["count"] != expected_count
            ):
                raise CaptureError("노이즈 RAW 메타데이터가 촬영 계획과 다릅니다.")
            roi = meta["roi"]
            settings = meta["settings"]
            targets = record["point"].get("camera_presets")
            if targets:
                if set(targets) != {c["serial"] for c in session["cameras"]}:
                    raise CaptureError(
                        "노이즈 실제 적용 계획의 카메라가 일치하지 않습니다."
                    )
                target = targets[record["serial"]]
                if (
                    abs(settings["exposure_us"] - target["exposure_us"])
                    > max(1.0, target["exposure_us"] * 0.005)
                    or abs(settings["gain_db"] - target["gain_db"]) > 0.05
                ):
                    raise CaptureError(
                        "노이즈 실제 적용값이 검증한 센서 설정과 다릅니다."
                    )
            if roi != list(
                roi_bounds((settings["height"], settings["width"]), noise.roi_size)
            ) or shape != (
                [settings["height"], settings["width"]]
                if noise.save_full_raw
                else [roi[3], roi[2]]
            ):
                raise CaptureError("노이즈 RAW 저장 범위·분석 ROI가 계획과 다릅니다.")
            if (
                meta["dtype"] != "<u2"
                or len(shape) != 2
                or any(type(n) is not int or n <= 0 for n in shape)
            ):
                raise CaptureError("노이즈 RAW 크기·dtype이 잘못되었습니다.")
            raw = _safe_child(root, record["raw"])
            if raw.stat().st_size != expected_count * math.prod(shape) * 2:
                raise CaptureError("노이즈 RAW 프레임 수·바이트 수가 다릅니다.")
            ids = [f["frame_id"] for f in meta["frames"]]
            if (
                len(ids) != expected_count
                or len(set(ids)) != expected_count
                or any(b <= a for a, b in zip(ids, ids[1:]))
            ):
                raise CaptureError("노이즈 프레임이 누락·중복·역순입니다.")
            if phase == "bright":
                _verify_illumination(
                    root,
                    record,
                    meta,
                    expected_count,
                    {c["serial"] for c in session["cameras"]},
                )
            with np.load(
                _safe_child(root, record["stats"]), allow_pickle=False
            ) as stats:
                if (
                    int(stats["count"]) != expected_count
                    or list(stats["mean"].shape) != meta["roi"][2:][::-1]
                ):
                    raise CaptureError("노이즈 통계의 촬영 수·ROI가 다릅니다.")
        if keys != expected:
            raise CaptureError("일부 노이즈 촬영 단계 또는 카메라가 없습니다.")
        return session, manifest
    except (OSError, KeyError, TypeError, ValueError) as error:
        raise CaptureError(f"노이즈 세션을 검사하지 못했습니다: {error}") from error


def verify_noise_session(path, cancel=None):
    root = Path(path).resolve()
    session, dark = verify_phase(root, "dark", cancel)
    _, bright = verify_phase(root, "bright", cancel)
    try:
        if session["status"] != "complete":
            raise CaptureError("노이즈 분석이 완료되지 않았습니다.")
        marker = json.loads((root / "COMPLETE.json").read_text())
        for name in (
            "session.json",
            "dark.json",
            "bright.json",
            "analysis/manifest.json",
        ):
            if sha256_file(_safe_child(root, name)) != marker["sha256"][name]:
                raise CaptureError("노이즈 완료 목록 체크섬이 다릅니다.")
        manifest = json.loads((root / "analysis/manifest.json").read_text())
        for record in manifest["files"]:
            if cancel:
                check_cancel(cancel)
            if (
                sha256_file(_safe_child(root / "analysis", record["path"]))
                != record["sha256"]
            ):
                raise CaptureError(f"분석 결과 체크섬 불일치: {record['path']}")
        return {
            "path": str(root),
            "stacks": len(dark["stacks"]) + len(bright["stacks"]),
            "passed": True,
        }
    except (OSError, KeyError, TypeError, ValueError) as error:
        raise CaptureError(f"노이즈 결과를 검사하지 못했습니다: {error}") from error


def _compatible_settings(before, after):
    keys = (
        "width",
        "height",
        "pixel_format",
        "adc_bit_depth",
        "sensor_white_level",
        "gain_conversion_mode",
        "raw_processing",
        "black_level_percent",
        "black_level_clamping",
        "defect_pixel_correction",
        "sensor_modes",
    )
    return (
        all(before.get(k) == after.get(k) for k in keys)
        and abs(before["exposure_us"] - after["exposure_us"])
        <= max(2.0, before["exposure_us"] * 0.001)
        and abs(before["gain_db"] - after["gain_db"]) <= 0.05
    )


def _reuse_dark_point(source, root, folder, point, settings, noise, cancel, progress):
    """Adopt only complete stacks after checking source hashes or legacy RAW moments."""
    if not source:
        return []
    source = Path(source).resolve()
    try:
        old = json.loads((source / "session.json").read_text())
        current = json.loads((root / "session.json").read_text())
        if (
            old["kind"] != "rawrig_noise"
            or NoiseConfig.from_dict(old["noise_config"]).protocol != noise.protocol
            or old["cameras"] != current["cameras"]
        ):
            return []
        committed = {}
        if (source / "DARK_COMPLETE.json").is_file():
            data = _phase_records(source, "dark")
            if data["session_identity"] != _session_identity(old):
                raise CaptureError("기존 차광 완료 목록 설정 불일치")
            committed = {r["metadata"]: r for r in data["stacks"]}
        elif (source / "DARK_CHECKPOINT.json").is_file():
            marker = json.loads((source / "DARK_CHECKPOINT.json").read_text())
            checkpoint = source / "dark_checkpoint.json"
            if sha256_file(checkpoint) != marker["sha256"]:
                raise CaptureError("기존 차광 체크포인트 체크섬 불일치")
            data = json.loads(checkpoint.read_text())
            if data["session_identity"] != _session_identity(old):
                raise CaptureError("기존 차광 체크포인트 설정 불일치")
            committed = {r["metadata"]: r for r in data["stacks"]}
        count = noise.stack_count("dark", point)
        sources = []
        for s in settings:
            check_cancel(cancel)
            candidates = sorted(
                source.glob(f"dark_*/point_{point['index']:03d}/{s['serial']}.json"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if not candidates:
                return []
            metadata = _safe_child(source, str(candidates[0].relative_to(source)))
            meta = json.loads(metadata.read_text())
            for key in ("index", "gain_index", "level", "exposure_us", "gain_db"):
                if meta["point"][key] != point[key]:
                    return []
            roi = roi_bounds((s["height"], s["width"]), noise.roi_size)
            shape = (
                [s["height"], s["width"]] if noise.save_full_raw else [roi[3], roi[2]]
            )
            if (
                meta["phase"] != "dark"
                or meta["serial"] != s["serial"]
                or meta["count"] != count
                or meta["roi"] != list(roi)
                or meta["stored_shape"] != shape
                or meta["dtype"] != "<u2"
                or not _compatible_settings(meta["settings"], s)
            ):
                return []
            raw, stats = metadata.with_suffix(".raw"), metadata.with_suffix(".npz")
            raw = _safe_child(source, str(raw.relative_to(source)))
            stats = _safe_child(source, str(stats.relative_to(source)))
            if (
                raw.stat().st_size != count * math.prod(shape) * 2
                or len(meta["frames"]) != count
            ):
                raise CaptureError("기존 차광 RAW 촬영 수 불일치")
            ids = [f["frame_id"] for f in meta["frames"]]
            if any(b <= a for a, b in zip(ids, ids[1:])):
                raise CaptureError("기존 차광 프레임 중복/역순")
            for f in meta["frames"]:
                if (
                    f["transport"].get("incomplete")
                    or f["transport"].get("crc_passed") is False
                ):
                    raise CaptureError("기존 차광 전송 검사 실패")
                for key, tolerance in (
                    ("exposure_us", max(1.0, s["exposure_us"] * 0.005)),
                    ("gain_db", 0.05),
                ):
                    value = f["chunk"].get(key)
                    if value is not None and (
                        not math.isfinite(value) or abs(value - s[key]) > tolerance
                    ):
                        raise CaptureError("기존 차광 chunk 설정 불일치")
            hashes = {
                name: sha256_file(path)
                for name, path in (
                    ("raw", raw),
                    ("stats", stats),
                    ("metadata", metadata),
                )
            }
            record = committed.get(str(metadata.relative_to(source)))
            if record:
                if any(hashes[name] != record[name + "_sha256"] for name in hashes):
                    raise CaptureError("기존 차광 RAW/통계/메타데이터 체크섬 불일치")
            else:
                # Older versions saved complete points without a checkpoint.
                # Rebuild every cached statistic from RAW before trusting that point.
                progress(f"{s['serial']}: 기존 차광 {count}장 RAW·통계 대조 중…")
                accumulator = Moments((roi[3], roi[2]), s["pixel_format"], count)
                pixels = np.memmap(raw, dtype="<u2", mode="r", shape=(count, *shape))
                try:
                    for array, f in zip(pixels, meta["frames"]):
                        check_cancel(cancel)
                        if noise.save_full_raw:
                            x, y, w, h = roi
                            array = array[y : y + h, x : x + w]
                        accumulator.add(
                            array,
                            s.get("sensor_white_level", s.get("pixel_max", 65535)),
                            f["host_received_ns"],
                        )
                    rebuilt = accumulator.arrays()
                finally:
                    del pixels
                with np.load(stats, allow_pickle=False) as cached:
                    if set(cached.files) != set(rebuilt) or any(
                        not np.allclose(
                            cached[k], v, rtol=1e-12, atol=1e-9, equal_nan=False
                        )
                        for k, v in rebuilt.items()
                    ):
                        raise CaptureError("기존 차광 RAW와 통계가 다릅니다.")
            sources.append((raw, stats, meta, hashes))
    except Cancelled:
        raise
    except (CaptureError, OSError, KeyError, ValueError, TypeError) as error:
        progress(f"기존 차광 재사용 제외 · 해당 단계 재촬영: {error}")
        return []
    folder.mkdir(parents=True)
    records = []
    for raw, stats, meta, hashes in sources:
        check_cancel(cancel)
        serial = meta["serial"]
        copied = {
            "raw": folder / f"{serial}.raw",
            "stats": folder / f"{serial}.npz",
            "metadata": folder / f"{serial}.json",
        }
        shutil.copyfile(raw, copied["raw"])
        shutil.copyfile(stats, copied["stats"])
        for name in ("raw", "stats"):
            with copied[name].open("rb") as stream:
                os.fsync(stream.fileno())
            if sha256_file(copied[name]) != hashes[name]:
                raise CaptureError("재사용 차광 복사 후 체크섬 불일치")
        meta = {
            **meta,
            "point": point,
            "reused_from": {"session": str(source), "sha256": hashes},
        }
        write_json(copied["metadata"], meta)
        record = {"serial": serial, "point": point}
        for name, path in copied.items():
            record[name] = str(path.relative_to(root))
            record[name + "_sha256"] = sha256_file(path)
        records.append(record)
    fsync_dir(folder)
    return records


def _save_noise_checkpoint(root, phase, session, records):
    checkpoint = root / f"{phase}_checkpoint.json"
    if phase == "bright" and checkpoint.exists():
        try:
            marker = json.loads((root / "BRIGHT_CHECKPOINT.json").read_text())
            prior = json.loads(checkpoint.read_text())
            if (
                sha256_file(checkpoint) == marker["sha256"]
                and prior["session_identity"] == _session_identity(session)
                and prior["phase"] == phase
            ):
                by_point = {
                    (r["point"]["index"], r["serial"]): r
                    for r in prior["stacks"] + records
                }
                records = [by_point[key] for key in sorted(by_point)]
        except (OSError, KeyError, ValueError, TypeError):
            # Corrupted checkpoint data cannot authorize reuse of its RAW files.
            pass
    write_json(
        checkpoint,
        {
            "schema_version": 1,
            "phase": phase,
            "session_identity": _session_identity(session),
            "stacks": records,
        },
    )
    write_json(
        root / f"{phase.upper()}_CHECKPOINT.json", {"sha256": sha256_file(checkpoint)}
    )


def _reuse_bright_point(root, point, settings, noise, cancel, progress):
    """Resume only committed, hash-verified groups from this unfinished session."""
    if (root / "BRIGHT_COMPLETE.json").exists() or not (
        root / "BRIGHT_CHECKPOINT.json"
    ).exists():
        return []
    try:
        marker = json.loads((root / "BRIGHT_CHECKPOINT.json").read_text())
        checkpoint = root / "bright_checkpoint.json"
        if sha256_file(checkpoint) != marker["sha256"]:
            raise CaptureError("밝은 체크포인트 체크섬 불일치")
        data = json.loads(checkpoint.read_text())
        session = json.loads((root / "session.json").read_text())
        if data["phase"] != "bright" or data["session_identity"] != _session_identity(
            session
        ):
            raise CaptureError("밝은 체크포인트 설정 불일치")
        records = [r for r in data["stacks"] if r["point"] == point]
        serials = {s["serial"] for s in settings}
        if len(records) != len(serials) or {r["serial"] for r in records} != serials:
            return []
        current = {s["serial"]: s for s in settings}
        for record in records:
            check_cancel(cancel)
            for key in ("raw", "stats", "metadata"):
                if (
                    sha256_file(_safe_child(root, record[key]))
                    != record[key + "_sha256"]
                ):
                    raise CaptureError("밝은 RAW·통계·메타데이터 체크섬 불일치")
            meta = json.loads(_safe_child(root, record["metadata"]).read_text())
            s = current[record["serial"]]
            roi = roi_bounds((s["height"], s["width"]), noise.roi_size)
            shape = (
                [s["height"], s["width"]] if noise.save_full_raw else [roi[3], roi[2]]
            )
            if (
                meta["serial"] != record["serial"]
                or meta["phase"] != "bright"
                or meta["point"] != point
                or meta["count"] != noise.stack_count("bright", point)
                or len(meta["frames"]) != noise.stack_count("bright", point)
                or meta["roi"] != list(roi)
                or meta["stored_shape"] != shape
                or meta["dtype"] != "<u2"
                or not _compatible_settings(meta["settings"], s)
                or _safe_child(root, record["raw"]).stat().st_size
                != noise.stack_count("bright", point) * math.prod(shape) * 2
            ):
                raise CaptureError("밝은 촬영 설정·RAW 크기 불일치")
            _verify_illumination(root, record, meta, noise.stack_count("bright", point), serials)
        return records
    except Cancelled:
        raise
    except (CaptureError, OSError, KeyError, ValueError, TypeError) as error:
        progress(f"기존 밝은 단계 재사용 제외 · 해당 단계만 재촬영: {error}")
        return []


def fork_bright_session(path, cancel=None, progress=None):
    """Preserve a finished measurement and copy only its verified dark stacks."""
    cancel = cancel or Event()
    progress = progress or (lambda *args: None)
    source = Path(path).resolve()
    session, dark = verify_phase(source, "dark", cancel)
    if session["status"] != "complete":
        return source
    verify_noise_session(source, cancel)
    root = source.parent / (
        "noise_"
        + time.strftime("%Y%m%dT%H%M%S", time.gmtime())
        + "_"
        + uuid.uuid4().hex[:8]
        + ".partial"
    )
    root.mkdir()
    try:
        names = {"dark.json", "DARK_COMPLETE.json"}
        names.update(r[k] for r in dark["stacks"] for k in ("raw", "stats", "metadata"))
        for index, name in enumerate(sorted(names)):
            check_cancel(cancel)
            original, target = _safe_child(source, name), _safe_child(root, name)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(original, target)
            with target.open("rb") as stream:
                os.fsync(stream.fileno())
            if sha256_file(original) != sha256_file(target):
                raise CaptureError("밝은 재촬영용 차광 복사 후 체크섬 불일치")
            progress(
                "기존 차광 RAW 재검사·복사 · 밝은 촬영만 다시 시작", index, len(names)
            )
        session = {
            **session,
            "status": "awaiting_bright",
            "created_at_utc": utc_now(),
            "bright_retake_from": {
                "path": str(source),
                "session_sha256": sha256_file(source / "session.json"),
            },
        }
        write_json(root / "session.json", session)
        verify_phase(root, "dark", cancel)
        fsync_dir(root)
        return root
    except Exception:
        # Only this newly created copy is disposable; source files are never touched.
        shutil.rmtree(root)
        raise


def _check_bright_source(
    backend,
    cfg,
    noise,
    plan,
    frozen,
    dark_records,
    root,
    attempt,
    cancel,
    progress,
    preview,
):
    """Detect a still-capped/dark scene at the longest matched exposure first."""
    point = plan[-1]
    preset = noise_step_preset(point, frozen)
    settings = backend.prepare(preset, cfg)
    dark_levels = {}
    for s in settings:
        r = dark_records[point["index"], s["serial"]]
        old = json.loads(_safe_child(root, r["metadata"]).read_text())
        if not _compatible_settings(old["settings"], s):
            raise CaptureError("밝은 사전 검사와 차광의 센서 설정이 다릅니다.")
        with np.load(_safe_child(root, r["stats"]), allow_pickle=False) as d:
            dark_levels[s["serial"]] = (
                float(d["mean"].mean()),
                max(2.0, 6 * math.sqrt(float(d["variance"].mean()) / d["mean"].size)),
            )
    audit = {
        "schema_version": 1,
        "point": point,
        "frames": [],
        "illuminated_serials": [],
    }
    folder = root / attempt
    folder.mkdir()
    if hasattr(backend, "noise_phase"):
        backend.noise_phase = "bright"
    backend.start()
    previous, lit = {}, set()
    rng = random.Random()
    try:
        if cancel.wait(cfg.settle_ms / 1000):
            check_cancel(cancel)
        for index in range(32):
            if not any(c.simulated for c in backend.cameras):
                cancel.wait(rng.uniform(0, 0.019))
            frames, _ = capture_checked(
                backend,
                preset,
                cfg,
                cancel,
                previous,
                lambda msg: progress(msg, 0, noise.count("bright")),
            )
            for f in frames:
                s = f.settings
                if f.pixels.dtype != np.uint16 or f.pixels.shape != (
                    s["height"],
                    s["width"],
                ):
                    raise CaptureError("밝은 사전 검사 RAW 형식·크기가 다릅니다.")
                level, threshold = dark_levels[f.camera.serial]
                d = describe_light(f, roi_bounds(f.pixels.shape, noise.roi_size), level)
                audit["frames"].append(d)
                if d["roi_mean_dn"] > level + threshold:
                    lit.add(f.camera.serial)
            audit["illuminated_serials"] = sorted(lit)
            if preview and index % 8 == 0:
                preview(frames)
            progress(
                f"밝은 장면 확인 {index + 1}/32 · 차광 상태 오촬영 방지",
                0,
                noise.count("bright"),
            )
            if lit == set(dark_levels):
                break
        if lit != set(dark_levels):
            missing = ", ".join(sorted(set(dark_levels) - lit))
            raise CaptureError(
                f"{missing}: 긴 노출시간에서도 암영상 수준입니다. 렌즈 캡을 제거하고 밝은 면을 비춘 뒤 ②를 다시 누르세요. 차광 자료는 유지됩니다."
            )
        audit["passed"] = True
    finally:
        write_json(folder / "LIGHT_PREFLIGHT.json", audit)
        backend.stop()
    return {
        "path": str((folder / "LIGHT_PREFLIGHT.json").relative_to(root)),
        "sha256": sha256_file(folder / "LIGHT_PREFLIGHT.json"),
    }


def run_noise_phase(
    backend,
    capture_config,
    noise,
    phase,
    session_path=None,
    cancel=None,
    progress=None,
    preview=None,
):
    """Acquire each fixed setting once; no auto exposure/WB or frame averaging in RAW."""
    noise.validate()
    if phase not in ("dark", "bright") or not backend or not backend.cameras:
        raise CaptureError("노이즈 측정 카메라를 연결하세요.")
    cancel = cancel or Event()
    progress = progress or (lambda *args: None)
    cfg = replace(
        capture_config,
        expected_cameras=len(backend.cameras),
        auto_reference=False,
        capture_strategy="bracket",
        dslr_enabled=False,
        sync_mode=noise.sync_mode,
    )
    cfg.validate()
    plan = noise.plan(phase)
    frozen = backend.operating_presets() if noise.protocol == "two_condition" else None
    limits = backend.limits()
    if noise.protocol == "automatic_ptc":
        if phase == "bright":
            saved, _ = verify_phase(session_path, "dark", cancel)
            plan = saved["acquisition_plan"]
        else:
            plan = supported_curve_plan(noise, limits)
    elif phase == "bright" and noise.protocol == "exposure_sweep":
        saved, _ = verify_phase(session_path, "dark", cancel)
        plan = saved.get("acquisition_plan", plan)
    for limit in limits:
        if any(
            not limit["exposure_us"][0] <= p["exposure_us"] <= limit["exposure_us"][1]
            or not limit["gain_db"][0] <= p["gain_db"] <= limit["gain_db"][1]
            for p in (
                plan
                if frozen is None
                else [
                    {
                        "exposure_us": frozen[limit["serial"]].exposure_us,
                        "gain_db": frozen[limit["serial"]].gain_db,
                    }
                ]
            )
        ):
            raise CaptureError(
                f"{limit['serial']}: 노이즈 노출시간·게인이 지원 범위 밖입니다. 값을 조정하세요."
            )
    root = None
    records = []
    total = noise.count(phase)
    done = 0
    try:
        backend.stop()
        check_cancel(cancel)
        if phase == "bright":
            root = Path(session_path).resolve()
            session, dark = verify_phase(root, "dark", cancel)
            if (
                asdict(NoiseConfig.from_dict(session["noise_config"])) != asdict(noise)
                or [asdict(c) for c in backend.cameras] != session["cameras"]
            ):
                raise CaptureError(
                    "차광 촬영과 카메라·측정 설정이 다릅니다. 기존 설정을 불러오거나 차광 촬영을 다시 시작하세요."
                )
            if session["status"] == "complete":
                root = fork_bright_session(root, cancel, progress)
                session, dark = verify_phase(root, "dark", cancel)
            dark_records = {
                (r["point"]["index"], r["serial"]): r for r in dark["stacks"]
            }
        if frozen is not None and phase == "bright":
            frozen = {
                serial: Preset(**data)
                for serial, data in session["operating_settings"].items()
            }
        profiles = backend.configure_profiles(cfg)
        if (
            phase == "dark"
            and noise.protocol in ("automatic_ptc", "exposure_sweep")
            and hasattr(backend, "resolve_noise_plan")
        ):
            progress(f"{len(plan)}개 노출시간을 센서에 적용·재확인합니다…", 0, total)
            plan = backend.resolve_noise_plan(plan, cfg, cancel)
        if noise.anti_flicker_period_us and noise.protocol == "automatic_ptc":
            period = noise.anti_flicker_period_us
            for point in plan:
                values = point.get("camera_presets", {}).values()
                actual_times = [p["exposure_us"] for p in values] or [
                    point["exposure_us"]
                ]
                if any(
                    abs(t - round(t / period) * period) > max(10.0, period * 0.002)
                    for t in actual_times
                ):
                    raise CaptureError(
                        "센서의 실제 노출시간이 플리커 회피 주기에서 벗어납니다. 촬영을 시작하지 않았습니다."
                    )
        total = sum(noise.stack_count(phase, p) for p in plan)
        sync = backend.synchronize(cfg, cancel, lambda msg: progress(msg, done, total))
        initial = backend.prepare(
            noise_step_preset(plan[0], frozen),
            cfg,
        )
        if phase == "dark":
            output = Path(cfg.output_dir).expanduser().resolve() / "noise_sessions"
            output.mkdir(parents=True, exist_ok=True)
            root = output / (
                "noise_"
                + time.strftime("%Y%m%dT%H%M%S", time.gmtime())
                + "_"
                + uuid.uuid4().hex[:8]
                + ".partial"
            )
            root.mkdir()
            session = {
                "schema_version": 1,
                "kind": "rawrig_noise",
                "status": "acquiring_dark",
                "created_at_utc": utc_now(),
                "paper": PAPER_URL,
                "noise_config": asdict(noise),
                "capture_config": cfg.to_dict(),
                "cameras": [asdict(c) for c in backend.cameras],
                "simulated": any(c.simulated for c in backend.cameras),
                "method": "two conditions (dark and bright), camera initial exposure/gain frozen; not full Fig.6 PTC"
                if frozen is not None
                else "fixed-light exposure-time sweep; matched dark at every exposure and gain",
                "profiles": profiles,
                "synchronization": sync,
                "units": "unaltered stored RAW DN; Bayer planes analyzed separately",
            }
            if frozen is not None:
                session["operating_settings"] = {
                    serial: asdict(preset) for serial, preset in frozen.items()
                }
            if noise.protocol != "two_condition":
                session["acquisition_plan"] = plan
            write_json(root / "session.json", session)
        roi_pixels = [
            math.prod(roi_bounds((s["height"], s["width"]), noise.roi_size)[2:])
            for s in initial
        ]
        raw_pixels = [
            s["height"] * s["width"] if noise.save_full_raw else pixels
            for s, pixels in zip(initial, roi_pixels)
        ]
        remaining_groups = total + (noise.count("bright") if phase == "dark" else 0)
        remaining_stacks = len(plan) * (2 if phase == "dark" else 1)
        # Reserve RAW, uncompressed statistics/metadata and the largest camera's
        # temporary memory-map working set before starting a long experiment.
        raw_estimate = sum(raw_pixels) * 2 * remaining_groups
        stats_estimate = (
            sum(roi_pixels) * 64 * remaining_stacks
            + remaining_groups * len(initial) * 8192
        )
        largest = max(roi_pixels)
        scratch = (
            largest * 128 * noise.levels if largest * 8 > 1024**2 else 16 * 1024**2
        )
        estimate = raw_estimate + stats_estimate + scratch
        if shutil.disk_usage(root).free < estimate * 1.1 + 128 * 1024**2:
            raise CaptureError(
                f"노이즈 저장 공간 부족: 남은 촬영·통계·분석 작업 공간 약 {estimate / 1024**3:.2f} GiB가 필요합니다."
            )
        attempt = f"{phase}_{uuid.uuid4().hex[:8]}"
        preflight = (
            _check_bright_source(
                backend,
                cfg,
                noise,
                plan,
                frozen,
                dark_records,
                root,
                attempt,
                cancel,
                progress,
                preview,
            )
            if phase == "bright"
            else None
        )
        for point in plan:
            check_cancel(cancel)
            preset = noise_step_preset(point, frozen)
            settings = backend.prepare(preset, cfg)
            if phase == "bright":
                dark_means = {}
                matched_dark_stats = {}
                for s in settings:
                    old = json.loads(
                        _safe_child(
                            root, dark_records[point["index"], s["serial"]]["metadata"]
                        ).read_text()
                    )
                    if not _compatible_settings(old["settings"], s):
                        raise CaptureError(
                            f"{s['serial']}: 암·밝은 영상의 센서/RAW/노출 컨트롤 설정이 다릅니다."
                        )
                    with np.load(
                        _safe_child(
                            root, dark_records[point["index"], s["serial"]]["stats"]
                        ),
                        allow_pickle=False,
                    ) as stats:
                        dark_means[s["serial"]] = float(stats["mean"].mean())
                        matched_dark_stats[s["serial"]] = {
                            key: stats[key]
                            for key in (
                                "mean",
                                "variance",
                                "pair_variance",
                                "frame_means",
                                "count",
                            )
                        }
            folder = root / attempt / f"point_{point['index']:03d}"
            count = noise.stack_count(phase, point)
            if phase == "dark" and noise.protocol != "two_condition":
                reused = _reuse_dark_point(
                    session_path,
                    root,
                    folder,
                    point,
                    settings,
                    noise,
                    cancel,
                    lambda msg: progress(msg, done, total),
                )
                if reused:
                    records.extend(reused)
                    done += count
                    _save_noise_checkpoint(root, phase, session, records)
                    progress(
                        f"차광 {count}장 무결성 검증·재사용 완료 · 남은 단계 이어 촬영",
                        done,
                        total,
                    )
                    continue
            if phase == "bright" and noise.protocol != "two_condition":
                reused = _reuse_bright_point(
                    root,
                    point,
                    settings,
                    noise,
                    cancel,
                    lambda msg: progress(msg, done, total),
                )
                if reused:
                    records.extend(reused)
                    done += count
                    progress(
                        f"밝은 {count}장 무결성 검증·재사용 · 남은 단계 이어 촬영",
                        done,
                        total,
                    )
                    continue
            folder.mkdir(parents=True)
            accumulators, streams, meta, digests = {}, {}, {}, {}
            illumination = {} if phase == "bright" else None
            try:
                for s in settings:
                    serial = s["serial"]
                    if not re.fullmatch(r"[A-Za-z0-9_-]+", serial):
                        raise CaptureError("카메라 시리얼이 안전한 파일명이 아닙니다.")
                    roi = roi_bounds((s["height"], s["width"]), noise.roi_size)
                    shape = (
                        [s["height"], s["width"]]
                        if noise.save_full_raw
                        else [roi[3], roi[2]]
                    )
                    accumulators[serial] = Moments(
                        (roi[3], roi[2]), s["pixel_format"], count
                    )
                    streams[serial] = (folder / f"{serial}.raw").open("xb")
                    digests[serial] = hashlib.sha256()
                    meta[serial] = {
                        "serial": serial,
                        "phase": phase,
                        "point": point,
                        "count": count,
                        "dtype": "<u2",
                        "stored_shape": shape,
                        "roi": list(roi),
                        "full_raw": noise.save_full_raw,
                        "settings": s,
                        "frames": [],
                    }
                if hasattr(backend, "noise_phase"):
                    backend.noise_phase = phase
                backend.start()
                if cancel.wait(cfg.settle_ms / 1000):
                    check_cancel(cancel)
                previous_ids = {}
                for _ in range(cfg.warmup_frames):
                    capture_checked(backend, preset, cfg, cancel, previous_ids)

                def acquire():
                    frames, trigger = capture_checked(
                        backend,
                        preset,
                        cfg,
                        cancel,
                        previous_ids,
                        lambda msg: progress(msg, done, total),
                    )
                    check_cancel(cancel)
                    for frame in frames:
                        serial = frame.camera.serial
                        s = meta[serial]["settings"]
                        if (
                            frame.pixels.dtype != np.dtype("uint16")
                            or frame.pixels.shape != (s["height"], s["width"])
                            or not _compatible_settings(s, frame.settings)
                        ):
                            raise CaptureError(
                                f"{serial}: 측정 도중 RAW 형식·촬영 설정이 바뀌었습니다."
                            )
                        x, y, w, h = meta[serial]["roi"]
                        roi_pixels = frame.pixels[y : y + h, x : x + w]
                        if int(np.prod(roi_pixels.shape)) != w * h:
                            raise CaptureError("노이즈 분석 ROI가 RAW 밖에 있습니다.")
                    return frames, trigger

                def describe(frames):
                    descriptions = {
                        f.camera.serial: describe_light(
                            f, meta[f.camera.serial]["roi"], dark_means[f.camera.serial]
                        )
                        for f in frames
                    }
                    for serial, description in descriptions.items():
                        stats = matched_dark_stats[serial]
                        common = (
                            float(np.var(stats["frame_means"], ddof=1))
                            - float(stats["variance"].mean()) / stats["mean"].size
                        )
                        description["dark_common_mean_sigma_dn"] = math.sqrt(
                            max(common, 0.0)
                        )
                    return descriptions

                phase_seed = None
                phase_reference_capture = None
                if (
                    phase == "bright"
                    and noise.protocol == "exposure_sweep"
                    and hasattr(backend, "set_light_phase")
                    and len(backend.cameras) == 1
                ):
                    from .light_phase import fit_light_phase, phase_for_exposure

                    serial = backend.cameras[0].serial
                    target_us=meta[serial]['settings']['exposure_us']
                    if (target_us<1351 and hasattr(backend,'probe_timestamp_position') and
                        getattr(backend,'camera_timestamp_position',None) is None):
                        progress('독립 조명 기준 전환 · 실제 노출시간·FrameID·timestamp 검증',done,total)
                        clock_probe=backend.probe_timestamp_position(cfg.timeout_ms,cancel)
                        write_json(root/'TIMESTAMP_REFERENCE_PROBE.json',clock_probe)
                    timestamp_position=getattr(backend,'camera_timestamp_position',None)
                    use_reference=(target_us<1351 and hasattr(backend,'capture_light_reference')
                                   and timestamp_position in ('exposure_start','exposure_end'))
                    if use_reference:
                        def phase_reference_capture():
                            return backend.capture_light_reference(1351.,cfg.timeout_ms,cancel)
                    pilot = []
                    for _ in range(PILOT_GROUPS):
                        if cancel.wait(random.uniform(0, 0.019)):
                            check_cancel(cancel)
                        frames, _ = phase_reference_capture() if use_reference else acquire()
                        pilot.append(describe(frames)[serial])
                    model = fit_light_phase(pilot)
                    reference_model=model if use_reference else None
                    if reference_model:
                        actual_reference_us=frames[0].settings['exposure_us']
                        model=phase_for_exposure(reference_model,actual_reference_us,target_us,timestamp_position)
                    if model and backend.set_light_phase(model):
                        phase_seed = {
                            "serial": serial,
                            "model": model,
                            "descriptions": pilot,
                            **({'reference_model':reference_model,'reference_exposure_us':actual_reference_us,
                                'target_exposure_us':target_us,'timestamp_position':timestamp_position} if reference_model else {}),
                        }
                        progress(
                            f"LED {model['frequency_hz']:.5f} Hz 위상 측정 · 노출시간·게인 유지",
                            done,
                            total,
                        )

                groups = (
                    lit_groups(
                        acquire,
                        describe,
                        count,
                        noise.max_flicker_percent,
                        cancel,
                        illumination,
                        lambda msg: progress(msg, done, total),
                        preview,
                        lowest_step=point["level"] == 0,
                        simulated=session["simulated"],
                        phase_control=backend.set_light_phase
                        if noise.protocol == "exposure_sweep"
                        and hasattr(backend, "set_light_phase")
                        and len(backend.cameras) == 1
                        else None,
                        phase_seed=phase_seed,
                        local_pair_protocol=noise.protocol == "exposure_sweep"
                        and (noise.frames >= 1000 or noise.levels == 32 or bool(noise.bright_frame_counts)),
                        fast_stack=bool(noise.bright_frame_counts) and count <= 100,
                        phase_reference_capture=phase_reference_capture,
                    )
                    if phase == "bright"
                    else (
                        (frames, trigger, None)
                        for frames, trigger in (acquire() for _ in range(count))
                    )
                )
                for repeat, (frames, trigger, descriptions) in enumerate(groups):
                    for frame in frames:
                        serial = frame.camera.serial
                        s = meta[serial]["settings"]
                        x, y, w, h = meta[serial]["roi"]
                        roi_pixels = frame.pixels[y : y + h, x : x + w]
                        pixels = frame.pixels if noise.save_full_raw else roi_pixels
                        payload = pixels.astype("<u2", copy=False).tobytes()
                        streams[serial].write(payload)
                        digests[serial].update(payload)
                        white = s.get("sensor_white_level", s.get("pixel_max", 65535))
                        accumulators[serial].add(
                            roi_pixels, white, frame.host_received_ns
                        )
                        meta[serial]["frames"].append(
                            {
                                "frame_id": frame.frame_id,
                                "camera_timestamp_ns": frame.camera_timestamp_ns,
                                "host_received_ns": frame.host_received_ns,
                                "chunk": frame.chunk,
                                "transport": frame.transport,
                                "trigger": trigger,
                                **(
                                    {"illumination": descriptions[serial]}
                                    if descriptions
                                    else {}
                                ),
                            }
                        )
                    done += 1
                    if preview and (repeat % 20 == 0 or repeat == count - 1):
                        preview(frames)
                    progress(
                        f"{'차광' if phase == 'dark' else '밝은 장면'} · "
                        + (
                            "노출시간·게인 시작값 고정"
                            if frozen is not None
                            else " / ".join(
                                (f"{s['serial']}: " if len(settings) > 1 else "")
                                + f"게인 {s['gain_db']:g} dB · 노출시간 {s['exposure_us']:g} µs"
                                for s in settings
                            )
                        )
                        + f" · 정상 {repeat + 1}/{count}"
                        + (
                            f" · 조명 이상 제외 {illumination['rejected_groups']}회·자동 보충"
                            if illumination and illumination["rejected_groups"]
                            else ""
                        ),
                        done,
                        total,
                    )
                backend.stop()
                if illumination is not None:
                    estimator = (
                        excess_pair_flicker
                        if illumination["schema_version"] == 3
                        else excess_flicker
                    )
                    illumination["whole_stack_excess_flicker_fraction"] = {
                        s: excess_flicker(
                            a.arrays(), dark_means[s], dark_stats=matched_dark_stats[s]
                        )
                        for s, a in accumulators.items()
                    }
                    illumination["excess_flicker_fraction"] = {
                        s: estimator(
                            a.arrays(), dark_means[s], dark_stats=matched_dark_stats[s]
                        )
                        for s, a in accumulators.items()
                    }
                    illumination["excess_flicker_lower_bound_fraction"] = {
                        s: estimator(
                            a.arrays(),
                            dark_means[s],
                            confidence_margin=True,
                            dark_stats=matched_dark_stats[s],
                        )
                        for s, a in accumulators.items()
                    }
                    # At near-dark signal, relative brightness fluctuation is
                    # ill-conditioned. Bound its effect on pixel noise variance
                    # directly at every explicitly configured sample count.
                    fast_quality = bool(noise.bright_frame_counts)
                    if fast_quality:
                        illumination["noise_variance_bias_upper_fraction"] = {
                            serial: pair_noise_bias_upper(
                                accumulator.arrays(), matched_dark_stats[serial],
                                channel_slices(meta[serial]["settings"]["pixel_format"]),
                            )
                            for serial, accumulator in accumulators.items()
                        }
                        illumination["noise_variance_bias_budget_fraction"] = 0.001
                        illumination["quality_acceptance"] = "legacy pair photometric stability OR four-SEM upper common illumination variance / CFA pixel pair variance; fixed LED spectrum"

                    unstable = {
                        s: v
                        for s, v in illumination[
                            "excess_flicker_lower_bound_fraction"
                        ].items()
                        if v > noise.max_flicker_percent / 100
                    }
                    if fast_quality:
                        unstable = {
                            s: v for s, v in illumination["noise_variance_bias_upper_fraction"].items()
                            if s in unstable and v > illumination["noise_variance_bias_budget_fraction"]
                        }
                    if unstable:
                        detail = ", ".join(
                            f"{s}: {v * 100:.3f}%" for s, v in unstable.items()
                        )
                        if fast_quality:
                            raise CaptureError(
                                f"조명 변동으로 인한 노이즈 분산 영향의 보수적 상한 {detail}가 0.1%를 넘었습니다. 차광과 완료한 밝은 자료는 유지됩니다."
                            )
                        raise CaptureError(
                            f"자동 제외 후에도 조명 변동 {detail}가 허용치 {noise.max_flicker_percent:g}%를 넘었습니다. 광원을 안정시키고 ②를 다시 누르세요. 차광 자료는 유지됩니다."
                        )
                    illumination["complete"] = True
                    write_json(folder / "illumination.json", illumination)
                for serial, stream in streams.items():
                    stream.flush()
                    os.fsync(stream.fileno())
                    stream.close()
                    raw = folder / f"{serial}.raw"
                    if sha256_file(raw) != digests[serial].hexdigest():
                        raise CaptureError(f"{serial}: RAW 저장 후 체크섬 불일치")
                    stats = folder / f"{serial}.npz"
                    metadata = folder / f"{serial}.json"
                    save_npz(stats, accumulators[serial].arrays())
                    write_json(metadata, meta[serial])
                    record = {"serial": serial, "point": point}
                    if illumination is not None:
                        record.update(
                            illumination_audit=str(
                                (folder / "illumination.json").relative_to(root)
                            ),
                            illumination_audit_sha256=sha256_file(
                                folder / "illumination.json"
                            ),
                        )
                    for name, path in (
                        ("raw", raw),
                        ("stats", stats),
                        ("metadata", metadata),
                    ):
                        record[name] = str(path.relative_to(root))
                        record[name + "_sha256"] = sha256_file(path)
                    records.append(record)
                fsync_dir(folder)
                if noise.protocol != "two_condition":
                    _save_noise_checkpoint(root, phase, session, records)
            finally:
                for stream in streams.values():
                    stream.close()
                if illumination is not None and not illumination.get("complete"):
                    # Preserve rejection reasons even when cancelled, disconnected,
                    # overbright, or unable to collect a stable stack.
                    illumination["complete"] = False
                    write_json(folder / "illumination_failed.json", illumination)
        write_json(
            root / f"{phase}.json",
            {
                "schema_version": 1,
                "phase": phase,
                "session_identity": _session_identity(session),
                "stacks": records,
                "completed_at_utc": utc_now(),
                **({"light_preflight": preflight} if preflight else {}),
            },
        )
        write_json(
            root / f"{phase.upper()}_COMPLETE.json",
            {"sha256": sha256_file(root / f"{phase}.json")},
        )
        verify_phase(root, phase, cancel)
        session.update(
            status="awaiting_bright" if phase == "dark" else "awaiting_analysis"
        )
        write_json(root / "session.json", session)
        fsync_dir(root)
        return {
            "phase": phase,
            "path": str(root),
            "frames": total * len(backend.cameras),
            "simulated": session["simulated"],
        }
    except Exception as error:
        if root and root.exists() and root.name.endswith(".partial"):
            try:
                write_json(
                    root / f"FAILED_{phase.upper()}_{uuid.uuid4().hex[:8]}.json",
                    {
                        "error": str(error),
                        "completed_groups": len(records),
                        "time_utc": utc_now(),
                    },
                )
            except OSError as storage_error:
                if hasattr(error, "add_note"):
                    error.add_note(f"실패 기록 저장 오류: {storage_error}")
            error.noise_session_path = str(root)
            error.add_note(f"미완성 측정 보존: {root}") if hasattr(
                error, "add_note"
            ) else None
        raise
    finally:
        if hasattr(backend, "noise_phase"):
            backend.noise_phase = None
        primary_error = sys.exc_info()[1]
        try:
            backend.stop()
        except Exception as cleanup_error:
            if primary_error is None:
                raise
            if hasattr(primary_error, "add_note"):
                primary_error.add_note(f"장치 중지 오류: {cleanup_error}")
