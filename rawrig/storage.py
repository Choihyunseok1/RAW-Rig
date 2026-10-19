from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import uuid
from types import SimpleNamespace

import numpy as np

from .models import CameraInfo, CaptureConfig, CaptureError, Frame


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fsync_dir(path: Path):
    if os.name != "nt":
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def write_bytes(path: Path, data: bytes):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def write_json(path: Path, value):
    write_bytes(path, (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8"))


class DatasetWriter:
    """A complete session is published only after all frames pass read-back checks."""

    def __init__(self, config: CaptureConfig, cameras: list[CameraInfo], settings: list[dict], plan=None):
        config.validate()
        if len(cameras) != config.expected_cameras or len({c.serial for c in cameras}) != len(cameras):
            raise CaptureError("선택된 카메라 수 또는 시리얼이 올바르지 않습니다.")
        root = Path(config.output_dir).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        self.plan = plan
        self.steps = len(plan) if plan is not None else 4
        estimated = (sum(s["width"] * s["height"] * 2 for s in settings) + (64 * 1024**2 if config.dslr_enabled else 0)) * self.steps * config.repeats
        if shutil.disk_usage(root).free < estimated * 1.2 + 64 * 1024**2:
            raise CaptureError("저장 공간이 부족합니다.")
        name = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:10]
        self.final = root / name
        self.staging = root / ("." + name + ".partial")
        self.staging.mkdir()
        fsync_dir(root)
        self.config = config
        self.cameras = cameras
        self.keys = set()
        self.manifest = {
            "schema_version": 1, "status": "in_progress", "created_at_utc": utc_now(),
            "config": config.to_dict(), "cameras": [asdict(c) for c in cameras],
            "simulated": any(c.simulated for c in cameras), "frames": [], "groups": [],
            "capture_plan": plan, "companions": [],
            "raw_encoding": "row-major Bayer samples, uint8 or little-endian uint16; camera values unchanged",
            "white_balance": "preview_only; camera red/blue ratios unity when available",
            "synchronization": {"mode": config.sync_mode, "exposure_skew_measured": False,
                "camera_clocks_comparable": False,
                "note": "Software dispatch time is not exposure skew. Hardware GPIO propagation is not calibrated."},
        }
        write_json(self.staging / "manifest.json", self.manifest)
        fsync_dir(self.staging)

    def save_group(self, frames: list[Frame], step: int, repeat: int, trigger: dict):
        expected = {c.serial for c in self.cameras}
        if len(frames) != len(expected) or {f.camera.serial for f in frames} != expected:
            raise CaptureError("촬영 그룹에 카메라 누락/중복이 있습니다.")
        records = []
        for frame in frames:
            key = (step, repeat, frame.camera.serial)
            if key in self.keys:
                raise CaptureError("같은 촬영 그룹이 두 번 저장되었습니다.")
            folder = self.staging / f"setting_{step + 1:02d}" / f"shot_{repeat + 1:04d}"
            folder.mkdir(parents=True, exist_ok=True)
            stem = f"camera_{frame.camera.serial}"
            raw_path, meta_path = folder / (stem + ".raw"), folder / (stem + ".json")
            dtype = "<u2" if frame.pixels.dtype.itemsize == 2 else "u1"
            payload = np.ascontiguousarray(frame.pixels, dtype=dtype).tobytes()
            expected_hash = hashlib.sha256(payload).hexdigest()
            write_bytes(raw_path, payload)
            if raw_path.stat().st_size != len(payload) or sha256_file(raw_path) != expected_hash:
                raise CaptureError(f"{stem}: 저장 후 RAW 재검증에 실패했습니다.")
            pixels = frame.pixels
            metadata = {
                "schema_version": 1, "camera": asdict(frame.camera), "step": step, "repeat": repeat,
                "preset": self.plan[step]["applied"][frame.camera.serial] if self.plan is not None else asdict(self.config.presets[step]), "pixel_format": frame.pixel_format,
                "capture_step": self.plan[step] if self.plan is not None else None,
                "bayer_pattern": frame.pixel_format[5:7], "shape": list(pixels.shape), "dtype": dtype,
                "byte_order": "little" if dtype == "<u2" else "not_applicable", "bytes": len(payload),
                "sha256": expected_hash, "frame_id": frame.frame_id,
                "camera_timestamp_ns": frame.camera_timestamp_ns,
                "host_received_monotonic_ns": frame.host_received_ns,
                "applied_settings": frame.settings, "chunk": frame.chunk, "transport": frame.transport,
                "preview_white_balance": {"red": self.config.preview_red, "blue": self.config.preview_blue},
                "statistics": {"min": int(pixels.min()), "max": int(pixels.max()), "mean": float(pixels.mean()),
                    "zero_fraction": float(np.count_nonzero(pixels == 0) / pixels.size),
                    "full_scale_fraction": float(np.count_nonzero(pixels >= frame.settings.get("sensor_white_level", frame.settings.get("pixel_max", np.iinfo(pixels.dtype).max))) / pixels.size)},
                "integrity": {"readback_sha256": "passed", "shape_dtype_bytes": "passed"},
            }
            write_json(meta_path, metadata)
            record = {"raw": str(raw_path.relative_to(self.staging)), "metadata": str(meta_path.relative_to(self.staging)),
                "raw_sha256": expected_hash, "metadata_sha256": sha256_file(meta_path),
                "bytes": len(payload), "serial": frame.camera.serial, "step": step, "repeat": repeat}
            records.append(record)
            self.keys.add(key)
            fsync_dir(folder)
            fsync_dir(folder.parent)
        self.manifest["frames"].extend(records)
        self.manifest["groups"].append({"step": step, "repeat": repeat, "trigger": trigger})
        write_json(self.staging / "manifest.json", self.manifest)
        fsync_dir(self.staging)

    def save_companion(self, shot, step, repeat):
        folder = self.staging / f"setting_{step + 1:02d}" / f"shot_{repeat + 1:04d}"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / "nikon_z5.nef"
        meta_path = folder / "nikon_z5.json"
        payload, metadata = shot
        digest = hashlib.sha256(payload).hexdigest()
        write_bytes(path, payload)
        if sha256_file(path) != digest:
            raise CaptureError("Z5 NEF 저장 후 체크섬이 일치하지 않습니다.")
        metadata = {**metadata, "step": step, "repeat": repeat, "sha256": digest, "bytes": len(payload)}
        write_json(meta_path, metadata)
        self.manifest["companions"].append({"step": step, "repeat": repeat, "raw": str(path.relative_to(self.staging)),
            "metadata": str(meta_path.relative_to(self.staging)), "raw_sha256": digest,
            "metadata_sha256": sha256_file(meta_path), "bytes": len(payload)})
        write_json(self.staging / "manifest.json", self.manifest)
        fsync_dir(folder)
        fsync_dir(folder.parent)
        fsync_dir(self.staging)

    def complete(self):
        required = {(s, r, c.serial) for s in range(self.steps) for r in range(self.config.repeats) for c in self.cameras}
        if self.keys != required:
            raise CaptureError("전체 설정의 카메라/반복 촬영이 완성되지 않았습니다.")
        self.manifest.update(status="complete", completed_at_utc=utc_now())
        write_json(self.staging / "manifest.json", self.manifest)
        verify_dataset(self.staging, require_complete=False)
        write_json(self.staging / "COMPLETE.json", {"manifest_sha256": sha256_file(self.staging / "manifest.json")})
        fsync_dir(self.staging)
        os.rename(self.staging, self.final)
        fsync_dir(self.final.parent)
        return self.final

    def fail(self, error: Exception):
        if self.staging.exists():
            self.manifest.update(status="failed", error=str(error), failed_at_utc=utc_now())
            write_json(self.staging / "manifest.json", self.manifest)
            write_json(self.staging / "FAILED.json", {"error": str(error)})
            fsync_dir(self.staging)


def _safe_child(root: Path, relative: str):
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise CaptureError("잘못된 데이터 경로입니다.")
    child = (root / relative).resolve()
    if not child.is_relative_to(root):
        raise CaptureError("데이터 폴더 밖을 가리키는 경로입니다.")
    return child


def verify_dataset(path: Path | str, require_complete=True):
    root = Path(path).resolve()
    try:
        manifest_path = root / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["schema_version"] != 1 or manifest["status"] != "complete":
            raise CaptureError("완성된 데이터셋이 아닙니다.")
        if require_complete:
            marker = json.loads((root / "COMPLETE.json").read_text(encoding="utf-8"))
            if marker["manifest_sha256"] != sha256_file(manifest_path):
                raise CaptureError("Manifest 체크섬이 일치하지 않습니다.")
        config = CaptureConfig.from_dict(manifest["config"])
        serials = {c["serial"] for c in manifest["cameras"]}
        if len(serials) != config.expected_cameras or len(serials) != len(manifest["cameras"]):
            raise CaptureError("카메라 목록이 올바르지 않습니다.")
        plan = manifest.get("capture_plan")
        steps = len(plan) if plan is not None else 4
        if plan is not None:
            if steps != config.planned_groups:
                raise CaptureError("촬영 계획의 설정 수가 모드와 일치하지 않습니다.")
            from .models import Preset
            for step in plan:
                if set(step["applied"]) != serials or set(step["requested"]) != serials:
                    raise CaptureError("촬영 계획의 카메라 목록이 일치하지 않습니다.")
                for preset in step["applied"].values():
                    Preset(**preset).validate()
                if config.capture_strategy == "cross_reference" and step["reference_serial"] not in serials:
                    raise CaptureError("교차 촬영의 기준 카메라가 올바르지 않습니다.")
                if config.capture_strategy != "per_camera" and len({(p["exposure_us"], p["gain_db"]) for p in step["applied"].values()}) != 1:
                    raise CaptureError("공통 조건 촬영의 카메라별 설정이 다릅니다.")
            if config.capture_strategy == "cross_reference":
                variants = range(1, 5) if config.cross_bracket else [1]
                expected_refs = {(s, v) for s in serials for v in variants}
                if {(s["reference_serial"], s["variant"]) for s in plan} != expected_refs:
                    raise CaptureError("교차 촬영 기준 또는 설정이 누락·중복되었습니다.")
        required = {(s, r, c) for s in range(steps) for r in range(config.repeats) for c in serials}
        actual, paths, group_frames = set(), set(), {}
        for record in manifest["frames"]:
            key = (record["step"], record["repeat"], record["serial"])
            if key in actual or key not in required:
                raise CaptureError("누락/중복/잘못된 촬영 그룹입니다.")
            actual.add(key)
            raw, meta_path = _safe_child(root, record["raw"]), _safe_child(root, record["metadata"])
            for child in (raw, meta_path):
                if child in paths:
                    raise CaptureError("데이터 파일 경로가 중복됩니다.")
                paths.add(child)
            if sha256_file(raw) != record["raw_sha256"] or sha256_file(meta_path) != record["metadata_sha256"]:
                raise CaptureError(f"파일 체크섬 불일치: {raw.name}")
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if (meta["step"], meta["repeat"], meta["camera"]["serial"]) != key:
                raise CaptureError("메타데이터 촬영 그룹이 일치하지 않습니다.")
            dtype = meta["dtype"]
            shape = meta["shape"]
            if dtype not in ("u1", "<u2") or len(shape) != 2 or any(type(n) is not int or n <= 0 for n in shape):
                raise CaptureError("RAW shape/dtype이 올바르지 않습니다.")
            expected_size = shape[0] * shape[1] * np.dtype(dtype).itemsize
            if not (raw.stat().st_size == record["bytes"] == meta["bytes"] == expected_size):
                raise CaptureError("RAW 바이트 수가 일치하지 않습니다.")
            if meta["sha256"] != record["raw_sha256"]:
                raise CaptureError("RAW/메타데이터 체크섬이 다릅니다.")
            from .models import Frame, CameraInfo, validate_frame
            pixels = np.memmap(raw, dtype=dtype, mode="r", shape=tuple(shape))
            frame = Frame(CameraInfo(**meta["camera"]), pixels, meta["pixel_format"], meta["frame_id"],
                meta["camera_timestamp_ns"], meta["host_received_monotonic_ns"], meta["applied_settings"], meta["chunk"], meta["transport"])
            from .models import Preset
            expected_preset = plan[record["step"]]["applied"][record["serial"]] if plan is not None else asdict(config.presets[record["step"]])
            if meta["preset"] != expected_preset:
                raise CaptureError("프레임 설정이 촬영 계획과 일치하지 않습니다.")
            validate_frame(frame, Preset(**expected_preset))
            group_frames.setdefault(key[:2], []).append(SimpleNamespace(camera_timestamp_ns=frame.camera_timestamp_ns, settings=frame.settings))
            del frame, pixels
        if actual != required:
            raise CaptureError("프레임이 누락되었습니다.")
        groups = {(g["step"], g["repeat"]) for g in manifest["groups"]}
        required_groups = {(s, r) for s in range(steps) for r in range(config.repeats)}
        if groups != required_groups or len(groups) != len(manifest["groups"]):
            raise CaptureError("트리거 그룹이 누락되거나 중복되었습니다.")
        companions = manifest.get("companions", [])
        if config.dslr_enabled:
            if {(r["step"], r["repeat"]) for r in companions} != required_groups or len(companions) != len(required_groups):
                raise CaptureError("Z5 NEF 촬영이 누락되거나 중복되었습니다.")
            from .dslr import validate_nef
            for record in companions:
                raw, meta_path = _safe_child(root, record["raw"]), _safe_child(root, record["metadata"])
                if raw in paths or meta_path in paths:
                    raise CaptureError("Z5 파일 경로가 중복됩니다.")
                paths.update((raw, meta_path))
                if sha256_file(raw) != record["raw_sha256"] or sha256_file(meta_path) != record["metadata_sha256"]:
                    raise CaptureError("Z5 파일 체크섬 불일치.")
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                if (meta["step"], meta["repeat"]) != (record["step"], record["repeat"]) or raw.stat().st_size != record["bytes"] or meta["bytes"] != record["bytes"] or meta["sha256"] != record["raw_sha256"]:
                    raise CaptureError("Z5 메타데이터가 일치하지 않습니다.")
                decoded = validate_nef(raw.read_bytes())
                planned = manifest["dslr_plan"][record["step"]]
                if meta["settings"] != planned or abs(decoded["exposure_us"] - planned["exposure_us"]) > max(1., planned["exposure_us"] * .01) or decoded["iso"] != planned["iso"]:
                    raise CaptureError("Z5 NEF 실제 노출시간·ISO가 촬영 계획과 일치하지 않습니다.")
                if decoded["shape"] != meta["raw_validation"]["shape"]:
                    raise CaptureError("Z5 RAW 해상도 재검사 실패.")
        elif companions:
            raise CaptureError("촬영 설정에 없는 Z5 파일 기록입니다.")
        if config.sync_mode == "ptp" and not manifest["simulated"]:
            from .sync import validate_ptp_snapshot, validate_ptp_frames
            for group in manifest["groups"]:
                trigger = group["trigger"]
                if trigger["mode"] != "ptp" or len(trigger["acknowledgements"]) != len(serials) or any(a["status"] for a in trigger["acknowledgements"]):
                    raise CaptureError("PTP 예약 촬영 응답 기록이 올바르지 않습니다.")
                for snapshot in (trigger["ptp_before"], trigger["ptp_after"]):
                    if {s["serial"] for s in snapshot} != serials or len(snapshot) != len(serials):
                        raise CaptureError("PTP 시계 기록의 카메라 목록이 일치하지 않습니다.")
                    validate_ptp_snapshot(snapshot, config.ptp_max_offset_us)
                spread = validate_ptp_frames(group_frames[(group["step"], group["repeat"])], trigger["scheduled_action_time_ns"], config.ptp_max_timestamp_spread_us)
                if spread != trigger["image_timestamp_spread_ns"]:
                    raise CaptureError("PTP 프레임 시차 기록이 일치하지 않습니다.")
        return {"status": "passed", "frames": len(actual), "cameras": len(serials), "nef_files": len(companions), "groups": steps * config.repeats, "simulated": manifest["simulated"], "path": str(root)}
    except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
        raise CaptureError(f"데이터셋 검사 실패: {error}") from error
