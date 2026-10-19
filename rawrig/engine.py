from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from threading import Event
import time
import sys
from typing import Callable

from .models import CameraBackend, CaptureConfig, CaptureError, check_cancel, bracket_presets
from .recovery import capture_checked
from .storage import DatasetWriter


def resolve_auto_presets(config: CaptureConfig, reference: dict, limits: list[dict]):
    exp_min = max(limit["exposure_us"][0] for limit in limits)
    exp_max = min(limit["exposure_us"][1] for limit in limits)
    gain_min = max(limit["gain_db"][0] for limit in limits)
    gain_max = min(limit["gain_db"][1] for limit in limits)
    if exp_min >= exp_max or gain_min >= gain_max:
        raise CaptureError("선택된 카메라의 공통 노출시간/게인 범위가 없습니다.")
    requested = [reference["exposure_us"] * config.low_exposure_factor,
                 reference["exposure_us"] * config.high_exposure_factor,
                 reference["gain_db"] + config.low_gain_offset_db,
                 reference["gain_db"] + config.high_gain_offset_db]
    resolved = [min(max(requested[0], exp_min), exp_max), min(max(requested[1], exp_min), exp_max),
                min(max(requested[2], gain_min), gain_max), min(max(requested[3], gain_min), gain_max)]
    if resolved[0] >= resolved[1] or resolved[2] >= resolved[3]:
        raise CaptureError("카메라 범위 제한 때문에 낮음/높음 설정이 같아졌습니다. 배수/오프셋을 조정하세요.")
    reference["relative_requested"] = requested
    reference["common_range_resolved"] = resolved
    reference["range_clipped"] = requested != resolved
    reference["baseline_setting_1"] = {"exposure_us": resolved[0], "gain_db": resolved[2]}
    return [replace(preset, name=original.name) for preset, original in zip(bracket_presets(*resolved), config.presets)]


def run_capture(backend: CameraBackend, config: CaptureConfig, cancel: Event | None = None,
                progress: Callable | None = None, preview: Callable | None = None,
                resolved: Callable | None = None, companion=None) -> Path:
    from concurrent.futures import ThreadPoolExecutor
    from .planning import resolve_plan, step_presets
    config.validate()
    cancel = cancel if cancel is not None else Event()
    progress = progress or (lambda *args: None)
    writer = None
    executor = None
    try:
        if len(backend.cameras) != config.expected_cameras:
            raise CaptureError(f"촬영에는 {config.expected_cameras}대가 필요합니다. 현재 {len(backend.cameras)}대 연결됨.")
        check_cancel(cancel)
        backend.stop()
        profiles = backend.configure_profiles(config) if hasattr(backend, "configure_profiles") else []
        total = config.planned_groups * config.repeats
        plan, reference = resolve_plan(backend, config, cancel, lambda msg: progress(msg, 0, total))
        if config.capture_strategy == "bracket":
            config = replace(config, presets=[next(iter(step_presets(step).values())) for step in plan])
        if resolved:
            resolved(config.presets, reference)
        synchronization = backend.synchronize(config, cancel, lambda msg: progress(msg, 0, total))
        initial = None
        for step in plan:
            check_cancel(cancel)
            settings = backend.prepare(step_presets(step), config)
            if initial is None:
                initial = settings
        dslr_plan = []
        if config.dslr_enabled:
            if companion is None:
                from .dslr import NikonCompanion
                companion = NikonCompanion(config, cancel)
            dslr_plan = companion.preflight(plan, config)
            executor = ThreadPoolExecutor(max_workers=1)
        writer = DatasetWriter(config, backend.cameras, initial, plan)
        writer.manifest["auto_reference"] = reference
        writer.manifest["camera_profiles"] = profiles
        writer.manifest["camera_limits"] = backend.limits()
        from .specs import SPECS
        writer.manifest["published_specifications"] = SPECS
        writer.manifest["dslr_plan"] = dslr_plan
        writer.manifest["synchronization"].update(synchronization)
        writer.manifest["synchronization"]["camera_clocks_comparable"] = config.sync_mode == "ptp"
        done = 0
        for index, step in enumerate(plan):
            check_cancel(cancel)
            presets = step_presets(step)
            progress(f"설정 {index+1}/{len(plan)} · {step['name']}" + (" · 지원 범위 제한됨" if step['range_clipped'] else ""), done, total)
            backend.prepare(presets, config)
            if config.dslr_enabled:
                companion.prepare(dslr_plan[index], config)
            backend.start()
            if cancel.wait(config.settle_ms / 1000):
                check_cancel(cancel)
            previous_ids = {}
            timeout = max(config.timeout_ms, int(max(p.exposure_us for p in presets.values()) / 1000) + 2000)
            def recovery_progress(message):
                progress(message, done, total)
            for warmup_index in range(config.warmup_frames):
                _, warmup_trigger = capture_checked(backend, presets, config, cancel, previous_ids, recovery_progress)
                if warmup_trigger.get("recovery"):
                    writer.manifest.setdefault("warmup_recoveries", []).append({"step": index,
                        "warmup_index": warmup_index, "recovery": warmup_trigger["recovery"]})
            for repeat in range(config.repeats):
                check_cancel(cancel)
                def acquire():
                    future = executor.submit(companion.capture, dslr_plan[index], config) if executor else None
                    industrial_error = None
                    try:
                        if future and config.dslr_dispatch_offset_ms < 0:
                            if cancel.wait(-config.dslr_dispatch_offset_ms / 1000):
                                check_cancel(cancel)
                        industrial_call = time.monotonic_ns()
                        frames, trigger = backend.capture(timeout, cancel)
                    except Exception as error:
                        industrial_error = error
                    # Join USB before retrying the entire group. A USB failure
                    # propagates and never causes a second blind shutter press.
                    shot = future.result() if future else None
                    if industrial_error:
                        raise industrial_error
                    if shot:
                        trigger["dslr"] = dict(shot[1]["timing"])
                        trigger["dslr"]["industrial_capture_call_monotonic_ns"] = industrial_call
                        trigger["dslr"]["command_dispatch_delta_ns"] = trigger["dslr"]["usb_command_started_monotonic_ns"] - industrial_call
                        trigger["dslr"]["industrial_scheduled_lead_ms"] = config.action_lead_ms if config.sync_mode == "ptp" else None
                    return frames, trigger, shot
                frames, trigger, shot = capture_checked(backend, presets, config, cancel, previous_ids,
                    recovery_progress, acquire)
                check_cancel(cancel)
                writer.save_group(frames, index, repeat, trigger)
                if shot:
                    writer.save_companion(shot, index, repeat)
                done += 1
                if preview:
                    preview(frames)
                progress(f"설정 {index+1}/{len(plan)} · {repeat+1}/{config.repeats}: RAW {len(frames)}장" + (" + NEF 1장" if shot else "") + " 저장·검사 완료", done, total)
            backend.stop()
        check_cancel(cancel)
        if companion:
            companion.close()
            writer.manifest["dslr_restore_errors"] = getattr(companion, "restore_errors", [])
            if writer.manifest["dslr_restore_errors"]:
                progress("Z5 촬영은 완료됐지만 일부 원래 설정을 복원하지 못했습니다. 검사 기록을 확인하세요.", total, total)
        progress("전체 데이터셋의 파일 수·체크섬·메타데이터 최종 검사 중…", total, total)
        return writer.complete()
    except Exception as error:
        if writer:
            try:
                if hasattr(error, "recovery"):
                    writer.manifest["recovery_failure"] = error.recovery
                writer.fail(error)
                progress(f"미완성 데이터와 실패 기록 보존: {writer.staging}", 0, config.planned_groups * config.repeats)
            except Exception as storage_error:
                error.add_note(f"실패 기록도 저장하지 못함: {storage_error}") if hasattr(error, "add_note") else None
        raise
    finally:
        primary_error = sys.exc_info()[1]
        cleanup_errors = []
        if executor:
            # No orphaned USB capture can continue after the session reports failure.
            cancel.set()
            executor.shutdown(wait=True, cancel_futures=True)
        if companion and hasattr(companion, "close"):
            try:
                companion.close()
            except Exception as error:
                cleanup_errors.append(error)
        try:
            backend.stop()
        except Exception as error:
            cleanup_errors.append(error)
        if cleanup_errors:
            if primary_error is not None:
                if hasattr(primary_error, "add_note"):
                    for error in cleanup_errors:
                        primary_error.add_note(f"장치 정리 오류: {error}")
            else:
                raise cleanup_errors[0]
