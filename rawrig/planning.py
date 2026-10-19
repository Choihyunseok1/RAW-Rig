"""Freeze all references before capturing; preserve requested and applied settings."""
from dataclasses import asdict, replace

from .models import Cancelled, CaptureError, RetryableCaptureError, Preset, bracket_presets, check_cancel
from .recovery import is_transient


def step_presets(step):
    return {serial: Preset(**value) for serial, value in step["applied"].items()}


def make_step(name, source, variant, requested, limits, policy):
    serials = set(requested)
    if set(limit["serial"] for limit in limits) != serials:
        raise CaptureError("설정 범위의 카메라 목록이 일치하지 않습니다.")
    applied = {}
    common = source != "per_camera"
    for limit in limits:
        serial = limit["serial"]
        preset = requested[serial]
        exp_range = (max(item["exposure_us"][0] for item in limits), min(item["exposure_us"][1] for item in limits)) if common else limit["exposure_us"]
        gain_range = (max(item["gain_db"][0] for item in limits), min(item["gain_db"][1] for item in limits)) if common else limit["gain_db"]
        if exp_range[0] > exp_range[1] or gain_range[0] > gain_range[1]:
            raise CaptureError("선택된 카메라의 공통 노출 컨트롤 범위가 없습니다.")
        exposure = min(max(preset.exposure_us, exp_range[0]), exp_range[1])
        gain = min(max(preset.gain_db, gain_range[0]), gain_range[1])
        if policy == "strict" and (exposure != preset.exposure_us or gain != preset.gain_db):
            raise CaptureError(f"{name}: 요청값 {preset.exposure_us:g} µs / {preset.gain_db:g} dB가 지원 범위 밖입니다. 범위 제한 허용 또는 측광 상한을 조정하세요.")
        applied[serial] = asdict(replace(preset, exposure_us=exposure, gain_db=gain))
    original = {serial: asdict(p) for serial, p in requested.items()}
    return {"name": name, "reference_serial": source, "variant": variant,
            "requested": original, "applied": applied, "range_clipped": original != applied}


def collect_references(backend, config, cancel, progress):
    cameras = backend.cameras if config.capture_strategy != "bracket" else [next(
        (c for c in backend.cameras if c.serial == config.reference_serial), backend.cameras[0])]
    if config.capture_strategy == "bracket" and config.reference_serial and cameras[0].serial != config.reference_serial:
        raise CaptureError("자동 측광 기준 카메라가 연결되어 있지 않습니다.")
    references = []
    for camera in cameras:
        check_cancel(cancel)
        progress(f"{camera.serial} 자동 노출 컨트롤 측광 중…")
        attempts = []
        for attempt in range(config.auto_retry_count + 1):
            check_cancel(cancel)
            seconds = min(30., config.meter_seconds * 2**attempt)
            try:
                reference = backend.meter(replace(config, reference_serial=camera.serial, meter_seconds=seconds), cancel)
                if reference.get("reference_serial") != camera.serial:
                    raise CaptureError(f"{camera.serial}: 측광 기준 카메라 식별값이 다릅니다.")
                attempts.append({"attempt": attempt + 1, "meter_seconds": seconds,
                    "stable": bool(reference.get("stable")), "samples": reference.get("samples"),
                    "last_samples": reference.get("last_samples")})
                if not reference.get("stable", False):
                    raise RetryableCaptureError(f"{camera.serial}: 자동 측광이 안정되지 않았습니다.")
                break
            except Cancelled:
                raise
            except Exception as error:
                if not is_transient(error):
                    raise
                if not attempts or attempts[-1]["attempt"] != attempt + 1:
                    attempts.append({"attempt": attempt + 1, "meter_seconds": seconds, "stable": False, "error": str(error)})
                if attempt == config.auto_retry_count:
                    raise CaptureError(f"{error}\n측광 시간을 자동 연장하고 {attempt}회 재시도했습니다. 장면·조명을 고정하고 다시 촬영하세요.") from error
                progress(f"{camera.serial}: 자동 측광 대기 연장 · {min(30., seconds * 2):g}초 · 재시도 {attempt + 1}/{config.auto_retry_count}")
        reference["meter_attempts"] = attempts
        Preset("reference", reference["exposure_us"], reference["gain_db"]).validate()
        references.append(reference)
    return references


def resolve_plan(backend, config, cancel, progress):
    from .engine import resolve_auto_presets
    serials = [c.serial for c in backend.cameras]
    limits = backend.limits()
    references = collect_references(backend, config, cancel, progress) if config.auto_reference or config.capture_strategy != "bracket" else []
    plan = []
    if config.capture_strategy == "bracket":
        presets = resolve_auto_presets(config, references[0], limits) if references else config.presets
        for index, preset in enumerate(presets):
            plan.append(make_step(preset.name, references[0]["reference_serial"] if references else "manual",
                                  index + 1, {s: preset for s in serials}, limits, "strict"))
    elif config.capture_strategy == "cross_reference":
        for reference in references:
            source = reference["reference_serial"]
            exposure, gain = reference["exposure_us"], reference["gain_db"]
            presets = bracket_presets(exposure * config.low_exposure_factor, exposure * config.high_exposure_factor,
                                      gain + config.low_gain_offset_db, gain + config.high_gain_offset_db) if config.cross_bracket else [Preset("auto", exposure, gain)]
            for index, preset in enumerate(presets):
                name = f"ref_{source}_setting_{index+1:02d}"
                preset = replace(preset, name=name)
                plan.append(make_step(name, source, index + 1, {s: preset for s in serials}, limits, config.range_policy))
    else:
        requested = {r["reference_serial"]: Preset("own_auto", r["exposure_us"], r["gain_db"]) for r in references}
        plan.append(make_step("own_auto", "per_camera", 1, requested, limits, config.range_policy))
    return plan, {"mode": config.capture_strategy, "references": references, "capture_plan": plan,
                  **(references[0] if config.capture_strategy == "bracket" and references else {})}
