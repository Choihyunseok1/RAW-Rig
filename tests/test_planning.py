import json
from threading import Event

import pytest

from rawrig.demo import DemoBackend
from rawrig.engine import run_capture
from rawrig.models import CaptureConfig, CaptureError, Cancelled
from rawrig.planning import resolve_plan
from rawrig.storage import verify_dataset, sha256_file


@pytest.fixture
def rig():
    backend = DemoBackend()
    backend.open([c.serial for c in backend.discover()])
    yield backend
    backend.close()


@pytest.mark.parametrize("mode,extended,groups,frames", [("cross_reference", False, 3, 9), ("cross_reference", True, 12, 36), ("per_camera", False, 1, 3)])
def test_all_reference_modes_end_to_end(tmp_path, rig, mode, extended, groups, frames):
    result = run_capture(rig, CaptureConfig(output_dir=str(tmp_path), capture_strategy=mode, cross_bracket=extended,
        meter_seconds=.5, warmup_frames=0, settle_ms=0))
    report = verify_dataset(result)
    assert report["frames"] == frames and report["groups"] == groups
    manifest = json.loads((result / "manifest.json").read_text())
    assert [r["reference_serial"] for r in manifest["auto_reference"]["references"]] == ["DEMO_1", "DEMO_2", "DEMO_3"]
    assert len(manifest["capture_plan"]) == groups
    if mode == "per_camera":
        applied = manifest["capture_plan"][0]["applied"]
        assert [applied[s]["exposure_us"] for s in applied] == [12000, 24000, 36000]
        assert [applied[s]["gain_db"] for s in applied] == [3, 6, 9]
    else:
        for step in manifest["capture_plan"]:
            assert len({(v["exposure_us"], v["gain_db"]) for v in step["applied"].values()}) == 1
        if not extended:
            assert [next(iter(s["applied"].values()))["exposure_us"] for s in manifest["capture_plan"]] == [12000, 24000, 36000]


def test_all_references_frozen_before_any_dataset_trigger(tmp_path, rig):
    events = []
    meter, capture = rig.meter, rig.capture
    def measured(config, event):
        events.append("meter:" + config.reference_serial)
        return meter(config, event)
    def captured(*args):
        events.append("trigger")
        return capture(*args)
    rig.meter, rig.capture = measured, captured
    run_capture(rig, CaptureConfig(output_dir=str(tmp_path), capture_strategy="cross_reference", warmup_frames=0, settle_ms=0))
    assert events[:3] == ["meter:DEMO_1", "meter:DEMO_2", "meter:DEMO_3"]


def test_cross_range_strict_fails_before_files_or_triggers(tmp_path, rig):
    limits = rig.limits()
    limits[-1]["exposure_us"][1] = 20000
    rig.limits = lambda: limits
    with pytest.raises(CaptureError, match="지원 범위 밖"):
        run_capture(rig, CaptureConfig(output_dir=str(tmp_path), capture_strategy="cross_reference"))
    assert rig.counter == 0 and not list(tmp_path.iterdir())


def test_clipping_keeps_cross_conditions_identical_and_records_original(rig):
    limits = rig.limits()
    limits[-1]["exposure_us"][1] = 20000
    rig.limits = lambda: limits
    plan, _ = resolve_plan(rig, CaptureConfig(capture_strategy="cross_reference", range_policy="clip"), Event(), lambda _: None)
    assert plan[-1]["range_clipped"]
    assert {p["exposure_us"] for p in plan[-1]["applied"].values()} == {20000}
    assert {p["exposure_us"] for p in plan[-1]["requested"].values()} == {36000}


def test_unstable_reference_aborts_entire_cross_session(tmp_path, rig):
    meter = rig.meter
    def unstable(config, event):
        result = meter(config, event)
        result["stable"] = config.reference_serial != "DEMO_2"
        return result
    rig.meter = unstable
    with pytest.raises(CaptureError, match="DEMO_2"):
        run_capture(rig, CaptureConfig(output_dir=str(tmp_path), capture_strategy="cross_reference"))
    assert rig.counter == 0 and not list(tmp_path.iterdir())


def test_unstable_meter_automatically_extends_time_and_records_attempts(rig):
    calls = []
    meter = rig.meter
    def settling(config, event):
        calls.append(config.meter_seconds)
        result = meter(config, event)
        result["stable"] = config.meter_seconds >= 8
        return result
    rig.meter = settling
    config = CaptureConfig(meter_seconds=2)
    _, reference = resolve_plan(rig, config, Event(), lambda _: None)
    assert calls == [2, 4, 8] and config.meter_seconds == 2
    assert [item["stable"] for item in reference["meter_attempts"]] == [False, False, True]


def test_meter_identity_mismatch_is_never_retried(rig):
    calls = []
    def wrong(config, event):
        calls.append(1)
        return {"reference_serial": "OTHER", "stable": True}
    rig.meter = wrong
    with pytest.raises(CaptureError, match="식별값"):
        resolve_plan(rig, CaptureConfig(), Event(), lambda _: None)
    assert len(calls) == 1


def test_cancel_during_all_reference_metering_has_no_dataset(tmp_path, rig):
    cancel = Event()
    def progress(message, *_):
        if "DEMO_2" in message:
            cancel.set()
    with pytest.raises(Cancelled):
        run_capture(rig, CaptureConfig(output_dir=str(tmp_path), capture_strategy="cross_reference"), cancel, progress)
    assert not list(tmp_path.iterdir())


def test_verifier_detects_plan_tamper_even_after_manifest_rehash(tmp_path, rig):
    result = run_capture(rig, CaptureConfig(output_dir=str(tmp_path), capture_strategy="per_camera", settle_ms=0, warmup_frames=0))
    path = result / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["capture_plan"][0]["applied"]["DEMO_2"]["gain_db"] += 1
    path.write_text(json.dumps(manifest))
    (result / "COMPLETE.json").write_text(json.dumps({"manifest_sha256": sha256_file(path)}))
    with pytest.raises(CaptureError, match="촬영 계획"):
        verify_dataset(result)
