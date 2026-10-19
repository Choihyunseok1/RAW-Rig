import json
from threading import Event

import numpy as np
import pytest

from rawrig.demo import DemoBackend
from rawrig.engine import run_capture, resolve_auto_presets
from rawrig.models import CaptureConfig, CaptureError, Cancelled, Preset, validate_frame
from rawrig.storage import verify_dataset, sha256_file
from rawrig.sync import validate_ptp_snapshot, validate_ptp_frames


@pytest.fixture
def rig():
    backend = DemoBackend()
    backend.open([c.serial for c in backend.discover()])
    yield backend
    backend.close()


def test_three_cameras_four_presets_repeats_lossless(tmp_path, rig):
    captured = []
    result = run_capture(rig, CaptureConfig(output_dir=str(tmp_path), repeats=2, settle_ms=0),
                         preview=lambda frames: captured.extend(frames))
    report = verify_dataset(result)
    assert report["frames"] == 24 and report["simulated"]
    manifest = json.loads((result / "manifest.json").read_text())
    assert manifest["auto_reference"]["baseline_setting_1"] == {"exposure_us": 12000., "gain_db": 3.}
    for record, original in zip(manifest["frames"], captured):
        meta = json.loads((result / record["metadata"]).read_text())
        restored = np.fromfile(result / record["raw"], dtype=meta["dtype"]).reshape(meta["shape"])
        np.testing.assert_array_equal(restored, original.pixels)
        assert meta["applied_settings"]["exposure_us"] == manifest["config"]["presets"][record["step"]]["exposure_us"]
    assert not list(tmp_path.glob("*.partial"))


def test_tampered_raw_detected(tmp_path, rig):
    result = run_capture(rig, CaptureConfig(output_dir=str(tmp_path), settle_ms=0))
    raw = next(result.rglob("*.raw"))
    with raw.open("r+b") as stream:
        stream.write(b"\xff\xff")
    with pytest.raises(CaptureError, match="체크섬"):
        verify_dataset(result)


def test_tampered_metadata_or_manifest_detected(tmp_path, rig):
    result = run_capture(rig, CaptureConfig(output_dir=str(tmp_path), settle_ms=0))
    meta = next(result.rglob("camera_*.json"))
    meta.write_text("{}")
    with pytest.raises(CaptureError, match="체크섬"):
        verify_dataset(result)


def test_cancelled_session_never_published(tmp_path, rig):
    cancel = Event()
    def progress(message, done, total):
        if done == 1:
            cancel.set()
    with pytest.raises(Cancelled):
        run_capture(rig, CaptureConfig(output_dir=str(tmp_path), settle_ms=0), cancel, progress)
    assert not list(tmp_path.rglob("COMPLETE.json"))
    partials = list(tmp_path.glob(".*.partial"))
    assert len(partials) == 1
    assert (partials[0] / "FAILED.json").is_file()
    with pytest.raises(CaptureError, match="완성"):
        verify_dataset(partials[0])


def test_incomplete_group_never_published(tmp_path, rig):
    capture = rig.capture
    def missing_camera(timeout, cancel):
        frames, trigger = capture(timeout, cancel)
        return frames[:-1], trigger
    rig.capture = missing_camera
    with pytest.raises(CaptureError, match="누락"):
        run_capture(rig, CaptureConfig(output_dir=str(tmp_path), settle_ms=0))
    assert not list(tmp_path.rglob("COMPLETE.json"))


def test_stale_chunk_rejected(tmp_path, rig):
    capture = rig.capture
    def wrong_exposure(timeout, cancel):
        frames, trigger = capture(timeout, cancel)
        frames[0].chunk["exposure_us"] = 999999.
        return frames, trigger
    rig.capture = wrong_exposure
    with pytest.raises(CaptureError, match="chunk"):
        run_capture(rig, CaptureConfig(output_dir=str(tmp_path), settle_ms=0))
    assert not list(tmp_path.rglob("COMPLETE.json"))


def test_wrong_camera_count_rejected_before_capture(tmp_path, rig):
    rig.cameras.pop()
    with pytest.raises(CaptureError, match="3대"):
        run_capture(rig, CaptureConfig(output_dir=str(tmp_path)))
    assert not list(tmp_path.iterdir())


def test_auto_presets_common_ranges_and_db_offsets(rig):
    config = CaptureConfig()
    reference = {"exposure_us": 8000., "gain_db": 12.}
    presets = resolve_auto_presets(config, reference, rig.limits())
    assert [(p.exposure_us, p.gain_db) for p in presets] == [(8000., 12.), (32000., 12.), (8000., 24.), (32000., 24.)]
    assert not reference["range_clipped"]
    reference = {"exposure_us": 2., "gain_db": 0.}
    limits = [{"exposure_us": [24., 30000000.], "gain_db": [0., 48.]}]
    with pytest.raises(CaptureError, match="같아졌"):
        resolve_auto_presets(config, reference, limits)


def test_reject_rgb_and_duplicate_ids(rig):
    preset = Preset("test", 1000, 0)
    rig.prepare(preset, CaptureConfig(sync_mode="software"))
    rig.start()
    frames, _ = rig.capture(1000, Event())
    frame = frames[0]
    with pytest.raises(CaptureError, match="중복"):
        validate_frame(frame, preset, frame.frame_id)
    frame.pixel_format = "RGB16"
    with pytest.raises(CaptureError, match="Bayer RAW"):
        validate_frame(frame, preset)


def ptp(status="Slave", parent=10, clock=20, offset=100):
    return {"serial": str(clock), "status": status, "parent_clock_id": parent, "clock_id": clock, "offset_ns": offset}


def test_ptp_one_common_domain_bounded_offset():
    master = ptp("Master", clock=10)
    assert validate_ptp_snapshot([master, ptp(), ptp(clock=30)], 10)
    assert validate_ptp_snapshot([ptp(), ptp(clock=30)], 10)  # external grandmaster
    for bad in ([master, ptp("Master", clock=30)], [master, ptp(parent=99)],
                [master, ptp(offset=10001)], [master, ptp("Listening")]):
        with pytest.raises(CaptureError):
            validate_ptp_snapshot(bad, 10)


def test_ptp_frame_spread_and_stale_target(rig):
    rig.prepare(Preset("test", 1000, 0), CaptureConfig(sync_mode="ptp"))
    rig.start()
    frames, _ = rig.capture(1000, Event())
    target = frames[0].camera_timestamp_ns
    assert validate_ptp_frames(frames, target, 100) == 2000
    frames[2].camera_timestamp_ns += 1_000_000
    with pytest.raises(CaptureError, match="시차"):
        validate_ptp_frames(frames, target, 100)
    for frame in frames:
        frame.camera_timestamp_ns = target - 100_000_000
    with pytest.raises(CaptureError, match="예약 시각"):
        validate_ptp_frames(frames, target, 100)


def test_verifier_rejects_missing_frame_even_if_manifest_rehashed(tmp_path, rig):
    result = run_capture(rig, CaptureConfig(output_dir=str(tmp_path), settle_ms=0))
    path = result / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["frames"].pop()
    path.write_text(json.dumps(manifest))
    (result / "COMPLETE.json").write_text(json.dumps({"manifest_sha256": sha256_file(path)}))
    with pytest.raises(CaptureError, match="누락"):
        verify_dataset(result)


def test_verifier_rejects_path_escape(tmp_path, rig):
    result = run_capture(rig, CaptureConfig(output_dir=str(tmp_path), settle_ms=0))
    path = result / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["frames"][0]["raw"] = "../escape.raw"
    path.write_text(json.dumps(manifest))
    (result / "COMPLETE.json").write_text(json.dumps({"manifest_sha256": sha256_file(path)}))
    with pytest.raises(CaptureError, match="폴더 밖"):
        verify_dataset(result)
