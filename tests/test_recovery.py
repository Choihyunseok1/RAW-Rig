import json
from threading import Event

import numpy as np
import pytest

from rawrig.demo import DemoBackend
from rawrig.engine import run_capture
from rawrig.models import CameraInfo, CaptureConfig, CaptureError, Cancelled, RetryableCaptureError
from rawrig.recovery import is_transient
from rawrig.spinnaker import SDKError, SpinnakerBackend, Device, H
from rawrig.storage import verify_dataset


def rig():
    backend = DemoBackend()
    backend.open([c.serial for c in backend.discover()])
    return backend


def config(path, **extra):
    return CaptureConfig(output_dir=str(path), auto_reference=False, sync_mode="software",
        warmup_frames=0, settle_ms=0, retry_delay_ms=0, **extra)


@pytest.mark.parametrize("problem", ["crc", "chunk", "timeout"])
def test_failed_group_is_replaced_as_a_whole_without_repeating_saved_groups(tmp_path, problem):
    backend = rig()
    original = backend.capture
    calls, originals = [], []
    def flaky(timeout, cancel):
        frames, trigger = original(timeout, cancel)
        calls.append(backend.counter)
        if len(calls) == 2:  # One good group was already committed.
            if problem == "timeout":
                raise SDKError("spinCameraGetNextImageEx", -1011, "timeout")
            if problem == "crc":
                frames[1].transport["crc_passed"] = False
            else:
                frames[1].chunk["exposure_us"] = 999999.
        else:
            originals.extend(frames)
        return frames, trigger
    backend.capture = flaky
    try:
        result = run_capture(backend, config(tmp_path))
        assert verify_dataset(result)["frames"] == 12
        manifest = json.loads((result / "manifest.json").read_text())
        assert len(calls) == 5 and len(manifest["groups"]) == 4
        recovery = manifest["groups"][1]["trigger"]["recovery"]
        assert recovery["attempts"] == 2 and len(recovery["discarded_groups"]) == 1
        for record, frame in zip(manifest["frames"], originals):
            meta = json.loads((result / record["metadata"]).read_text())
            raw = np.fromfile(result / record["raw"], dtype=meta["dtype"]).reshape(meta["shape"])
            np.testing.assert_array_equal(raw, frame.pixels)
    finally:
        backend.close()


def test_persistent_crc_does_not_publish_partial_as_complete(tmp_path):
    backend = rig()
    original, calls = backend.capture, []
    def broken(timeout, cancel):
        frames, trigger = original(timeout, cancel)
        calls.append(1)
        frames[0].transport["crc_passed"] = False
        return frames, trigger
    backend.capture = broken
    try:
        with pytest.raises(RetryableCaptureError, match="자동 복구 2회"):
            run_capture(backend, config(tmp_path))
        assert len(calls) == 3 and not list(tmp_path.rglob("COMPLETE.json"))
        partial = next(tmp_path.glob(".*.partial"))
        manifest = json.loads((partial / "manifest.json").read_text())
        assert not manifest["frames"]
        assert manifest["recovery_failure"]["attempts"] == 3
    finally:
        backend.close()


def test_unsupported_format_is_not_retried(tmp_path):
    backend = rig()
    original, calls = backend.capture, []
    def rgb(timeout, cancel):
        frames, trigger = original(timeout, cancel)
        calls.append(1)
        frames[0].pixel_format = "RGB16"
        return frames, trigger
    backend.capture = rgb
    try:
        with pytest.raises(CaptureError, match="Bayer RAW"):
            run_capture(backend, config(tmp_path))
        assert len(calls) == 1
    finally:
        backend.close()


def test_cancel_stops_recovery_before_another_trigger(tmp_path):
    backend, cancel, calls = rig(), Event(), []
    def timeout(*args):
        calls.append(1)
        raise RetryableCaptureError("timeout")
    backend.capture = timeout
    def progress(message, *args):
        if message.startswith("자동 복구"):
            cancel.set()
    try:
        with pytest.raises(Cancelled):
            run_capture(backend, config(tmp_path), cancel, progress)
        assert len(calls) == 1 and not list(tmp_path.rglob("COMPLETE.json"))
    finally:
        backend.close()


def test_access_and_storage_errors_are_not_classified_as_transient():
    assert not is_transient(SDKError("spinCameraInit", -1005, "Access denied"))
    assert not is_transient(OSError("Disk write failed"))
    wrapped = CaptureError("카메라 연결 실패")
    wrapped.__cause__ = SDKError("spinCameraInit", -1010, "IO")
    assert is_transient(wrapped)


def test_warmup_recovery_and_reset_frame_ids_are_recorded(tmp_path):
    backend = rig()
    original = backend.capture
    calls = []
    def reset_ids():
        backend.counter = 0
        backend.running = False
    backend.stop = reset_ids
    def stale_once(timeout, cancel):
        frames, trigger = original(timeout, cancel)
        calls.append(1)
        if len(calls) == 1:
            frames[0].chunk["exposure_us"] = 999999
        return frames, trigger
    backend.capture = stale_once
    cfg = config(tmp_path)
    cfg.warmup_frames = 1
    try:
        result = run_capture(backend, cfg)
        manifest = json.loads((result / "manifest.json").read_text())
        assert verify_dataset(result)["frames"] == 12
        assert manifest["warmup_recoveries"][0]["step"] == 0
        assert manifest["warmup_recoveries"][0]["recovery"]["attempts"] == 2
    finally:
        backend.close()


def test_handle_release_continues_after_stop_error():
    class Nodes:
        def close(self): calls.append("nodes")
    class API:
        def call(self, name, *args): calls.append(name)
    calls = []
    backend = object.__new__(SpinnakerBackend)
    info = CameraInfo("TEST_1", "Test camera", True)
    backend.devices = [Device(info, H(1), Nodes(), Nodes(), {})]
    backend.cameras, backend.api = [info], API()
    def broken_stop(): raise CaptureError("stop error")
    backend.stop = broken_stop
    with pytest.raises(CaptureError, match="stop error"):
        backend._release_devices()
    assert calls == ["nodes", "nodes", "spinCameraDeInit", "spinCameraRelease"]
    assert not backend.devices and not backend.cameras
