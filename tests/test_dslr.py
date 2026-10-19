import pytest

from rawrig.dslr import GPhoto, NikonCompanion, nearest_choice, parse_widgets, shutter_us, validate_nef
from rawrig.models import CaptureConfig, CaptureError


def test_parse_camera_choices_preserves_exact_values():
    widgets = parse_widgets("/main/capturesettings/shutterspeed\nLabel: Shutter Speed\nReadonly: 0\nType: RADIO\nCurrent: 1/50\nChoice: 0 1/100\nChoice: 1 1/50\nEND\n/main/imgsettings/imagequality\nCurrent: NEF (Raw)\nChoice: 0 NEF (Raw)\nChoice: 1 NEF + JPEG Fine\n")
    assert widgets["shutterspeed"]["choices"] == ["1/100", "1/50"]
    assert widgets["imagequality"]["choices"] == ["NEF (Raw)", "NEF + JPEG Fine"]


def test_quantization_and_unsupported_shutter_values():
    assert shutter_us("1/8000") == 125
    assert shutter_us("0.5") == 500000
    assert shutter_us("Bulb") is None
    choice, actual, error = nearest_choice(["Bulb", "1/80", "1/100"], 12000, shutter_us, .2)
    assert choice == "1/80" and actual == 12500 and 0 < error < .1
    with pytest.raises(CaptureError, match="지원값"):
        nearest_choice(["1/8000", "1/100"], 15, shutter_us, .2)


@pytest.mark.parametrize("payload", [b"JPEG data", b"II*\x00truncated", b"MM\x00*truncated"])
def test_invalid_or_truncated_nef_never_passes(payload):
    with pytest.raises(CaptureError):
        validate_nef(payload)


class FakeGPhoto:
    def __init__(self, *args):
        self.values = {"/main/isoauto": "On", "/main/imagequality": "JPEG Fine", "/main/rawcompression": "Lossy", "/main/shutterspeed": "1/100", "/main/iso": "100"}
        self.calls = []
    def run(self, action, value, **kwargs):
        self.calls.append((action, value))
        if action == "--set-config-value":
            key, val = value.split("=", 1)
            self.values[key] = val
            return ""
        return "Current: " + self.values[value] + "\n"


def companion():
    camera = NikonCompanion.__new__(NikonCompanion)
    camera.api = FakeGPhoto()
    camera.original, camera.applied, camera.restore_errors = {}, {}, []
    choices = {"isoauto": ["Off", "On"], "imagequality": ["JPEG Fine", "NEF (Raw)", "NEF + JPEG"],
        "rawcompression": ["Lossless", "Lossy"], "shutterspeed": ["1/50", "1/100"], "iso": ["100", "200"]}
    camera.widgets = {name: {"path": "/main/" + name, "current": camera.api.values["/main/" + name], "choices": values, "readonly": "0"} for name, values in choices.items()}
    return camera


def test_nikon_preflight_raw_only_autoiso_off_and_restore():
    camera = companion()
    plan = [{"applied": {"CAM_1": {"exposure_us": 20000, "gain_db": 0}}}]
    result = camera.preflight(plan, CaptureConfig(dslr_enabled=True))
    assert result[0]["iso"] == 100
    assert camera.api.values["/main/imagequality"] == "NEF (Raw)"
    assert camera.api.values["/main/isoauto"] == "Off"
    assert camera.api.values["/main/rawcompression"] == "Lossless"
    assert camera.api.values["/main/shutterspeed"] == "1/50"
    camera.close()
    assert camera.api.values["/main/imagequality"] == "JPEG Fine"
    assert camera.api.values["/main/isoauto"] == "On"
    assert camera.api.values["/main/shutterspeed"] == "1/100"


def test_follow_gain_is_explicit_nominal_iso_not_gige_gain():
    camera = companion()
    plan = [{"applied": {"CAM_1": {"exposure_us": 20000, "gain_db": 6.020599913}}}]
    result = camera.preflight(plan, CaptureConfig(dslr_enabled=True, dslr_follow=True))
    assert result[0]["iso"] == 200
    assert "no radiometric equivalence" in result[0]["gain_to_iso"]


def test_gphoto_detection_filters_headers_without_usb_device(monkeypatch):
    monkeypatch.setattr(GPhoto, "run", lambda *_: "Model Port\n---\nNikon Z5                      usb:001,003\nNikon Z50 usb:001,004\n")
    api = GPhoto.__new__(GPhoto)
    assert api.detect() == [{"model": "Nikon Z5", "port": "usb:001,003"}, {"model": "Nikon Z50", "port": "usb:001,004"}]


class FakeNikon:
    """Protocol fixture only; its bytes are deliberately not represented as a real NEF."""
    restore_errors = []
    def preflight(self, plan, config):
        return [{"exposure_us": 20000, "iso": 100} for _ in plan]
    def prepare(self, setting, config):
        pass
    def capture(self, setting, config):
        return b"UNIT_TEST_ONLY", {"settings": setting, "raw_validation": {"shape": [2, 2]},
            "timing": {"usb_command_started_monotonic_ns": 1}}
    def close(self):
        pass


def test_companion_group_storage_and_reverification(tmp_path, monkeypatch):
    from rawrig.demo import DemoBackend
    from rawrig.engine import run_capture
    from rawrig.storage import verify_dataset
    monkeypatch.setattr("rawrig.dslr.validate_nef", lambda _: {"shape": [2, 2], "exposure_us": 20000, "iso": 100})
    rig = DemoBackend()
    rig.open([c.serial for c in rig.discover()])
    result = run_capture(rig, CaptureConfig(output_dir=str(tmp_path), capture_strategy="cross_reference",
        dslr_enabled=True, warmup_frames=0, settle_ms=0), companion=FakeNikon())
    assert verify_dataset(result)["nef_files"] == 3
    nef = next(result.rglob("*.nef"))
    nef.write_bytes(b"changed")
    with pytest.raises(CaptureError, match="체크섬"):
        verify_dataset(result)


def test_companion_failure_never_publishes_complete_dataset(tmp_path):
    from rawrig.demo import DemoBackend
    from rawrig.engine import run_capture
    rig = DemoBackend()
    rig.open([c.serial for c in rig.discover()])
    camera = FakeNikon()
    def fail(*_):
        raise CaptureError("USB capture failed")
    camera.capture = fail
    with pytest.raises(CaptureError, match="USB capture failed"):
        run_capture(rig, CaptureConfig(output_dir=str(tmp_path), dslr_enabled=True, warmup_frames=0, settle_ms=0), companion=camera)
    assert not list(tmp_path.rglob("COMPLETE.json"))
    assert len(list(tmp_path.rglob("FAILED.json"))) == 1


def test_industrial_failure_joins_usb_then_reacquires_complete_group(tmp_path, monkeypatch):
    import json
    import time
    from threading import Event
    from rawrig.demo import DemoBackend
    from rawrig.engine import run_capture
    from rawrig.models import RetryableCaptureError
    from rawrig.storage import verify_dataset
    monkeypatch.setattr("rawrig.dslr.validate_nef", lambda _: {"shape": [2, 2], "exposure_us": 20000, "iso": 100})
    rig = DemoBackend()
    rig.open([c.serial for c in rig.discover()])
    camera, finished, calls = FakeNikon(), Event(), []
    original_usb, original_industrial = camera.capture, rig.capture
    def usb(setting, config):
        finished.clear()
        time.sleep(.04)
        payload, metadata = original_usb(setting, config)
        metadata["usb_attempt"] = len(calls)
        finished.set()
        return payload, metadata
    def industrial(timeout, cancel):
        if len(calls) == 1:
            assert finished.is_set(), "USB from rejected group is still active"
        calls.append(1)
        if len(calls) == 1:
            raise RetryableCaptureError("industrial timeout")
        return original_industrial(timeout, cancel)
    camera.capture, rig.capture = usb, industrial
    result = run_capture(rig, CaptureConfig(output_dir=str(tmp_path), capture_strategy="per_camera",
        dslr_enabled=True, warmup_frames=0, settle_ms=0, retry_delay_ms=0), companion=camera)
    assert verify_dataset(result)["nef_files"] == 1 and len(calls) == 2
    manifest = json.loads((result / "manifest.json").read_text())
    meta = json.loads((result / manifest["companions"][0]["metadata"]).read_text())
    assert meta["usb_attempt"] == 2
    assert manifest["groups"][0]["trigger"]["recovery"]["attempts"] == 2
