"""Regression for 158.489 µs -> 162 µs sensor quantization."""

import math
from pathlib import Path
import json
from dataclasses import replace

import pytest

from rawrig.spinnaker import NodeMap
from rawrig.models import CaptureError, CaptureConfig
from rawrig.demo import DemoBackend
from rawrig.noise import NoiseConfig, run_noise_phase, verify_phase, noise_step_preset


class QuantizedNode(NodeMap):
    def __init__(self, kind="round"):
        self.current = 9999.0
        self.kind = kind
        self.api = self

    def bounds(self, name):
        return [9.0, 1000000.0]

    def writable(self, name):
        return True

    def node(self, name):
        return name

    def call(self, name, node, value):
        if self.kind == "round":
            self.current = round(value / 9) * 9.0
        elif self.kind == "scale":
            self.current = 2 * value

    def get_float(self, name):
        return self.current


def test_sensor_quantization_is_verified_and_frame_tolerance_is_not_widened():
    requested = 158.48931924611142
    node = QuantizedNode()
    with pytest.raises(CaptureError):
        node.floating("ExposureTime", requested, tolerance=1.0)
    assert node.resolve_float("ExposureTime", requested, 1.0) == 162.0
    assert node.current == 162.0
    for kind in ("ignored", "scale"):
        with pytest.raises(CaptureError):
            QuantizedNode(kind).resolve_float("ExposureTime", requested, 1.0)
    node.writable = lambda _: False
    with pytest.raises(CaptureError):
        node.resolve_float("ExposureTime", requested, 1.0)


def test_quantization_probe_expands_past_a_single_sensor_timing_bin():
    class TimingBin(QuantizedNode):
        def call(self, name, node, value):
            self.current = math.floor(value / 10) * 10 + 5.0

    node = TimingBin()
    assert node.resolve_float("ExposureTime", 206.22925652535872, 1.04) == 205.0
    assert node.current == 205.0


class QuantizedDemo(DemoBackend):
    def _settings(self, index):
        return {**super()._settings(index), "width": 32, "height": 32}

    def resolve_noise_plan(self, plan, cfg, cancel):
        return [
            {
                **p,
                "camera_presets": {
                    c.serial: {
                        "name": f"noise_{p['index']:03d}",
                        "exposure_us": round(p["exposure_us"] / 9) * 9.0,
                        "gain_db": p["gain_db"],
                    }
                    for c in self.cameras
                },
            }
            for p in plan
        ]

    def prepare(self, preset, cfg):
        values = list(preset.values()) if isinstance(preset, dict) else [preset]
        for p in values:
            actual = round(p.exposure_us / 9) * 9.0
            if abs(actual - p.exposure_us) > max(1.0, p.exposure_us * 0.005):
                raise CaptureError("sensor time quantum mismatch")
        return super().prepare(preset, cfg)


@pytest.mark.parametrize("period", [0.0, 1e6 / 120])
def test_all_16_points_use_verified_actual_values_in_both_dark_and_bright(
    tmp_path, period
):
    b = QuantizedDemo()
    b.open(["DEMO_1"])
    cfg = CaptureConfig(
        output_dir=str(tmp_path),
        expected_cameras=1,
        sync_mode="software",
        auto_reference=False,
        settle_ms=0,
        warmup_frames=0,
    )
    noise = replace(NoiseConfig.fixed_curve(), anti_flicker_period_us=period)
    dark = run_noise_phase(b, cfg, noise, "dark")
    s, d = verify_phase(dark["path"], "dark")
    if not period:
        assert (
            s["acquisition_plan"][1]["camera_presets"]["DEMO_1"]["exposure_us"] == 162.0
        )
        assert not math.isclose(s["acquisition_plan"][1]["exposure_us"], 162.0)
    else:
        assert len(s["acquisition_plan"]) == 12
        assert all(
            abs(
                p["camera_presets"]["DEMO_1"]["exposure_us"]
                - round(p["exposure_us"] / period) * period
            )
            < 10
            for p in s["acquisition_plan"]
        )
    bright = run_noise_phase(b, cfg, noise, "bright", dark["path"])
    _, bm = verify_phase(bright["path"], "bright")
    for dr, br in zip(d["stacks"], bm["stacks"]):
        md = json.loads((Path(dark["path"]) / dr["metadata"]).read_text())
        mb = json.loads((Path(bright["path"]) / br["metadata"]).read_text())
        actual = noise_step_preset(dr["point"])["DEMO_1"]
        assert (
            md["settings"]["exposure_us"]
            == mb["settings"]["exposure_us"]
            == actual.exposure_us
        )
        assert all(
            f["chunk"]["exposure_us"] == actual.exposure_us for f in mb["frames"]
        )
    b.close()


class LegacyQuantizedDemo(DemoBackend):
    def _settings(self, index):
        values = super()._settings(index)
        return {
            **values,
            "width": 32,
            "height": 32,
            "exposure_us": round(values["exposure_us"] / 9) * 9.0,
        }

    def prepare(self, preset, cfg):
        values = list(preset.values()) if isinstance(preset, dict) else [preset]
        if any(
            abs(round(p.exposure_us / 9) * 9.0 - p.exposure_us)
            > max(1.0, p.exposure_us * 0.005)
            for p in values
        ):
            raise CaptureError("legacy quantum mismatch")
        return super().prepare(preset, cfg)


@pytest.mark.parametrize("corrupt", [False, True])
def test_failed_1000_frame_bias_is_reused_only_after_integrity_checks(
    tmp_path, corrupt
):
    cfg = CaptureConfig(
        output_dir=str(tmp_path),
        expected_cameras=1,
        sync_mode="software",
        auto_reference=False,
        settle_ms=0,
        warmup_frames=0,
    )
    noise = NoiseConfig.fixed_curve()
    legacy = LegacyQuantizedDemo()
    legacy.open(["DEMO_1"])
    with pytest.raises(CaptureError) as failure:
        run_noise_phase(legacy, cfg, noise, "dark")
    old = Path(failure.value.noise_session_path)
    assert (old / "DARK_CHECKPOINT.json").exists()
    # Simulate a saved point from the previous application version.
    (old / "DARK_CHECKPOINT.json").unlink()
    (old / "dark_checkpoint.json").unlink()
    raw = next(old.glob("dark_*/point_000/*.raw"))
    if corrupt:
        with raw.open("r+b") as stream:
            stream.write(b"\xff\xff")
    backend = QuantizedDemo()
    backend.open(["DEMO_1"])
    result = run_noise_phase(backend, cfg, noise, "dark", str(old))
    _, manifest = verify_phase(result["path"], "dark")
    meta = json.loads(
        (Path(result["path"]) / manifest["stacks"][0]["metadata"]).read_text()
    )
    assert backend.counter == (2500 if corrupt else 1500)
    assert ("reused_from" in meta) is (not corrupt)
    assert raw.exists()
    backend.close()
    legacy.close()
