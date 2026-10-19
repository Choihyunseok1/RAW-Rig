import json
from pathlib import Path
from threading import Event

import numpy as np
import pytest

from rawrig.demo import DemoBackend
from rawrig.models import CaptureConfig, CaptureError, Cancelled
from rawrig.noise import NoiseConfig, run_noise_phase, verify_phase
from rawrig.illumination import PILOT_GROUPS


class FlickerBackend(DemoBackend):
    def __init__(self, mode="flicker", cameras=1):
        super().__init__()
        self.mode = mode
        self.phase_counter = 0
        self.originals = {}
        self.open([f"DEMO_{i + 1}" for i in range(cameras)])

    def _settings(self, index):
        return {**super()._settings(index), "width": 64, "height": 64}

    def prepare(self, preset, config):
        self.phase_counter = 0
        return super().prepare(preset, config)

    def capture(self, *args):
        frames, trigger = super().capture(*args)
        index = self.phase_counter
        self.phase_counter += 1
        for frame in frames:
            signal = frame.settings["exposure_us"] * 0.3
            pixels = self.rng.normal(1000, 2, (64, 64))
            if self.noise_phase == "bright":
                state = ["off", "off", "on", "on", "off", "saturated", "band", "on"][
                    index % 8
                ]
                if (
                    self.mode in ("stable", "center_only", "low_short")
                    or frame.camera.serial == "DEMO_1"
                    and len(frames) > 1
                ):
                    state = "on"
                if self.mode == "low_short":
                    signal = max(0, frame.settings["exposure_us"] - 1000) * 0.3
                if self.mode == "capped":
                    state = "off"
                if self.mode == "saturated":
                    state = "saturated"
                if self.mode == "never_returns" and index >= PILOT_GROUPS:
                    state = "off"
                if self.mode == "unstable":
                    signal = (index + 1) * 100
                    state = "on"
                if state == "on":
                    pixels += signal
                elif state == "band":
                    pixels[32:] += signal
                elif state == "saturated":
                    pixels[:] = 65504  # Below the advertised 65520 white level.
                if self.mode == "center_only" and index % 2:
                    pixels[24:40, 24:40] += signal
            frame.pixels = np.clip(pixels, 0, 65535).astype(np.uint16)
            self.originals[frame.camera.serial, frame.frame_id] = frame.pixels.copy()
        return frames, trigger


def config(tmp_path):
    return CaptureConfig(
        output_dir=str(tmp_path),
        expected_cameras=1,
        auto_reference=False,
        sync_mode="software",
        settle_ms=0,
        warmup_frames=0,
    )


def protocol(frames=24):
    return NoiseConfig(
        frames=frames,
        bias_frames=24,
        levels=3,
        roi_size=16,
        exposure_min_us=1000,
        exposure_max_us=10000,
        max_flicker_percent=0.1,
    )


@pytest.mark.parametrize("cameras", [1, 2])
def test_flicker_replacement_keeps_exact_unmodified_groups_and_direct_pixel_variance(
    tmp_path, cameras
):
    b = FlickerBackend(cameras=cameras)
    cfg, noise = config(tmp_path), protocol()
    dark = run_noise_phase(b, cfg, noise, "dark")
    bright = run_noise_phase(b, cfg, noise, "bright", dark["path"])
    root = Path(bright["path"])
    _, manifest = verify_phase(root, "bright")
    ids_by_point = {}
    for record in manifest["stacks"]:
        meta = json.loads((root / record["metadata"]).read_text())
        audit = json.loads((root / record["illumination_audit"]).read_text())
        assert audit["complete"] and audit["accepted_groups"] == noise.frames
        assert audit["rejected_groups"] > 0 and audit["pilot_groups"] == PILOT_GROUPS
        assert all(m["monitor_outside_roi"] for m in audit["models"].values())
        reasons = [
            reason for e in audit["observations"] for reason in e["reasons"].values()
        ]
        assert any("소등" in r for r in reasons) and any("포화" in r for r in reasons)
        ids = [f["frame_id"] for f in meta["frames"]]
        index = record["point"]["index"]
        if index in ids_by_point:
            assert (
                ids == ids_by_point[index]
            )  # A bad peer discards the whole trigger group.
        ids_by_point[index] = ids
        x, y, w, h = meta["roi"]
        originals = np.stack(
            [b.originals[record["serial"], i][y : y + h, x : x + w] for i in ids]
        )
        saved = np.fromfile(root / record["raw"], dtype="<u2").reshape(
            noise.frames, h, w
        )
        np.testing.assert_array_equal(saved, originals)
        with np.load(root / record["stats"]) as stats:
            np.testing.assert_allclose(
                stats["mean"], originals.mean(axis=0), atol=1e-10
            )
            np.testing.assert_allclose(
                stats["variance"],
                originals.astype(float).var(axis=0, ddof=1),
                atol=1e-10,
            )
        assert np.all(saved.mean((1, 2)) > 1200)
    # Detection of light loss is distinct from RAW transport corruption.
    record = manifest["stacks"][0]
    with (root / record["illumination_audit"]).open("ab") as stream:
        stream.write(b"corrupt audit")
    with pytest.raises(CaptureError, match="체크섬"):
        verify_phase(root, "bright")
    b.close()


def test_stable_shot_noise_is_not_trimmed_by_roi_residuals(tmp_path):
    b = FlickerBackend("stable")
    cfg, noise = config(tmp_path), protocol(frames=180)
    dark = run_noise_phase(b, cfg, noise, "dark")
    bright = run_noise_phase(b, cfg, noise, "bright", dark["path"])
    root = Path(bright["path"])
    _, manifest = verify_phase(root, "bright")
    for record in manifest["stacks"]:
        audit = json.loads((root / record["illumination_audit"]).read_text())
        assert audit["rejected_groups"] == 0
        with np.load(root / record["stats"]) as stats:
            assert stats["variance"].mean() == pytest.approx(4, abs=0.16)
    b.close()


@pytest.mark.parametrize(
    "mode,pattern",
    [
        ("saturated", "포화"),
        ("unstable", "안정된"),
        ("never_returns", "보충 한도"),
        ("center_only", "자동 제외 후에도"),
    ],
)
def test_physical_failures_are_bounded_keep_dark_and_never_mark_bright_complete(
    tmp_path, mode, pattern
):
    b = FlickerBackend(mode)
    cfg, noise = config(tmp_path), protocol()
    dark = run_noise_phase(b, cfg, noise, "dark")
    before = b.counter
    with pytest.raises(CaptureError, match=pattern):
        run_noise_phase(b, cfg, noise, "bright", dark["path"])
    root = Path(dark["path"])
    assert not (root / "BRIGHT_COMPLETE.json").exists()
    assert verify_phase(root, "dark")[0]["status"] == "awaiting_bright"
    assert not b.running and b.noise_phase is None
    audit = json.loads(
        next(root.glob("bright_*/point_*/illumination_failed.json")).read_text()
    )
    assert not audit["complete"] and b.counter - before <= 32 + 96 + 256
    b.mode = "stable"
    run_noise_phase(b, cfg, noise, "bright", dark["path"])
    assert verify_phase(root, "bright")[1]["phase"] == "bright"
    b.close()


def test_cancel_during_pilot_does_not_count_pilot_as_measurement(tmp_path):
    b = FlickerBackend()
    cfg, noise = config(tmp_path), protocol()
    dark = run_noise_phase(b, cfg, noise, "dark")
    event = Event()

    def progress(message, done, total):
        if "사전 검사 3/" in message:
            event.set()

    with pytest.raises(Cancelled):
        run_noise_phase(b, cfg, noise, "bright", dark["path"], event, progress)
    root = Path(dark["path"])
    assert verify_phase(root, "dark") and not (root / "BRIGHT_COMPLETE.json").exists()
    audit = json.loads(
        next(root.glob("bright_*/point_*/illumination_failed.json")).read_text()
    )
    assert audit["accepted_groups"] == 0 and len(audit["observations"]) == 3
    assert next(root.glob("bright_*/point_*/*.raw")).stat().st_size == 0
    b.close()


def test_cap_is_detected_before_long_bright_sequence_and_valid_short_darklike_level_is_retained(
    tmp_path,
):
    b = FlickerBackend("capped")
    cfg, noise = config(tmp_path), protocol()
    dark = run_noise_phase(b, cfg, noise, "dark")
    root = Path(dark["path"])
    before = b.counter
    with pytest.raises(CaptureError, match="렌즈 캡"):
        run_noise_phase(b, cfg, noise, "bright", root)
    assert b.counter - before == 32
    audit = json.loads(next(root.glob("bright_*/LIGHT_PREFLIGHT.json")).read_text())
    assert not audit.get("passed") and not audit["illuminated_serials"]
    assert not list(root.glob("bright_*/point_*"))
    assert verify_phase(root, "dark") and not (root / "BRIGHT_COMPLETE.json").exists()
    b.mode = "low_short"
    result = run_noise_phase(b, cfg, noise, "bright", root)
    _, manifest = verify_phase(result["path"], "bright")
    r = manifest["stacks"][0]
    audit = json.loads((root / r["illumination_audit"]).read_text())
    assert audit["models"]["DEMO_1"]["state"] == "below_detection"
    assert audit["accepted_groups"] == noise.frames
    b.close()


def test_near_dark_pilot_with_rare_threshold_crossings_is_not_a_missing_bright_plateau():
    from rawrig.illumination import fit_light_model, light_rejection

    values = np.full(96, 633.85)
    values[:3] = [634.55, 634.53, 634.54]
    pilot = [dict(means_dn=[v]*4, mean_dn=v, sem_dn=[.11]*4,
                  monitor_pixel_counts=[16384]*4, dark_mean_dn=633.144,
                  dark_common_mean_sigma_dn=.2145, roi_near_white_fraction=0.,
                  monitor_outside_roi=True, labels=['R','G1','G2','B']) for v in values]
    model = fit_light_model(pilot, .1)
    assert model['state'] == 'below_detection'
    assert light_rejection(pilot[0], model) == ''
    # An actual populated illuminated state is retained, not classified away.
    for item in pilot[:32]:
        item.update(means_dn=[650.]*4,mean_dn=650.)
    assert fit_light_model(pilot, .1)['state'] == 'illuminated'


def test_common_ripple_passed_by_regional_bounds_is_rejected_by_dense_pooled_photometry():
    from rawrig.illumination import fit_light_model, light_rejection

    # A small scene-wide 0.23% change used to pass every regional bound (texture
    # made those bounds several DN wide). Model independent guard photometry on
    # 2.2 million pixels, without looking at any analysis-ROI residual.
    pilot = []
    base = np.array([215.0, 395.0, 397.0, 200.0] * 4)
    offset = 16.4
    counts = [140000] * 16
    for index in range(96):
        ripple = [-0.004, -0.002, 0.0, 0.002, 0.004, 0.0, 0.0003, -0.0003][index % 8]
        means = offset + (base - offset) * (1 + ripple)
        pilot.append(
            {
                "means_dn": means.tolist(),
                "sem_dn": [0.11] * 16,
                "monitor_pixel_counts": counts,
                "mean_dn": float(means.mean()),
                "dark_mean_dn": offset,
                "roi_near_white_fraction": 0.0,
                "monitor_outside_roi": True,
                "labels": [f"part_{i}" for i in range(16)],
            }
        )
    model = fit_light_model(pilot, 0.1)
    baseline = {**pilot[2], "mean_dn": float(base.mean())}
    assert light_rejection(baseline, model) == ""
    varied = {**baseline, "means_dn": (offset + (base - offset) * 1.0023).tolist()}
    varied["mean_dn"] = float(np.mean(varied["means_dn"]))
    assert np.all(
        np.abs(np.array(varied["means_dn"]) - model["center_dn"]) <= model["bound_dn"]
    )
    assert "전체 밝기" in light_rejection(varied, model)
    # Repeatable decisions for already stored v1 audits are retained.
    legacy = {**model, "schema_version": 1}
    assert light_rejection(varied, legacy) == ""


def test_stable_illumination_rms_is_not_mistaken_for_a_per_frame_peak_limit():
    from rawrig.illumination import fit_light_model, light_rejection

    pilot = []
    for ripple in np.linspace(-0.0009, 0.0009, 32):
        pilot.append(
            {
                "means_dn": [15000 * (1 + ripple)] * 4,
                "mean_dn": 15000 * (1 + ripple),
                "sem_dn": [0.3] * 4,
                "monitor_pixel_counts": [49000] * 4,
                "dark_mean_dn": 16.0,
                "roi_near_white_fraction": 0.0,
                "monitor_outside_roi": True,
                "labels": ["R", "G1", "G2", "B"],
            }
        )
    model = fit_light_model(pilot, 0.1)
    assert model["acceptance_policy"].startswith("stable RMS")
    ordinary = {**pilot[0], "means_dn": [15030.0] * 4, "mean_dn": 15030.0}
    assert light_rejection(ordinary, model) == ""
    assert light_rejection(
        {**ordinary, "means_dn": [1500.0] * 4, "mean_dn": 1500.0}, model
    )


def test_extended_pilot_uses_stable_tail_after_an_exposure_transition():
    from rawrig.illumination import fit_light_model, light_rejection

    pilot = []
    for value in np.r_[
        np.linspace(45200.0, 45550.0, 32), 45000.0 + np.sin(np.arange(32)) * 10
    ]:
        pilot.append(
            {
                "means_dn": [value] * 4,
                "mean_dn": value,
                "sem_dn": [0.5] * 4,
                "monitor_pixel_counts": [49000] * 4,
                "dark_mean_dn": 16.0,
                "roi_near_white_fraction": 0.0,
                "monitor_outside_roi": True,
                "labels": ["R", "G1", "G2", "B"],
            }
        )
    model = fit_light_model(pilot, 0.1)
    assert model["reference_pilot_start_index"] == 32
    assert model["scalar_center_dn"] == pytest.approx(45000.0, abs=10)
    assert (
        light_rejection(
            {**pilot[-1], "means_dn": [45010.0] * 4, "mean_dn": 45010.0}, model
        )
        == ""
    )


def test_dense_monitor_keeps_static_texture_out_of_roi_and_does_not_subsample_its_mean():
    from rawrig.illumination import describe_light
    from rawrig.models import Frame, CameraInfo

    yy, xx = np.mgrid[:512, :768]
    pixels = (200 + (xx * 17 + yy * 31) % 180).astype(np.uint16)
    roi = (256, 128, 256, 256)
    frame = Frame(
        CameraInfo("texture", "test", True),
        pixels,
        "BayerRG16",
        1,
        1,
        1,
        {"sensor_white_level": 65520},
        {},
        {},
    )
    before = describe_light(frame, roi, 16.4)
    pixels[128:384, 256:512] = 60000  # Changing ROI noise must not affect the guard.
    after = describe_light(frame, roi, 16.4)
    assert before["mean_dn"] == after["mean_dn"]
    assert before["means_dn"] == after["means_dn"]
    assert sum(before["monitor_pixel_counts"]) == 512 * 512 - 256 * 256
    outside = np.zeros(pixels.shape, dtype=bool)
    outside[:, 128:640] = True
    outside[128:384, 256:512] = False
    assert before["mean_dn"] == pytest.approx(float(pixels[outside].mean()), abs=1e-12)
    pixels[:, :128] = 65535
    pixels[:, 640:] = 0
    assert describe_light(frame, roi, 16.4)["mean_dn"] == before["mean_dn"]


def test_continuous_daylight_ripple_is_filled_without_trimming_analysis_pixels(
    tmp_path,
):
    import time
    from rawrig.models import Frame, check_cancel

    class Daylight(DemoBackend):
        def __init__(self):
            super().__init__()
            self.open(["DEMO_1"])
            y, x = np.mgrid[:768, :1152]
            self.flat = 1000 + 30 * np.sin(x * 0.1) + 15 * np.cos(y * 0.1)

        def _settings(self, index):
            return {**super()._settings(index), "width": 1152, "height": 768}

        def capture(self, timeout_ms, cancel):
            check_cancel(cancel)
            self.counter += 1
            s = self._settings(0)
            signal = s["exposure_us"] * 0.3 if self.noise_phase == "bright" else 0.0
            ripple = self.rng.normal(0, 0.00528) if signal else 0.0
            sigma = 8.0 if signal else 2.0
            pixels = np.clip(
                self.flat
                + signal * (1 + ripple)
                + self.rng.normal(0, sigma, self.flat.shape),
                0,
                65535,
            ).astype(np.uint16)
            frame = Frame(
                self.cameras[0],
                pixels,
                "BayerRG16",
                self.counter,
                self.counter * 100000000,
                time.monotonic_ns(),
                s,
                {"exposure_us": s["exposure_us"], "gain_db": s["gain_db"]},
                {"incomplete": False, "crc_passed": None},
            )
            return [frame], {"mode": "software", "simulated": True}

    b = Daylight()
    cfg = config(tmp_path)
    noise = NoiseConfig(
        frames=16,
        bias_frames=16,
        levels=3,
        roi_size=256,
        exposure_min_us=1000,
        exposure_max_us=10000,
        max_flicker_percent=0.1,
    )
    dark = run_noise_phase(b, cfg, noise, "dark")
    bright = run_noise_phase(b, cfg, noise, "bright", dark["path"])
    root = Path(bright["path"])
    _, manifest = verify_phase(root, "bright")
    for r in manifest["stacks"]:
        audit = json.loads((root / r["illumination_audit"]).read_text())
        assert audit["accepted_groups"] == 16 and audit["rejected_groups"] > 0
        assert audit["excess_flicker_lower_bound_fraction"]["DEMO_1"] <= 0.001
        assert audit["models"]["DEMO_1"]["monitor_outside_roi"]
    with np.load(root / manifest["stacks"][0]["stats"]) as stats:
        assert stats["variance"].mean() == pytest.approx(64, rel=0.025)
    b.close()


@pytest.mark.parametrize("tamper_first", [False, True])
def test_bright_resume_preserves_verified_points_and_recaptures_corruption(
    tmp_path, tamper_first
):
    class InterruptedBright(FlickerBackend):
        fail = False

        def capture(self, *args):
            frames, trigger = super().capture(*args)
            if (
                self.fail
                and self.noise_phase == "bright"
                and frames[0].settings["exposure_us"] >= 9999
                and self.phase_counter > PILOT_GROUPS
            ):
                self.fail = False
                raise CaptureError("test interruption at final point")
            return frames, trigger

    b = InterruptedBright("stable")
    cfg, noise = config(tmp_path), protocol()
    dark = run_noise_phase(b, cfg, noise, "dark")
    root = Path(dark["path"])
    b.fail = True
    with pytest.raises(CaptureError, match="test interruption"):
        run_noise_phase(b, cfg, noise, "bright", root)
    committed = json.loads((root / "bright_checkpoint.json").read_text())["stacks"]
    assert [r["point"]["index"] for r in committed] == [0, 1]
    if tamper_first:
        with (root / committed[0]["raw"]).open("r+b") as stream:
            stream.write(b"\x00\x00")
    run_noise_phase(b, cfg, noise, "bright", root)
    _, manifest = verify_phase(root, "bright")
    assert len(manifest["stacks"]) == 3
    assert manifest["stacks"][1] == committed[1]
    if tamper_first:
        assert manifest["stacks"][0]["raw"] != committed[0]["raw"]
    else:
        assert manifest["stacks"][0] == committed[0]
    b.close()
