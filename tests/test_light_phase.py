from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest

from rawrig.illumination import (
    excess_flicker,
    excess_pair_flicker,
    lit_groups,
    fit_drift_light_model,
    light_rejection,
)
from rawrig.light_phase import (
    fit_light_phase,
    phase_target_ns,
    fit_phase_servo,
    phase_for_exposure,
)
from rawrig.spinnaker import SpinnakerBackend


@pytest.mark.parametrize("position,sign", [("exposure_start", 1), ("exposure_end", -1)])
def test_integrated_reference_phase_transfers_to_short_exposure(position, sign):
    f = 120.013
    ref = 1351.0
    angle = 0.7 + sign * 2 * np.pi * f * ref / 2e6
    amplitude = ref * np.sinc(f * ref / 1e6)
    model = {
        "frequency_hz": f,
        "period_ns": 1e9 / f,
        "harmonic_coefficients": [
            amplitude * np.cos(angle),
            -amplitude * np.sin(angle),
            0,
            0,
            0,
            0,
        ],
    }
    result = phase_for_exposure(model, ref, target, position)
    phase = 0.7 + sign * 2 * np.pi * f * target / 2e6
    expected = target * np.sinc(f * target / 1e6)
    np.testing.assert_allclose(
        result["harmonic_coefficients"][:2],
        [expected * np.cos(phase), -expected * np.sin(phase)],
    )
    assert np.cos(2 * np.pi * f * result["phase_ns"] / 1e9 + phase) > 0.99999


def test_light_reference_restores_actual_exposure_even_on_failure():
    class Nodes:
        def floating(self, name, value, **kwargs):
            return value

    backend = SpinnakerBackend.__new__(SpinnakerBackend)
    d = SimpleNamespace(nodes=Nodes(), settings={"exposure_us": 35.0})
    backend.devices = [d]
    backend.config = SimpleNamespace(sync_mode="software")

    def capture(timeout, cancel):
        return [
            SimpleNamespace(
                chunk={"exposure_us": d.settings["exposure_us"]},
                settings=dict(d.settings),
            )
        ], {}

    backend.capture = capture
    frames, _ = backend.capture_light_reference(1351, 1000, Event())
    assert frames[0].chunk["exposure_us"] == 1351 and d.settings["exposure_us"] == 35

    def failed(timeout, cancel):
        raise RuntimeError("transport failed")

    backend.capture = failed
    with pytest.raises(RuntimeError):
        backend.capture_light_reference(1351, 1000, Event())
    assert d.settings["exposure_us"] == 35


@pytest.mark.parametrize("target", [35.0, 12.0])
def test_independent_long_reference_never_enters_short_exposure_noise_pairs(target):
    epoch = 1760000000000000000
    f = 120.0
    ref = 1351.0
    phase = 0.7 + 2 * np.pi * f * ref / 2e6
    amplitude = 0.003 * ref * np.sinc(f * ref / 1e6)
    long_model = {
        "epoch_ns": epoch,
        "frequency_hz": f,
        "period_ns": 1e9 / f,
        "phase_ns": (-phase % (2 * np.pi)) / (2 * np.pi) * 1e9 / f,
        "harmonic_coefficients": [
            amplitude * np.cos(phase),
            -amplitude * np.sin(phase),
            0,
            0,
            0,
            0,
        ],
    }
    short_model = phase_for_exposure(long_model, ref, target)
    state = {"model": short_model, "fraction": None, "now": epoch + 1000000000, "id": 0}

    def control(model, fraction=None):
        state.update(model=model, fraction=fraction)
        return True

    def take(exposure):
        stamp = phase_target_ns(
            state["now"], state["model"], phase_fraction=state["fraction"]
        )
        state["now"] = stamp + 55000000
        state["id"] += 1
        t = (stamp - epoch) / 1e9
        mean = (
            633
            + 0.04 * exposure
            + 0.003
            * exposure
            * np.sinc(f * exposure / 1e6)
            * np.cos(2 * np.pi * f * t + np.pi * f * exposure / 1e6 + 0.7)
        )
        description = {
            "camera_timestamp_ns": stamp,
            "frame_id": state["id"],
            "exposure_us": exposure,
            "mean_dn": mean,
            "means_dn": [mean] * 4,
            "sem_dn": [0.04] * 4,
            "dark_mean_dn": 633.0,
            "monitor_pixel_counts": [16384] * 4,
            "monitor_outside_roi": True,
            "labels": ["R", "G1", "G2", "B"],
            "roi_near_white_fraction": 0.0,
        }
        return [description], {}

    seed = {
        "serial": "SIM",
        "model": short_model,
        "reference_model": long_model,
        "reference_exposure_us": ref,
        "target_exposure_us": target,
        "timestamp_position": "exposure_start",
        "descriptions": [],
    }
    audit = {}
    groups = list(
        lit_groups(
            lambda: take(target),
            lambda frames: {"SIM": frames[0]},
            100,
            0.1,
            Event(),
            audit,
            lambda *args: None,
            simulated=True,
            phase_control=control,
            phase_seed=seed,
            local_pair_protocol=True,
            phase_reference_capture=lambda: take(ref),
        )
    )
    assert len(groups) == 100
    assert all(group[0][0]["exposure_us"] == target for group in groups)
    calibrated = [
        o for o in audit["observations"] if o["kind"] in ("phase_probe", "phase_servo")
    ]
    assert calibrated and all(
        o["cameras"]["SIM"]["exposure_us"] == ref for o in calibrated
    )


def test_phase_fit_preserves_large_camera_clock_and_predicts_peak():
    rng = np.random.default_rng(17)
    epoch = 1760000000000000000
    times = np.sort(rng.uniform(0, 4, 96))
    frequency = 120.013
    descriptions = [
        {
            "camera_timestamp_ns": epoch + round(t * 1e9),
            "mean_dn": 2000 + 300 * np.cos(2 * np.pi * frequency * t + 0.7),
            "sem_dn": [0.2],
        }
        for t in times
    ]
    model = fit_light_phase(descriptions)
    assert model["frequency_hz"] == pytest.approx(frequency, abs=0.0002)
    assert model["fit_r_squared"] > 0.99999
    now = epoch + 5000000000
    target = phase_target_ns(now, model)
    assert now + 2000000 <= target <= now + 2000000 + model["period_ns"]
    assert np.cos(2 * np.pi * frequency * ((target - epoch) / 1e9) + 0.7) > 0.99999
    for description in descriptions:
        description["mean_dn"] = 2000
    assert fit_light_phase(descriptions) is None


def test_guard_phase_servo_uses_actual_jittered_camera_timestamps():
    epoch = 1760000000000000000
    model = {
        "epoch_ns": epoch,
        "frequency_hz": 120.0,
        "period_ns": 1e9 / 120,
        "phase_ns": 0.0,
        "harmonic_coefficients": [15, 0, 0, 0, 0, 0],
        "last_observed_camera_timestamp_ns": epoch + 500000000,
    }
    rng = np.random.default_rng(5)
    descriptions = []
    for i, offset in enumerate((-0.1, 0, 0.1, 0, 0.1, 0, -0.1, 0)):
        now = epoch + 2000000000 + i * 75000000
        stamp = phase_target_ns(now, model, phase_fraction=offset % 1) + round(
            rng.normal(0, 70000)
        )
        t = (stamp - epoch) / 1e9
        descriptions.append(
            {
                "camera_timestamp_ns": stamp,
                "mean_dn": 650 + 16.5 * np.cos(2 * np.pi * 120 * t - 0.21) + 0.3 * t,
            }
        )
    updated = fit_phase_servo(descriptions, model)
    assert updated["servo_fit_r_squared"] > 0.99999
    assert updated["servo_phase_shift_ns"] == pytest.approx(
        0.21 / (2 * np.pi) / 120 * 1e9, abs=500
    )
    target = phase_target_ns(updated["epoch_ns"] + 20000000, updated)
    assert np.cos(2 * np.pi * 120 * ((target - epoch) / 1e9) - 0.21) > 0.99999


def test_matched_dark_common_noise_is_not_mistaken_for_led_flicker():
    # Exactly specified sample variances distinguish correlated sensor noise
    # from an added illumination fluctuation, including at low signal.
    waveform = np.sin(np.linspace(0, 20 * np.pi, 1000))
    waveform /= waveform.std(ddof=1)
    dark = {
        "mean": np.full((256, 256), 633.0),
        "variance": np.full((256, 256), 190.0),
        "frame_means": 633 + 0.22 * waveform,
        "count": 1000,
    }
    bright = {**dark, "mean": dark["mean"] + 10, "frame_means": 643 + 0.22 * waveform}
    assert excess_flicker(bright, 633) > 0.02
    assert excess_flicker(bright, 633, dark_stats=dark) < 1e-8
    bright["frame_means"] = 643 + np.sqrt(0.22**2 + 0.1**2) * waveform
    assert excess_flicker(bright, 633, dark_stats=dark) == pytest.approx(0.01)


def test_pair_estimator_separates_slow_drift_from_fast_flicker():
    rng = np.random.default_rng(84)
    count = 1000
    dark = {
        "mean": np.full((256, 256), 633.0),
        "variance": np.full((256, 256), 190.0),
        "pair_variance": np.full((256, 256), 190.0),
        "frame_means": 633 + rng.normal(0, 0.22, count),
        "count": count,
    }
    bright = {
        **dark,
        "mean": dark["mean"] + 50,
        "frame_means": 683 + rng.normal(0, 0.23, count) + np.linspace(-2, 2, count),
        "variance": np.full((256, 256), 460.0),
        "pair_variance": np.full((256, 256), 460.0),
    }
    assert excess_flicker(bright, 633, dark_stats=dark) > 0.02
    assert excess_pair_flicker(bright, 633, True, dark) == 0
    bright["frame_means"] += np.tile([-1, 1], count // 2)
    assert excess_pair_flicker(bright, 633, True, dark) > 0.02


def test_independent_drift_reference_predicts_slow_changes_but_rejects_outliers():
    reference = []
    for i in range(32):
        value = 1000 + i * 0.2
        reference.append(
            {
                "camera_timestamp_ns": 1000000000 + i * 100000000,
                "means_dn": [value] * 4,
                "mean_dn": value,
                "sem_dn": [0.1] * 4,
                "monitor_pixel_counts": [1000] * 4,
                "dark_mean_dn": 633.0,
                "dark_common_mean_sigma_dn": 0.22,
                "roi_near_white_fraction": 0,
                "monitor_outside_roi": True,
                "labels": [str(j) for j in range(4)],
            }
        )
    model = fit_drift_light_model(reference, 0.1)
    later = {
        **reference[-1],
        "camera_timestamp_ns": 9100000000,
        "means_dn": [1016.2] * 4,
        "mean_dn": 1016.2,
    }
    assert model["schema_version"] == 4
    assert light_rejection(later, model) == ""
    flicker = {**later, "means_dn": [1005.0] * 4, "mean_dn": 1005.0}
    assert light_rejection(flicker, model)


@pytest.mark.parametrize(
    "local_pairs,servo,retry_reference",
    [
        (False, False, False),
        (True, False, False),
        (True, True, False),
        (True, False, True),
    ],
)
def test_phase_probes_never_enter_accepted_stack(
    tmp_path, monkeypatch, local_pairs, servo, retry_reference
):
    if retry_reference:
        import rawrig.illumination as illumination
        from rawrig.models import CaptureError

        original = illumination.fit_drift_light_model
        reference_calls = [0]

        def transient_reference_failure(*args):
            reference_calls[0] += 1
            if reference_calls[0] == 1:
                raise CaptureError("transient independent reference instability")
            return original(*args)

        monkeypatch.setattr(
            illumination, "fit_drift_light_model", transient_reference_failure
        )
    counter = 32
    calls = []
    model = {
        "frequency_hz": 120.0,
        "epoch_ns": 0,
        "period_ns": 1e9 / 120,
        "phase_ns": 0,
    }
    if servo:
        model["harmonic_coefficients"] = [15, 0, 0, 0, 0, 0]

    def capture():
        nonlocal counter
        counter += 1
        return counter, {}

    def describe(frame):
        return {
            "camera": {
                "frame_id": frame,
                "camera_timestamp_ns": frame * 100000000,
                "mean_dn": 2000.0,
                "means_dn": [2000.0] * 4,
                "sem_dn": [0.1] * 4,
                "monitor_pixel_counts": [1000] * 4,
                "dark_mean_dn": 633.0,
                "roi_near_white_fraction": 0,
                "monitor_outside_roi": True,
                "labels": [str(i) for i in range(4)],
            }
        }

    audit = {}
    seed = {
        "serial": "camera",
        "model": model,
        "descriptions": [describe(i)["camera"] for i in range(1, 33)],
    }
    accepted = list(
        lit_groups(
            capture,
            describe,
            220,
            0.1,
            Event(),
            audit,
            lambda _: None,
            simulated=True,
            phase_control=lambda *args: calls.append(args) or True,
            phase_seed=seed,
            local_pair_protocol=local_pairs,
        )
    )
    assert len(accepted) == audit["accepted_groups"] == 220
    assert audit["phase_probe_groups"] == (64 if local_pairs else 32)
    accepted_ids = {item[0] for item in accepted}
    probe_ids = {
        e["cameras"]["camera"]["frame_id"]
        for e in audit["observations"]
        if e["kind"] == "phase_probe"
    }
    assert not accepted_ids & probe_ids
    assert len(calls) == (66 if local_pairs else 33) + (117 if servo else 0)
    if local_pairs:
        assert audit["phase_reference_groups"] == 80 + (52 if servo else 0) + (
            32 if retry_reference else 0
        )
        assert audit["phase_servo_groups"] == (104 if servo else 0)
        es = [e for e in audit["observations"] if e["kind"] == "accepted"]
        assert all(
            a["model_segment"] == b["model_segment"] for a, b in zip(es[::2], es[1::2])
        )
        # Replay the on-disk integrity check, including independently captured
        # references and the exact decision model used for each accepted frame.
        import json
        from rawrig.noise import _verify_illumination
        from rawrig.storage import sha256_file

        audit["complete"] = True
        path = tmp_path / "illumination.json"
        path.write_text(json.dumps(audit))
        record = {
            "serial": "camera",
            "illumination_audit": path.name,
            "illumination_audit_sha256": sha256_file(path),
        }
        meta = {
            "frames": [
                {
                    "frame_id": e["cameras"]["camera"]["frame_id"],
                    "illumination": e["cameras"]["camera"],
                }
                for e in es
            ]
        }
        _verify_illumination(tmp_path, record, meta, 220, {"camera"})


def test_phase_timing_requires_known_nanosecond_camera_clock():
    backend = object.__new__(SpinnakerBackend)
    ticks = [1000000000]
    nodes = SimpleNamespace(has=lambda _: True, get_int=lambda _: ticks[0])
    backend.devices = [SimpleNamespace(nodes=nodes)]
    backend.config = SimpleNamespace(sync_mode="software")
    assert backend.set_light_phase({"frequency_hz": 120})
    ticks[0] = 100000000
    assert not backend.set_light_phase({"frequency_hz": 120})
    assert backend.light_phase is None
    ticks[0] = 1000000000
    backend.config.sync_mode = "ptp"
    assert not backend.set_light_phase({"frequency_hz": 120})
    backend.config.sync_mode = "software"
    backend.devices *= 2
    assert not backend.set_light_phase({"frequency_hz": 120})
