from dataclasses import asdict, replace
import json
from pathlib import Path
import shutil
from threading import Event

import numpy as np
import pytest

from rawrig.demo import DemoBackend
from rawrig.models import CaptureConfig, CaptureError, Cancelled
from rawrig.noise import (
    NoiseConfig,
    Moments,
    channel_slices,
    roi_bounds,
    run_noise_phase,
    verify_phase,
    verify_noise_session,
)
from rawrig.noise_analysis import characterize, fit_gain, analyze_noise_session


class SmallDemo(DemoBackend):
    def _settings(self, index):
        return {**super()._settings(index), "height": 64, "width": 64}


def camera():
    backend = SmallDemo()
    backend.open(["DEMO_1"])
    return backend


def capture_config(path):
    return CaptureConfig(
        output_dir=str(path),
        expected_cameras=1,
        sync_mode="software",
        auto_reference=False,
        warmup_frames=0,
        settle_ms=0,
    )


def short_config(**values):
    return NoiseConfig(
        frames=8,
        bias_frames=12,
        levels=4,
        roi_size=16,
        exposure_min_us=1000.0,
        exposure_max_us=100000.0,
        max_flicker_percent=10.0,
        **values,
    )


def test_completed_dark_reuse_accepts_new_bright_budget_and_verifies_hashes(tmp_path):
    backend=camera();cfg=capture_config(tmp_path)
    noise=short_config()
    first=run_noise_phase(backend,cfg,noise,'dark')
    _,old=verify_phase(first['path'],'dark')
    revised=replace(noise,bright_frame_counts={'0':16})
    messages=[]
    second=run_noise_phase(backend,cfg,revised,'dark',first['path'],progress=lambda m,*_:messages.append(m))
    _,new=verify_phase(second['path'],'dark')
    assert len([m for m in messages if '재사용 완료' in m])==4
    assert [r['raw_sha256'] for r in new['stacks']]==[r['raw_sha256'] for r in old['stacks']]
    assert all('reused_from' in json.loads((Path(second['path'])/r['metadata']).read_text()) for r in new['stacks'])
    # A committed hash mismatch causes real recapture, not adoption.
    r=old['stacks'][0];path=Path(first['path'])/r['raw']
    with path.open('r+b') as f:f.write(b'bad!')
    third=run_noise_phase(backend,cfg,revised,'dark',first['path'])
    _,manifest=verify_phase(third['path'],'dark')
    assert 'reused_from' not in json.loads((Path(third['path'])/manifest['stacks'][0]['metadata']).read_text())
    backend.close()


@pytest.fixture(scope="module")
def measured(tmp_path_factory):
    backend = camera()
    cfg = capture_config(tmp_path_factory.mktemp("noise"))
    noise = NoiseConfig(
        frames=100,
        bias_frames=320,
        levels=10,
        roi_size=32,
        exposure_min_us=1000.0,
        exposure_max_us=100000.0,
        max_flicker_percent=10.0,
        quantum_efficiency={"DEMO_1": 0.75},
    )
    dark = run_noise_phase(backend, cfg, noise, "dark")
    bright = run_noise_phase(backend, cfg, noise, "bright", dark["path"])
    result = analyze_noise_session(bright["path"])
    backend.close()
    return result, noise


def test_unbiased_temporal_maps_and_independent_pairs_equal_direct_numpy():
    rng = np.random.default_rng(13)
    samples = rng.integers(50000, 65000, size=(9, 16, 16), dtype=np.uint16)
    accumulator = Moments((16, 16), "BayerRG16")
    for i, pixels in enumerate(samples):
        accumulator.add(pixels, 65535, i)
    stats = accumulator.arrays()
    expected = samples.astype(float)
    np.testing.assert_allclose(stats["mean"], expected.mean(axis=0))
    np.testing.assert_allclose(stats["variance"], expected.var(axis=0, ddof=1))
    np.testing.assert_allclose(
        stats["pair_variance"],
        ((expected[:8:2] - expected[1:8:2]) ** 2).mean(axis=0) / 2,
    )
    np.testing.assert_allclose(
        stats["half_variance"][0], expected[::2].var(axis=0, ddof=1)
    )
    np.testing.assert_allclose(stats["half_mean"][1], expected[1::2].mean(axis=0))
    assert stats["sample_values"].dtype == np.uint16


@pytest.mark.parametrize(
    "pattern,labels",
    [
        ("RG", ("R", "G1", "G2", "B")),
        ("GR", ("G1", "R", "B", "G2")),
        ("GB", ("G1", "B", "R", "G2")),
        ("BG", ("B", "G1", "G2", "R")),
    ],
)
def test_cfa_planes_are_separate_without_demosaic_or_binning(pattern, labels):
    array = np.arange(16).reshape(4, 4)
    planes = channel_slices("Bayer" + pattern + "16")
    assert tuple(planes) == labels
    assert sum(array[s].size for s in planes.values()) == array.size
    assert roi_bounds((66, 70), 32)[:2] == (18, 16)
    assert roi_bounds((66, 70), 0) == (0, 0, 70, 66)


def test_known_poisson_gaussian_sensor_recovers_conversion_gain_read_noise_and_qe(
    measured,
):
    result, noise = measured
    assert verify_noise_session(result["path"])["passed"]
    report = result["report"]
    assert report["simulated"] and report["noise_config"] == asdict(noise)
    assert len(report["plots"]) == 13
    assert "PTC · 노이즈 분리 (DN)" in report["plots"][0]["label"]
    assert "PTC · 전자 기준" in report["plots"][1]["label"]
    for data in report["cameras"][0]["channels"].values():
        assert data["fit"]["valid"]
        assert data["summary"]["gain_dn_per_e"] == pytest.approx(32.0, rel=0.06)
        assert data["summary"]["read_noise_rms_e"] == pytest.approx(3.0, rel=0.06)
        assert data["fit"]["r_squared"] > 0.98
        assert data["precision"]["offset"]["theoretical_rms_full"] > 0
        row = next(r for r in data["curve"] if r["fit_used"])
        assert row["signal_photons"] == pytest.approx(row["signal_e"] / 0.75)
        assert row["relative_snr_photon"] == pytest.approx(
            row["snr_total"] / np.sqrt(row["signal_photons"])
        )
    blue = report["cameras"][0]["channels"]["B"]["summary"]
    assert blue["effective_saturation_capacity_e"] is None  # Blue channel never clips.
    assert blue["dynamic_range_db"] is None
    assert blue["capacity_lower_bound_e"] > 0
    root = Path(result["path"]) / "analysis"
    assert (root / "report.html").is_file()
    for plot in report["plots"]:
        assert (root / plot["path"]).read_bytes().startswith(b"\x89PNG")
        assert "<svg" in (root / plot["svg"]).read_text()
    with np.load(root / report["cameras"][0]["maps"]) as maps:
        assert maps["read_noise_dn"].shape == (32, 32)
        assert np.isfinite(maps["gain_dn_per_e"]).mean() > 0.95
        assert maps["single_pixel_dark_R"].size == noise.bias_frames
        assert maps["single_pixel_bright_R"].size == noise.frames
        # Histogram inputs retain individual ADC samples, not pair differences
        # or pooled pixel distributions.
        assert np.all(maps["single_pixel_dark_R"] % 1 == 0)


def test_saved_roi_is_original_16bit_raw_and_complete_phase_has_exact_frame_counts(
    tmp_path,
):
    backend = camera()
    originals = []
    original = backend.capture

    def capture(*args):
        frames, trigger = original(*args)
        originals.extend(f.pixels.copy() for f in frames)
        return frames, trigger

    backend.capture = capture
    noise = short_config()
    dark = run_noise_phase(backend, capture_config(tmp_path), noise, "dark")
    session, manifest = verify_phase(dark["path"], "dark")
    assert session["status"] == "awaiting_bright"
    record = manifest["stacks"][0]
    meta = json.loads((Path(dark["path"]) / record["metadata"]).read_text())
    saved = np.fromfile(Path(dark["path"]) / record["raw"], dtype="<u2").reshape(
        noise.bias_frames, 16, 16
    )
    x, y, w, h = meta["roi"]
    np.testing.assert_array_equal(
        saved,
        np.stack([p[y : y + h, x : x + w] for p in originals[: noise.bias_frames]]),
    )
    from rawrig.noise_analysis import _load

    loaded = _load(Path(dark["path"]), record)
    for channel, values in loaded["single_pixel_values"].items():
        column, row = loaded["single_pixel_roi_coordinates"][channel]
        np.testing.assert_array_equal(values, saved[:, row, column])
    values = saved.astype(float)
    pairs = (values[::2] - values[1::2]) ** 2 / 2
    for half in (0, 1):
        np.testing.assert_allclose(
            loaded["pair_half_variance"][half], pairs[half::2].mean(0)
        )
    assert backend.noise_phase is None and not backend.running
    assert not (Path(dark["path"]) / "COMPLETE.json").exists()


def test_full_sensor_analysis_and_full_raw_record_original_shape(tmp_path):
    backend = camera()
    noise = replace(short_config(), roi_size=0, save_full_raw=True)
    dark = run_noise_phase(backend, capture_config(tmp_path), noise, "dark")
    _, manifest = verify_phase(dark["path"], "dark")
    meta = json.loads(
        (Path(dark["path"]) / manifest["stacks"][0]["metadata"]).read_text()
    )
    assert meta["roi"] == [0, 0, 64, 64]
    assert meta["stored_shape"] == [64, 64]


@pytest.mark.parametrize("changed", ["config", "camera", "settings"])
def test_bright_stage_rejects_mismatched_reference_before_accepting_any_data(
    tmp_path, changed
):
    backend = camera()
    noise = short_config()
    cfg = capture_config(tmp_path)
    dark = run_noise_phase(backend, cfg, noise, "dark")
    if changed == "config":
        noise = replace(noise, frames=9)
    elif changed == "camera":
        backend.open(["DEMO_2"])
    else:
        original = backend._settings
        backend._settings = lambda i: {**original(i), "adc_bit_depth": "Bit10"}
    count = backend.counter
    with pytest.raises(CaptureError, match="다릅니다"):
        run_noise_phase(backend, cfg, noise, "bright", dark["path"])
    assert backend.counter == count
    assert verify_phase(dark["path"], "dark")[0]["status"] == "awaiting_bright"
    assert not (Path(dark["path"]) / "BRIGHT_COMPLETE.json").exists()


def test_cancel_bright_keeps_verified_dark_and_retry_uses_new_files(tmp_path):
    backend = camera()
    cfg, noise = capture_config(tmp_path), short_config()
    dark = run_noise_phase(backend, cfg, noise, "dark")
    event = Event()

    def cancel_after_a_frame(message, done, total):
        if done == 2:
            event.set()

    with pytest.raises(Cancelled):
        run_noise_phase(
            backend, cfg, noise, "bright", dark["path"], event, cancel_after_a_frame
        )
    assert verify_phase(dark["path"], "dark")[0]["status"] == "awaiting_bright"
    assert not (Path(dark["path"]) / "BRIGHT_COMPLETE.json").exists()
    assert backend.noise_phase is None and not backend.running
    run_noise_phase(backend, cfg, noise, "bright", dark["path"])
    assert verify_phase(dark["path"], "bright")[1]["phase"] == "bright"
    assert len([p for p in Path(dark["path"]).glob("bright_*") if p.is_dir()]) == 2


@pytest.mark.parametrize("kind", ["raw", "stats", "metadata", "plot"])
def test_checksum_tampering_is_detected_before_using_corrupt_data(
    measured, tmp_path, kind
):
    result, _ = measured
    root = tmp_path / "copy"
    shutil.copytree(result["path"], root)
    if kind == "plot":
        target = root / "analysis" / result["report"]["plots"][0]["path"]
    else:
        record = json.loads((root / "dark.json").read_text())["stacks"][0]
        target = root / record[kind]
    with target.open("ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(CaptureError, match="체크섬"):
        verify_noise_session(root)


def test_failed_gain_fit_never_invents_electron_noise_full_well_or_photon_snr():
    fit = fit_gain([1.0, 2.0, 3.0], [8.0, 1.0, 17.0], np.ones(3, dtype=bool))
    assert not fit["valid"] and fit["e_per_dn"] is None
    assert not fit_gain([2.0, 2.0, 2.0], [1.0, 2.0, 3.0], np.ones(3, dtype=bool))[
        "valid"
    ]


def test_completed_bright_retake_preserves_original_and_reuses_verified_dark(measured):
    from rawrig.noise import fork_bright_session
    from rawrig.storage import sha256_file

    result, noise = measured
    source = Path(result["path"])
    digest = sha256_file(source / "session.json")
    root = fork_bright_session(source)
    assert root != source and root.name.endswith(".partial")
    session, dark = verify_phase(root, "dark")
    assert session["bright_retake_from"]["path"] == str(source)
    assert session["status"] == "awaiting_bright"
    assert not (root / "BRIGHT_COMPLETE.json").exists()
    for record in dark["stacks"]:
        for kind in ("raw", "stats", "metadata"):
            assert sha256_file(root / record[kind]) == sha256_file(
                source / record[kind]
            )
    backend = camera()
    bright = run_noise_phase(
        backend, capture_config(source.parent.parent), noise, "bright", root
    )
    assert verify_phase(bright["path"], "bright")
    assert sha256_file(source / "session.json") == digest
    assert verify_noise_session(source)["passed"]
    backend.close()


def test_flicker_excludes_gain_estimation_and_unknown_qe_remains_unmeasured(
    measured, tmp_path
):
    result, noise = measured
    root = Path(result["path"])
    _, dark = verify_phase(root, "dark")
    _, bright = verify_phase(root, "bright")
    d = {(r["point"]["index"], r["serial"]): r for r in dark["stacks"]}
    b = {(r["point"]["index"], r["serial"]): r for r in bright["stacks"]}
    from unittest.mock import patch

    with patch("rawrig.noise_analysis._flicker", return_value=(0.5, 0.5)):
        report, maps = characterize(
            root,
            {"serial": "DEMO_1", "model": "test", "simulated": True},
            0,
            replace(noise, quantum_efficiency={}),
            d,
            b,
            Event(),
        )
    for channel in report["channels"].values():
        assert not channel["fit"]["valid"]
        assert channel["summary"]["read_noise_rms_e"] is None
        assert channel["summary"]["dynamic_range_db"] is None
        assert all(r["signal_photons"] is None for r in channel["curve"])
    assert not np.isfinite(maps["gain_dn_per_e"]).any()
    assert any("조명 변동" in warning for warning in report["warnings"])


def test_preflight_out_of_range_creates_no_session(tmp_path):
    backend = camera()
    noise = replace(short_config(), gains_db=[99.0])
    with pytest.raises(CaptureError, match="지원 범위"):
        run_noise_phase(backend, capture_config(tmp_path), noise, "dark")
    assert not (tmp_path / "noise_sessions").exists()


def test_locked_protocols_disallow_editing_and_distinguish_100_from_1000():
    curve = NoiseConfig.fixed_curve()
    assert curve.frames == 100 and curve.bias_frames == 1000
    assert curve.count("dark") == 2500 and curve.count("bright") == 1600
    fixed = NoiseConfig.fixed()
    assert fixed.count("dark") == fixed.count("bright") == 1000
    for cfg in (curve, fixed):
        for key, value in (("frames", 200), ("roi_size", 128), ("gains_db", [6.0])):
            with pytest.raises(CaptureError):
                NoiseConfig.from_dict({**asdict(cfg), key: value})


def test_automatic_curve_matches_dark_points_fits_known_poisson_gain_and_keeps_1000_bias(
    tmp_path,
):
    from rawrig.noise import supported_curve_plan

    backend = camera()
    noise = NoiseConfig.fixed_curve()
    limits = backend.limits()
    plan = supported_curve_plan(noise, [{**limits[0], "exposure_us": [250.0, 75000.0]}])
    assert (
        len(plan) == 16
        and plan[0]["exposure_us"] == 250
        and plan[-1]["exposure_us"] == 75000
    )
    dark = run_noise_phase(backend, capture_config(tmp_path), noise, "dark")
    _, d = verify_phase(dark["path"], "dark")
    assert (
        json.loads((Path(dark["path"]) / d["stacks"][0]["metadata"]).read_text())[
            "count"
        ]
        == 1000
    )
    assert all(
        json.loads((Path(dark["path"]) / r["metadata"]).read_text())["count"] == 100
        for r in d["stacks"][1:]
    )
    bright = run_noise_phase(
        backend, capture_config(tmp_path), noise, "bright", dark["path"]
    )
    session, b = verify_phase(bright["path"], "bright")
    assert (
        [r["point"] for r in d["stacks"]]
        == [r["point"] for r in b["stacks"]]
        == session["acquisition_plan"]
    )
    dr = {(r["point"]["index"], r["serial"]): r for r in d["stacks"]}
    br = {(r["point"]["index"], r["serial"]): r for r in b["stacks"]}
    result, _ = characterize(
        Path(bright["path"]),
        session["cameras"][0],
        0,
        noise,
        dr,
        br,
        Event(),
        acquisition_plan=session["acquisition_plan"],
    )
    for data in result["channels"].values():
        assert len(data["curve"]) == 16 and data["summary"]["dark_bias_frames"] == 1000
        assert data["fit"]["valid"]
        assert data["summary"]["gain_dn_per_e"] == pytest.approx(32, rel=0.10)
    backend.close()


def test_fixed_dark_only_analysis_never_fakes_ptc_and_bright_restores_saved_settings(
    tmp_path, monkeypatch
):
    from rawrig.noise_analysis import analyze_dark_session
    from rawrig.models import Preset

    backend = camera()
    cfg = capture_config(tmp_path)
    backend.prepare(Preset("original", 9999.0, 3.0), cfg)
    noise = NoiseConfig.fixed()
    dark = run_noise_phase(backend, cfg, noise, "dark")
    report = analyze_dark_session(dark["path"])["report"]
    assert len(report["plots"]) == 5
    assert report["cameras"][0]["settings"]["exposure_us"] == 9999
    for data in report["cameras"][0]["channels"].values():
        assert data["curve"] == [] and not data["fit"]["valid"]
        assert data["summary"]["gain_dn_per_e"] is None
        assert data["summary"]["read_noise_rms_e"] is None
        assert data["summary"]["dark_bias_frames"] == 1000
    backend.prepare(Preset("intervening", 1234.0, 0.0), cfg)
    bright = run_noise_phase(backend, cfg, noise, "bright", dark["path"])
    _, manifest = verify_phase(bright["path"], "bright")
    metadata = json.loads(
        (Path(bright["path"]) / manifest["stacks"][0]["metadata"]).read_text()
    )
    assert (
        metadata["settings"]["exposure_us"] == 9999
        and metadata["settings"]["gain_db"] == 3
    )
    result = analyze_noise_session(bright["path"])
    assert verify_noise_session(result["path"])["passed"]
    assert not any(p["path"].endswith("ptc_e.png") for p in result["report"]["plots"])
    assert len(result["report"]["plots"]) == 8
    for data in result["report"]["cameras"][0]["channels"].values():
        assert len(data["curve"]) == 1 and not data["fit"]["valid"]
        assert data["summary"]["gain_dn_per_e"] is None
    backend.close()
