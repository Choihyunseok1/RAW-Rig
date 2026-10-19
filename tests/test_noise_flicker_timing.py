from dataclasses import replace, asdict
import json
from types import SimpleNamespace

import numpy as np
import pytest

from rawrig.models import CaptureError
from rawrig.noise import NoiseConfig, supported_curve_plan
from rawrig.flicker import measured_period, measured_preview_exposure, calibration_path


def test_measured_120hz_period_removes_short_and_nonintegral_exposures():
    cfg = replace(NoiseConfig.fixed_curve(), anti_flicker_period_us=1e6 / 120)
    cfg.validate()
    for phase in ("dark", "bright"):
        plan = cfg.plan(phase)
        assert len(plan) == 12
        assert np.allclose(
            [p["exposure_us"] for p in plan], np.arange(1, 13) * 1e6 / 120
        )
    assert cfg.plan("dark") == cfg.plan("bright")
    assert cfg.count("dark") == 2100 and cfg.count("bright") == 1200
    # Legacy records without the new field still deserialize exactly.
    old = asdict(NoiseConfig.fixed_curve())
    old.pop("anti_flicker_period_us")
    assert NoiseConfig.from_dict(old) == NoiseConfig.fixed_curve()


def test_camera_bounds_do_not_clip_period_multiples_into_unsafe_times():
    cfg = replace(NoiseConfig.fixed_curve(), anti_flicker_period_us=1e6 / 120)
    limits = [{"exposure_us": [9000.0, 75000.0], "gain_db": [0.0, 24.0]}]
    plan = supported_curve_plan(cfg, limits)
    assert len(plan) == 8
    assert plan[0]["exposure_us"] == pytest.approx(2e6 / 120)
    assert plan[-1]["exposure_us"] == pytest.approx(75000.0)
    with pytest.raises(CaptureError, match="3개 미만"):
        supported_curve_plan(
            cfg, [{"exposure_us": [9000.0, 24000.0], "gain_db": [0.0, 24.0]}]
        )


def test_calibration_applies_only_to_verified_camera_set(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    path = calibration_path()
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"period_us": 1e6 / 120, "serials": ["26315393"]}))
    camera = SimpleNamespace(serial="26315393", simulated=False)
    assert measured_period([camera]) == pytest.approx(1e6 / 120)
    assert measured_period([SimpleNamespace(serial="new", simulated=False)]) == 0
    assert (
        measured_period([camera, SimpleNamespace(serial="new", simulated=False)]) == 0
    )
    assert measured_period([SimpleNamespace(serial="26315393", simulated=True)]) == 0
    path.write_text(
        json.dumps(
            {
                "period_us": 1e6 / 120,
                "serials": ["26315393"],
                "preview_exposure_us": 16662.0,
            }
        )
    )
    assert measured_preview_exposure([camera]) == 16662.0
    path.write_text(
        json.dumps(
            {
                "period_us": 1e6 / 120,
                "serials": ["26315393"],
                "preview_exposure_us": 10000.0,
            }
        )
    )
    assert measured_preview_exposure([camera]) is None
