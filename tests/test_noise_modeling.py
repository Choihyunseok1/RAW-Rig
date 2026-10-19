import numpy as np
import pytest
from rawrig.noise_modeling import weighted_gain, weighted_raw_noise


def test_heteroscedastic_fit_known_gain_and_independent_precision():
    rng = np.random.default_rng(613)
    shape = (15, 64, 64)
    signal = np.broadcast_to(np.geomspace(300, 30000, 15)[:, None, None], shape)
    gain = 5.85
    dark = np.full(shape, 190.0)
    estimates = []
    for _ in range(2):
        b = (gain * signal + dark) * rng.chisquare(500, size=shape) / 500
        d = dark * rng.chisquare(500, size=shape) / 500
        estimates.append(
            weighted_gain(signal, b, d, [500] * 15, [500] * 15, reference_gain=gain)
        )
    full = estimates[0]
    assert abs(np.mean(full["gain_dn_per_e"]) - gain) < 0.01
    empirical = np.std(
        estimates[0]["gain_dn_per_e"] - estimates[1]["gain_dn_per_e"]
    ) / np.sqrt(2)
    predicted = np.sqrt(np.mean(full["gain_standard_error_dn_per_e"] ** 2))
    assert 0.92 < empirical / predicted < 1.08
    assert np.mean(full["gain_standard_error_dn_per_e"]) / gain < 0.02


def test_insufficient_levels_are_nan_without_invented_values():
    s = np.ones((2, 3, 3)) * 1000
    d = np.ones_like(s) * 190
    r = weighted_gain(s, 5.9 * s + d, d, [100] * 2, [100] * 2)
    assert not r["valid"].any()
    assert np.isnan(r["gain_dn_per_e"]).all()
    with pytest.raises(ValueError):
        weighted_gain(s, s, d, [1] * 2, [100] * 2)


def test_raw_observation_model_retains_near_dark_levels_and_recovers_intercept():
    rng = np.random.default_rng(901)
    signal = np.broadcast_to(
        np.r_[np.geomspace(0.4, 20, 12), np.geomspace(80, 6000, 15)][:, None, None],
        (27, 64, 64),
    )
    pairs = np.r_[np.full(12, 1000), np.full(15, 500)]
    truth = 5.85 * signal + 192.5
    measured = (
        truth
        * rng.chisquare(pairs[:, None, None], size=truth.shape)
        / pairs[:, None, None]
    )
    fit = weighted_raw_noise(signal + 633, measured, pairs)
    assert fit["valid"].all()
    assert (fit["fit_points"] == 27).all()
    assert abs(fit["alpha_dn_per_e"].mean() - 5.85) < 0.01
    assert abs(fit["variance_at_reference_dn2"].mean() - 192.5) < 0.1
    assert (
        abs(
            np.std(fit["variance_at_reference_dn2"])
            / np.mean(fit["variance_at_reference_standard_error_dn2"])
            - 1
        )
        < 0.1
    )
