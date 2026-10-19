"""Heteroscedastic PTC fits with explicit per-pixel sampling uncertainty.

No RAW correction, smoothing or residual-dependent selection is performed.
The reference gain is from an independent pilot, not each noisy variance point.
"""

import numpy as np


def weighted_raw_noise(
    raw_mean,
    pair_variance,
    pairs,
    reference_offset=633.0,
    reference_gain=5.9,
    reference_read_variance=190.0,
    white_level=65520,
):
    """Fit Var(RAW)=a*E(RAW)+b without identifying offset vs read variance.

    This observation model also covers low light where old-dark subtraction
    can dominate the tiny signal. The intercept is not physical read variance.
    Approximate analytic SE is checked using independent whole frame pairs.
    """
    mean = np.asarray(raw_mean, dtype=float)
    variance = np.asarray(pair_variance, dtype=float)
    if mean.shape != variance.shape:
        raise ValueError("Mean and variance stacks must have identical shapes")
    n = np.asarray(pairs, dtype=float).reshape((-1,) + (1,) * (mean.ndim - 1))
    if np.any(n < 2):
        raise ValueError("At least two pairs per level are required")
    x = mean - reference_offset
    expected = reference_gain * np.maximum(x, 0) + reference_read_variance
    uncertainty2 = (
        2 * expected**2 + reference_gain**2 * reference_read_variance / 2
    ) / n
    valid = (
        np.isfinite(mean)
        & np.isfinite(variance)
        & (variance >= 0)
        & (mean < white_level * 0.8)
    )
    w = np.where(valid, 1 / uncertainty2, 0)
    sx = np.sum(np.where(valid, w * x, 0), axis=0)
    sw = w.sum(axis=0)
    sxx = np.sum(np.where(valid, w * x * x, 0), axis=0)
    sy = np.sum(np.where(valid, w * variance, 0), axis=0)
    sxy = np.sum(np.where(valid, w * x * variance, 0), axis=0)
    denominator = sw * sxx - sx * sx
    count = valid.sum(axis=0)
    good = (count >= 4) & (denominator > 0)
    slope = np.divide(
        sw * sxy - sx * sy, denominator, out=np.full(sw.shape, np.nan), where=good
    )
    at_reference = np.divide(
        sxx * sy - sx * sxy, denominator, out=np.full(sw.shape, np.nan), where=good
    )
    slope_var = np.divide(sw, denominator, out=np.full(sw.shape, np.nan), where=good)
    reference_var = np.divide(
        sxx, denominator, out=np.full(sw.shape, np.nan), where=good
    )
    covariance = np.divide(-sx, denominator, out=np.full(sw.shape, np.nan), where=good)
    return dict(
        alpha_dn_per_e=slope,
        intercept_raw_dn2=at_reference - slope * reference_offset,
        variance_at_reference_dn2=at_reference,
        reference_offset_dn=np.broadcast_to(reference_offset, sw.shape),
        alpha_standard_error_dn_per_e=np.sqrt(slope_var),
        variance_at_reference_standard_error_dn2=np.sqrt(reference_var),
        alpha_reference_variance_covariance=covariance,
        valid=good,
        fit_points=count,
    )


def weighted_gain(
    signal,
    bright_variance,
    dark_variance,
    bright_pairs,
    dark_pairs,
    reference_gain=5.9,
    white_level=65520,
):
    signal = np.asarray(signal, dtype=float)
    b = np.asarray(bright_variance, dtype=float)
    d = np.asarray(dark_variance, dtype=float)
    nb = np.asarray(bright_pairs, dtype=float).reshape((-1,) + (1,) * (signal.ndim - 1))
    nd = np.asarray(dark_pairs, dtype=float).reshape((-1,) + (1,) * (signal.ndim - 1))
    if signal.shape != b.shape or b.shape != d.shape:
        raise ValueError("Signal and variance stacks must have identical shapes")
    if np.any(nb < 2) or np.any(nd < 2):
        raise ValueError("At least two independent pairs per level are required")
    expected = reference_gain * np.maximum(signal, 0) + d
    # Difference-of-pair-variance uncertainty plus finite mean uncertainty.
    uncertainty2 = 2 * expected**2 / nb + 2 * d**2 / nd
    uncertainty2 += reference_gain**2 * (expected / (2 * nb) + d / (2 * nd))
    valid = np.isfinite(signal) & np.isfinite(b) & np.isfinite(d)
    valid &= (signal > 3 * np.sqrt(np.maximum(d, 0))) & (signal < white_level * 0.75)
    valid &= (uncertainty2 > 0) & (d >= 0)
    w = np.where(valid, 1 / np.maximum(uncertainty2, 1e-300), 0)
    denominator = np.sum(np.where(valid, w * signal**2, 0), axis=0)
    numerator = np.sum(np.where(valid, w * signal * (b - d), 0), axis=0)
    count = valid.sum(axis=0)
    good = (count >= 3) & (denominator > 0)
    gain = np.divide(
        numerator, denominator, out=np.full(denominator.shape, np.nan), where=good
    )
    se = np.sqrt(
        np.divide(1, denominator, out=np.full(denominator.shape, np.nan), where=good)
    )
    residual = b - d - signal * gain
    chi2 = np.sum(np.where(valid, w * residual**2, 0), axis=0)
    reduced = np.divide(chi2, count - 1, out=np.full(gain.shape, np.nan), where=good)
    return dict(
        gain_dn_per_e=gain,
        gain_standard_error_dn_per_e=se,
        fit_points=count,
        reduced_chi_squared=reduced,
        valid=good,
    )
