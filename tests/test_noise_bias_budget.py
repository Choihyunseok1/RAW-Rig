import numpy as np
from rawrig.illumination import pair_noise_bias_upper


def test_noise_variance_budget_rejects_contamination_without_rewriting_samples():
    rng = np.random.default_rng(2026)
    shape = (256, 256)
    dark = {'mean':np.full(shape, 100.0), 'pair_variance':np.full(shape, 100.0),
            'frame_means':100 + rng.normal(0, np.sqrt(100 / 65536), 1000)}
    bright = {'mean':np.full(shape, 1000.0), 'pair_variance':np.full(shape, 1000.0),
              'frame_means':1000 + rng.normal(0, np.sqrt(1000 / 65536), 100)}
    original = bright['frame_means'].copy()
    planes = {'R':np.s_[::2,::2], 'G1':np.s_[::2,1::2],
              'G2':np.s_[1::2,::2], 'B':np.s_[1::2,1::2]}
    assert pair_noise_bias_upper(bright, dark, planes) < .001
    np.testing.assert_array_equal(original, bright['frame_means'])
    bright['frame_means'] = original + np.tile([5.0,-5.0], 50)
    assert pair_noise_bias_upper(bright, dark, planes) > .001
