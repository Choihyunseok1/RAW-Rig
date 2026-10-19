from dataclasses import asdict, replace
import pytest
from rawrig.models import CaptureError
from rawrig.noise import NoiseConfig


def test_fast_bright_preserves_completed_stacks_and_matched_dark():
    cfg = replace(
        NoiseConfig.precision(),
        frames=100,
        dark_frames=1000,
        bright_frame_counts={str(i): 1000 for i in range(4)},
    )
    restored = NoiseConfig.from_dict(asdict(cfg))
    assert restored.count("dark") == 32000
    assert restored.count("bright") == 6800
    assert restored.stack_count("bright", restored.plan()[3]) == 1000
    assert restored.stack_count("bright", restored.plan()[4]) == 100
    assert restored.stack_count("dark", restored.plan("dark")[4]) == 1000


@pytest.mark.parametrize("counts", [{"32": 100}, {"0": True}, {"00": 100}, {"1": 0}])
def test_invalid_stack_counts_do_not_authorize_capture(counts):
    with pytest.raises(CaptureError):
        replace(NoiseConfig.precision(), bright_frame_counts=counts).validate()


def test_high_signal_dark_counts_preserve_dense_low_light_bias():
    cfg = replace(
        NoiseConfig.precision(),
        bias_frames=10000,
        dark_frames=1000,
        dark_frame_counts={"0": 100, "31": 100},
    )
    restored = NoiseConfig.from_dict(asdict(cfg))
    restored.validate()
    assert restored.stack_count("dark", restored.plan("dark")[0]) == 10000
    assert restored.stack_count("dark", restored.plan("dark")[31]) == 100
    assert restored.count("dark") == 10000 + 30 * 1000 + 100


@pytest.mark.parametrize("counts", [{"32": 100}, {"0": True}, {"00": 100}, {"1": 0}])
def test_invalid_dark_counts_fail_before_capture(counts):
    with pytest.raises(CaptureError):
        replace(NoiseConfig.precision(), dark_frame_counts=counts).validate()
