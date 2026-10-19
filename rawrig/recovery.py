"""Bounded recovery before any group is committed to the dataset."""

from dataclasses import asdict
import time

from .models import (
    Cancelled,
    CaptureError,
    RetryableCaptureError,
    check_cancel,
    validate_frame,
)


def is_transient(error):
    # Spinnaker's documented IO, timeout and chunk-parse errors. Access denial,
    # unsupported formats, invalid settings and storage errors are not retried.
    seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        if isinstance(error, RetryableCaptureError) or getattr(error, "code", None) in (
            -1010,
            -1011,
            -1018,
            -2007,
        ):
            return True
        error = error.__cause__
    return False


def capture_checked(
    backend,
    presets,
    config,
    cancel,
    previous_ids=None,
    progress=None,
    capture_call=None,
):
    """Discard every camera's frame together, rearm and reacquire on transport failure.

    capture_call may also acquire a USB companion. It must finish all of its
    futures before raising; a successful companion from a rejected attempt is
    discarded too. No previously saved group is repeated or overwritten.
    """
    presets = (
        presets
        if isinstance(presets, dict)
        else {c.serial: presets for c in backend.cameras}
    )
    previous_ids = previous_ids if previous_ids is not None else {}
    progress = progress or (lambda _: None)
    timeout = max(
        config.timeout_ms,
        int(max(p.exposure_us for p in presets.values()) / 1000) + 2000,
    )
    capture_call = capture_call or (lambda: backend.capture(timeout, cancel))
    discarded = []
    for attempt in range(config.auto_retry_count + 1):
        check_cancel(cancel)
        try:
            result = capture_call()
            frames, trigger = result[:2]
            serials = [f.camera.serial for f in frames]
            if len(serials) != len(presets) or set(serials) != set(presets):
                raise CaptureError("촬영 그룹에 카메라 누락/중복이 있습니다.")
            for frame in frames:
                validate_frame(
                    frame,
                    presets[frame.camera.serial],
                    previous_ids.get(frame.camera.serial),
                )
            # Update only after all cameras have passed, never halfway through.
            previous_ids.update({f.camera.serial: f.frame_id for f in frames})
            if discarded:
                trigger["recovery"] = {
                    "attempts": attempt + 1,
                    "discarded_groups": discarded,
                    "presets_unchanged": {s: asdict(p) for s, p in presets.items()},
                }
                progress(
                    f"자동 복구 완료 · 전체 카메라 재촬영 {attempt}회 · RAW 검증 통과"
                )
            return result
        except Cancelled:
            raise
        except Exception as error:
            if not is_transient(error):
                raise
            if attempt == config.auto_retry_count:
                failure = RetryableCaptureError(
                    f"{error}\n자동 복구 {attempt}회 후에도 정상 프레임을 받지 못했습니다."
                )
                failure.recovery = {
                    "attempts": attempt + 1,
                    "discarded_groups": discarded
                    + [
                        {
                            "attempt": attempt + 1,
                            "reason": str(error),
                            "host_monotonic_ns": time.monotonic_ns(),
                        }
                    ],
                }
                raise failure from error
            discarded.append(
                {
                    "attempt": attempt + 1,
                    "reason": str(error),
                    "host_monotonic_ns": time.monotonic_ns(),
                }
            )
            progress(
                f"자동 복구 {attempt + 1}/{config.auto_retry_count} · 묶음 전체를 다시 준비합니다 · {error}"
            )
            # Drains/clears acquisition and scheduled actions before another
            # trigger. A failure to stop/rearm propagates; never trigger blindly.
            backend.stop()
            if cancel.wait(config.retry_delay_ms / 1000):
                check_cancel(cancel)
            phase_model = getattr(backend, "light_phase", None)
            phase_fraction = getattr(backend, "light_phase_fraction", None)
            backend.prepare(presets, config)
            if phase_model is not None and hasattr(backend, "set_light_phase"):
                backend.set_light_phase(phase_model, phase_fraction)
            if config.sync_mode == "ptp":
                backend.synchronize(config, cancel, progress)
            backend.start()
            previous_ids.clear()  # Frame IDs may reset on BeginAcquisition.
            if cancel.wait(config.settle_ms / 1000):
                check_cancel(cancel)
