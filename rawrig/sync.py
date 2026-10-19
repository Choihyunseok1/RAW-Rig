from .models import RetryableCaptureError


def validate_ptp_snapshot(snapshot: list[dict], max_offset_us: float):
    """Require one time domain, stable port roles, and bounded slave offsets."""
    if not snapshot or any(s["status"] not in ("Master", "Slave") for s in snapshot):
        raise RetryableCaptureError("PTP Master/Slave 상태가 아직 안정되지 않았습니다.")
    if sum(s["status"] == "Master" for s in snapshot) > 1:
        raise RetryableCaptureError("PTP 마스터가 여러 개입니다. 같은 네트워크 시간 영역에 있지 않습니다.")
    domains = {s["parent_clock_id"] if s["status"] == "Slave" else s["clock_id"] for s in snapshot}
    if len(domains) != 1 or 0 in domains:
        raise RetryableCaptureError("카메라들이 같은 PTP 기준 시계를 사용하지 않습니다.")
    for s in snapshot:
        if s["status"] == "Slave" and abs(s["offset_ns"]) > max_offset_us * 1000:
            raise RetryableCaptureError(f"{s['serial']}: PTP offset {s['offset_ns']/1000:.3f} µs > {max_offset_us} µs")
    return True


def validate_ptp_frames(frames, target_ns, max_spread_us):
    timestamps = [f.camera_timestamp_ns for f in frames]
    spread = max(timestamps) - min(timestamps)
    if spread > max_spread_us * 1000:
        raise RetryableCaptureError(f"프레임 timestamp 시차 {spread/1000:.3f} µs가 허용값 {max_spread_us} µs를 초과했습니다.")
    for frame in frames:
        exposure_ns = frame.settings["exposure_us"] * 1000
        if frame.camera_timestamp_ns < target_ns - max_spread_us * 1000 or frame.camera_timestamp_ns > target_ns + exposure_ns + 100_000_000:
            raise RetryableCaptureError("예약 시각과 맞지 않는 프레임이 수신되었습니다.")
    return spread
