from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from pathlib import Path
import re
from threading import Event
from typing import Callable, Protocol

import numpy as np


class CaptureError(RuntimeError):
    pass


class Cancelled(CaptureError):
    pass


class WhiteBalanceRestoreError(CaptureError):
    """The camera must be disconnected before any more RAW acquisition."""


class RetryableCaptureError(CaptureError):
    """A whole, uncommitted capture group may be acquired again."""


def check_cancel(cancel: Event) -> None:
    if cancel.is_set():
        raise Cancelled("촬영이 중지되었습니다. 미완성 데이터는 .partial 폴더에 남습니다.")


@dataclass(frozen=True)
class Preset:
    name: str
    exposure_us: float
    gain_db: float

    def validate(self) -> None:
        if not self.name or not math.isfinite(self.exposure_us) or self.exposure_us <= 0:
            raise CaptureError("노출시간은 0보다 큰 유한한 값이어야 합니다.")
        if not math.isfinite(self.gain_db) or self.gain_db < 0:
            raise CaptureError("게인은 0 이상의 유한한 값이어야 합니다.")


def bracket_presets(low_us=1000.0, high_us=4000.0, low_db=0.0, high_db=12.0):
    return [Preset("01_low_exp_low_gain", low_us, low_db),
            Preset("02_high_exp_low_gain", high_us, low_db),
            Preset("03_low_exp_high_gain", low_us, high_db),
            Preset("04_high_exp_high_gain", high_us, high_db)]


@dataclass
class CaptureConfig:
    output_dir: str = "datasets"
    presets: list[Preset] = field(default_factory=bracket_presets)
    repeats: int = 1
    expected_cameras: int = 3
    sync_mode: str = "ptp"
    master_serial: str = ""
    output_line: str = "Line2"
    input_line: str = "Line0"
    trigger_edge: str = "RisingEdge"
    settle_ms: int = 100
    warmup_frames: int = 1
    timeout_ms: int = 3000
    preview_red: float = 1.0
    preview_blue: float = 1.0
    preview_brightness_ev: float = 0.0
    preview_fps: float = 4.0
    white_balance_reference_serial: str = ""
    white_balance_timeout_seconds: float = 10.0
    white_balance_profile: str = "Indoor"
    white_balance_calibration: dict = field(default_factory=dict)
    noise_config: dict = field(default_factory=dict)
    auto_reference: bool = True
    reference_serial: str = ""
    meter_seconds: float = 4.0
    low_exposure_factor: float = 1.0
    high_exposure_factor: float = 4.0
    low_gain_offset_db: float = 0.0
    high_gain_offset_db: float = 12.0
    ptp_wait_seconds: float = 45.0
    ptp_max_offset_us: float = 10.0
    ptp_max_timestamp_spread_us: float = 100.0
    action_lead_ms: int = 200
    network_interface: str = ""
    network_probe_seconds: float = 4.0
    network_rediscover_seconds: float = 12.0
    auto_retry_count: int = 2
    retry_delay_ms: int = 400
    capture_strategy: str = "bracket"
    cross_bracket: bool = False
    range_policy: str = "strict"
    meter_target_percent: float = 20.0
    meter_max_exposure_us: float = 100000.0
    meter_max_gain_db: float = 24.0
    camera_profiles: dict = field(default_factory=dict)
    dslr_enabled: bool = False
    dslr_port: str = ""
    dslr_follow: bool = False
    dslr_reference_serial: str = ""
    dslr_exposure_us: float = 20000.0
    dslr_iso: int = 100
    dslr_aperture: str = ""
    dslr_white_balance: str = ""
    dslr_max_quantization_ev: float = 0.2
    dslr_dispatch_offset_ms: int = 0
    dslr_timeout_seconds: float = 90.0

    def validate(self):
        if not isinstance(self.noise_config, dict):
            raise CaptureError("노이즈 측정 설정은 객체여야 합니다.")
        if len(self.presets) != 4:
            raise CaptureError("촬영 설정은 정확히 4개여야 합니다.")
        for preset in self.presets:
            preset.validate()
        if not 1 <= self.repeats <= 100 or not 1 <= self.expected_cameras <= 3:
            raise CaptureError("설정별 촬영 수는 1~100, 카메라 수는 1~3이어야 합니다.")
        if self.sync_mode not in ("software", "hardware", "ptp"):
            raise CaptureError("지원하지 않는 동기 방식입니다.")
        if self.trigger_edge not in ("RisingEdge", "FallingEdge"):
            raise CaptureError("트리거 에지는 RisingEdge 또는 FallingEdge여야 합니다.")
        if not 0 <= self.settle_ms <= 10000 or not 0 <= self.warmup_frames <= 10:
            raise CaptureError("설정 안정화 시간/워밍업 프레임 값이 올바르지 않습니다.")
        if not 100 <= self.timeout_ms <= 120000:
            raise CaptureError("수신 제한시간은 100~120000 ms여야 합니다.")
        if not all(math.isfinite(v) and 0.1 <= v <= 8 for v in (self.preview_red, self.preview_blue)):
            raise CaptureError("미리보기 화이트밸런스 값은 0.1~8이어야 합니다.")
        if not math.isfinite(self.preview_brightness_ev) or not -4 <= self.preview_brightness_ev <= 8:
            raise CaptureError("미리보기 밝기는 -4~8 EV여야 합니다.")
        if not math.isfinite(self.preview_fps) or not .5 <= self.preview_fps <= 20:
            raise CaptureError("미리보기 갱신 속도는 0.5~20 FPS여야 합니다.")
        if (not math.isfinite(self.white_balance_timeout_seconds) or not 1 <= self.white_balance_timeout_seconds <= 60
                or self.white_balance_profile not in ("Indoor", "Outdoor")):
            raise CaptureError("WB 측정 제한시간은 1~60초, 조명 프로필은 Indoor 또는 Outdoor여야 합니다.")
        if not isinstance(self.white_balance_reference_serial, str) or (self.white_balance_reference_serial
                and not re.fullmatch(r"[A-Za-z0-9_-]+", self.white_balance_reference_serial)):
            raise CaptureError("WB 기준 카메라 시리얼이 올바르지 않습니다.")
        if not isinstance(self.white_balance_calibration, dict):
            raise CaptureError("WB 측정 기록이 올바르지 않습니다.")
        if not .5 <= self.meter_seconds <= 30:
            raise CaptureError("자동 측광 시간은 0.5~30초여야 합니다.")
        if not (0 < self.low_exposure_factor < self.high_exposure_factor <= 100):
            raise CaptureError("노출시간 배수는 0 < 기준값 < 증가값 ≤ 100이어야 합니다.")
        if not all(math.isfinite(v) for v in (self.low_gain_offset_db, self.high_gain_offset_db)) or self.low_gain_offset_db >= self.high_gain_offset_db:
            raise CaptureError("낮은 게인 오프셋은 높은 게인 오프셋보다 작아야 합니다.")
        if not 1 <= self.ptp_wait_seconds <= 180 or not 50 <= self.action_lead_ms <= 5000:
            raise CaptureError("PTP 대기/예약 선행 시간 값이 올바르지 않습니다.")
        if not all(math.isfinite(v) and v > 0 for v in (self.ptp_max_offset_us, self.ptp_max_timestamp_spread_us)):
            raise CaptureError("PTP 오차 허용값은 0보다 커야 합니다.")
        if not 1 <= self.network_probe_seconds <= 10 or not 2 <= self.network_rediscover_seconds <= 60:
            raise CaptureError("주소 충돌 검사 시간은 1~10초, 재검색 제한시간은 2~60초여야 합니다.")
        if type(self.auto_retry_count) is not int or not 0 <= self.auto_retry_count <= 5 or type(self.retry_delay_ms) is not int or not 0 <= self.retry_delay_ms <= 5000:
            raise CaptureError("자동 복구 횟수는 0~5회, 재시도 간격은 0~5000 ms여야 합니다.")
        if self.capture_strategy not in ("bracket", "cross_reference", "per_camera") or self.range_policy not in ("strict", "clip"):
            raise CaptureError("촬영 모드 또는 설정 범위 처리 방식이 올바르지 않습니다.")
        if not (1 <= self.meter_target_percent <= 90 and 15 <= self.meter_max_exposure_us <= 30000000
                and 0 <= self.meter_max_gain_db <= 48):
            raise CaptureError("자동 측광 목표 밝기·최대 노출시간·최대 게인을 확인하세요.")
        if not isinstance(self.camera_profiles, dict):
            raise CaptureError("카메라 특성 설정이 올바르지 않습니다.")
        for serial, profile in self.camera_profiles.items():
            if not re.fullmatch(r"[A-Za-z0-9_-]+", serial) or not isinstance(profile, dict):
                raise CaptureError("카메라 특성의 시리얼 또는 항목이 올바르지 않습니다.")
            if profile.get("gain_conversion_mode", "preserve") not in ("preserve", "Low", "High") or profile.get("raw_mode", "unknown") not in ("unknown", "linear", "hdr"):
                raise CaptureError("변환 게인 또는 RAW 모드가 올바르지 않습니다.")
            if 'black_level_clamping' in profile and type(profile['black_level_clamping']) is not bool:
                raise CaptureError('블랙 레벨 클램핑 설정은 true/false여야 합니다.')
            if 'black_level_percent' in profile and (isinstance(profile['black_level_percent'], bool) or not isinstance(profile['black_level_percent'], (int,float)) or not math.isfinite(profile['black_level_percent'])):
                raise CaptureError('블랙 레벨은 유한한 숫자여야 합니다.')
            if 'link_throughput_bytes_per_second' in profile and (type(profile['link_throughput_bytes_per_second']) is not int or profile['link_throughput_bytes_per_second'] <= 0):
                raise CaptureError('전송 대역폭은 양의 정수여야 합니다.')
        if not (0 < self.dslr_exposure_us <= 30000000 and 25 <= self.dslr_iso <= 204800
                and 0 <= self.dslr_max_quantization_ev <= 1 and -5000 <= self.dslr_dispatch_offset_ms <= 5000
                and 10 <= self.dslr_timeout_seconds <= 180):
            raise CaptureError("Z5의 노출시간·ISO·설정 반올림 허용값·촬영 제한시간을 확인하세요.")
        if not all(math.isfinite(v) for v in (self.meter_target_percent, self.meter_max_exposure_us,
                self.meter_max_gain_db, self.dslr_exposure_us, self.dslr_max_quantization_ev, self.dslr_timeout_seconds)):
            raise CaptureError("설정에는 유한한 수만 사용할 수 있습니다.")
        Path(self.output_dir).expanduser()

    @property
    def planned_groups(self):
        if self.capture_strategy == "cross_reference":
            return self.expected_cameras * (4 if self.cross_bracket else 1)
        return 1 if self.capture_strategy == "per_camera" else 4

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        if "presets" in value:
            value["presets"] = [Preset(**p) for p in value["presets"]]
        result = cls(**value)
        result.validate()
        return result


@dataclass(frozen=True)
class CameraInfo:
    serial: str
    model: str
    simulated: bool = False

    def __post_init__(self):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", self.serial):
            raise CaptureError(f"안전하지 않은 카메라 시리얼: {self.serial!r}")


@dataclass
class Frame:
    camera: CameraInfo
    pixels: np.ndarray
    pixel_format: str
    frame_id: int
    camera_timestamp_ns: int
    host_received_ns: int
    settings: dict
    chunk: dict = field(default_factory=dict)
    transport: dict = field(default_factory=dict)


RAW_FORMATS = {f"Bayer{pattern}{bits}" for pattern in ("RG", "GR", "GB", "BG") for bits in (8, 16)}


def validate_frame(frame: Frame, preset: Preset, previous_id: int | None = None):
    pixels = frame.pixels
    if frame.pixel_format not in RAW_FORMATS or pixels.ndim != 2:
        raise CaptureError("Bayer RAW 단일 평면만 저장할 수 있습니다. RGB/Mono/packed는 거부합니다.")
    bits = 16 if frame.pixel_format.endswith("16") else 8
    if pixels.dtype.kind != "u" or pixels.dtype.itemsize != bits // 8 or not pixels.size:
        raise CaptureError("RAW 배열의 크기 또는 자료형이 올바르지 않습니다.")
    if pixels.shape != (frame.settings["height"], frame.settings["width"]):
        raise CaptureError(f"{frame.camera.serial}: 설정 해상도와 프레임 크기가 다릅니다.")
    if frame.transport.get("incomplete", False):
        raise RetryableCaptureError(f"{frame.camera.serial}: 불완전 프레임입니다.")
    if frame.transport.get("crc_passed") is False:
        raise RetryableCaptureError(f"{frame.camera.serial}: 카메라 CRC 검사 실패.")
    if previous_id is not None and frame.frame_id <= previous_id:
        raise RetryableCaptureError(f"{frame.camera.serial}: 중복/역순 프레임입니다.")
    for name, requested, tolerance in (("exposure_us", preset.exposure_us, max(1.0, preset.exposure_us * .005)),
                                        ("gain_db", preset.gain_db, .05)):
        actual = frame.settings.get(name)
        if actual is None or not math.isfinite(actual) or abs(actual - requested) > tolerance:
            raise CaptureError(f"{frame.camera.serial}: 요청한 {name}과 실제 적용값이 다릅니다: {requested} / {actual}")
        chunk = frame.chunk.get(name)
        if chunk is not None and (not math.isfinite(chunk) or abs(chunk - actual) > tolerance):
            raise RetryableCaptureError(f"{frame.camera.serial}: 프레임 chunk {name}이 설정과 다릅니다.")


class CameraBackend(Protocol):
    cameras: list[CameraInfo]

    def discover(self) -> list[CameraInfo]: ...
    def open(self, serials: list[str]) -> list[CameraInfo]: ...
    def prepare(self, preset: Preset, config: CaptureConfig) -> list[dict]: ...
    def start(self): ...
    def stop(self): ...
    def capture(self, timeout_ms: int, cancel: Event) -> tuple[list[Frame], dict]: ...
    def close(self): ...
    def limits(self) -> list[dict]: ...
    def meter(self, config: CaptureConfig, cancel: Event) -> dict: ...
    def white_balance_once(self, config: CaptureConfig, cancel: Event) -> dict: ...
    def synchronize(self, config: CaptureConfig, cancel: Event, progress: Callable) -> dict: ...
