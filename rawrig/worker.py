"""SDK work stays off the Qt UI thread, including metering and verification."""

from dataclasses import replace
import json
import math
from queue import Empty, Queue
from threading import Event
import time
import traceback

from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage

from .demo import DemoBackend
from .diagnostics import save_diagnostic
from .engine import run_capture
from .planning import resolve_plan, step_presets
from .models import (
    Cancelled,
    CaptureConfig,
    CaptureError,
    Preset,
    WhiteBalanceRestoreError,
    check_cancel,
)
from .network import setup_network, cancelled
from .preview import preview_rgb
from .recovery import capture_checked
from .recovery import is_transient
from .storage import verify_dataset


class CameraWorker(QThread):
    discovered = Signal(object)
    connected = Signal(object)
    images = Signal(object)
    progress = Signal(str, int, int)
    message = Signal(str)
    error = Signal(str)
    result = Signal(str)
    resolved = Signal(object, object)
    verified = Signal(object)
    network_result = Signal(object)
    nikon_result = Signal(object)
    nikon_image = Signal(object)
    white_balance_result = Signal(object)
    noise_result = Signal(object)
    aborted = Signal(str)
    idle = Signal(str)

    def __init__(self, demo=False):
        super().__init__()
        self.demo = demo
        self.commands = Queue()
        self.shutdown = Event()
        self.cancel = Event()
        self.backend = None
        self.preview_enabled = True
        self.preview_config = CaptureConfig(sync_mode="software", auto_reference=False)
        self.preview_preset = Preset("preview", 10000.0, 0.0)
        self.preview_active = False
        self.display_options = (1.0, 1.0, 0.0)
        self.preview_fps = 4.0
        self.last_frames = []

    def submit(self, command, payload=None):
        self.commands.put((command, payload))

    def _emit_images(self, frames):
        self.last_frames = list(frames)
        images = []
        for frame in frames:
            rgb = preview_rgb(frame, *self.display_options)
            image = QImage(
                rgb.data,
                rgb.shape[1],
                rgb.shape[0],
                rgb.strides[0],
                QImage.Format.Format_RGB888,
            ).copy()
            images.append((frame.camera, image, frame.settings, frame.frame_id))
        self.images.emit(images)

    def _stop_preview(self):
        if self.backend:
            self.backend.stop()
        self.preview_active = False

    def _preview_once(self):
        if not self.preview_active:
            self.backend.prepare(self.preview_preset, self.preview_config)
            self.backend.start()
            self.preview_active = True
            self.shutdown.wait(0.1)
        frames, _ = capture_checked(
            self.backend,
            self.preview_preset,
            self.preview_config,
            self.shutdown,
            progress=self.message.emit,
        )
        self._emit_images(frames)

    def _new_backend(self):
        self.last_frames = []
        if self.backend:
            self.backend.close()
            self.backend = None
        if self.demo:
            self.backend = DemoBackend()
        else:
            from .spinnaker import SpinnakerBackend

            self.backend = SpinnakerBackend()
        self.preview_active = False

    def _fit_preview(self):
        if not self.backend.cameras:
            return
        mapped = isinstance(self.preview_preset, dict)
        fallback = (
            next(iter(self.preview_preset.values())) if mapped else self.preview_preset
        )
        fitted = {}
        for limit in self.backend.limits():
            previous = (
                self.preview_preset.get(limit["serial"], fallback)
                if mapped
                else fallback
            )
            fitted[limit["serial"]] = replace(
                previous,
                exposure_us=min(
                    max(previous.exposure_us, limit["exposure_us"][0]),
                    limit["exposure_us"][1],
                ),
                gain_db=min(
                    max(previous.gain_db, limit["gain_db"][0]), limit["gain_db"][1]
                ),
            )
        values = {(p.exposure_us, p.gain_db) for p in fitted.values()}
        self.preview_preset = (
            fitted if mapped or len(values) > 1 else next(iter(fitted.values()))
        )

    def _record_error(self, command, error):
        path = save_diagnostic(
            "error",
            {
                "command": command,
                "error": str(error),
                "error_type": type(error).__name__,
                "traceback": traceback.format_exc(),
                "recovery": getattr(error, "recovery", None),
                "cameras": [camera.serial for camera in self.backend.cameras]
                if self.backend
                else [],
            },
        )
        if path:
            self.message.emit(f"진단 기록 자동 저장: {path}")

    def _handle(self, command, payload):
        if command == "scan":
            self._stop_preview()
            self.connected.emit([])
            self._new_backend()
            self.discovered.emit(self.backend.discover())
        elif command == "connect":
            self._stop_preview()
            self.connected.emit([])
            self._new_backend()
            cameras = self.backend.open(payload)
            self._fit_preview()
            self.connected.emit(cameras)
            self.message.emit(
                f"카메라 {len(self.backend.cameras)}대 연결 · Bayer16 RAW 미리보기 시작"
            )
        elif command == "network":
            previous = (
                [camera.serial for camera in self.backend.cameras]
                if self.backend
                else []
            )
            self._stop_preview()
            self.connected.emit([])
            self._new_backend()
            config, event = payload
            self.cancel = event

            def progress(message):
                self.progress.emit(message, 0, 1)

            report = setup_network(self.backend, config, event, progress, self.demo)
            cancelled(event)
            try:
                found = self.backend.discover()
            except CaptureError as error:
                found = []
                report.setdefault("warnings", []).append(
                    f"검사 후 목록 갱신 지연: {error}"
                )
            self.discovered.emit(found)
            ready = [row["serial"] for row in report["cameras"] if row["verified"]]
            serials = [serial for serial in previous if serial in ready]
            serials.extend(serial for serial in ready if serial not in serials)
            if serials:
                try:
                    cancelled(event)
                    report["connection_attempts"] = []
                    for attempt in range(config.auto_retry_count + 1):
                        cancelled(event)
                        try:
                            cameras = self.backend.open(
                                serials[: config.expected_cameras]
                            )
                            report["connection_attempts"].append(
                                {"attempt": attempt + 1, "passed": True}
                            )
                            break
                        except Cancelled:
                            raise
                        except Exception as error:
                            report["connection_attempts"].append(
                                {
                                    "attempt": attempt + 1,
                                    "passed": False,
                                    "error": str(error),
                                }
                            )
                            if (
                                not is_transient(error)
                                or attempt == config.auto_retry_count
                            ):
                                raise
                            progress(
                                f"검증된 카메라 자동 연결 재시도 {attempt + 1}/{config.auto_retry_count}"
                            )
                            event.wait(config.retry_delay_ms / 1000)
                    limits = self.backend.limits()
                    low_time = max(item["exposure_us"][0] for item in limits)
                    high_time = min(item["exposure_us"][1] for item in limits)
                    low_gain = max(item["gain_db"][0] for item in limits)
                    high_gain = min(item["gain_db"][1] for item in limits)
                    prior_preset = (
                        next(iter(self.preview_preset.values()))
                        if isinstance(self.preview_preset, dict)
                        else self.preview_preset
                    )
                    from .flicker import measured_preview_exposure

                    flicker_time = measured_preview_exposure(cameras)
                    if (
                        flicker_time is not None
                        and low_time <= flicker_time <= high_time
                    ):
                        prior_preset = replace(prior_preset, exposure_us=flicker_time)
                    if low_time > high_time or low_gain > high_gain:
                        self._fit_preview()
                        self.message.emit(
                            "공통 미리보기 범위가 없어 카메라별 지원 범위로 표시합니다. 촬영 조건은 계획 검사에서 별도로 확인합니다."
                        )
                    else:
                        self.preview_preset = replace(
                            prior_preset,
                            exposure_us=min(
                                max(prior_preset.exposure_us, low_time), high_time
                            ),
                            gain_db=min(max(prior_preset.gain_db, low_gain), high_gain),
                        )
                    cancelled(event)
                    self.connected.emit(cameras)
                except Cancelled:
                    self.backend.close()
                    raise
                except Exception as error:
                    self.backend.close()
                    report["passed"] = False
                    report["errors"].append(f"검증 후 자동 연결 실패: {error}")
            report["connected_serials"] = [
                camera.serial for camera in self.backend.cameras
            ]
            report["diagnostic_path"] = save_diagnostic("network", report)
            self.preview_config = replace(
                self.preview_config,
                auto_retry_count=config.auto_retry_count,
                retry_delay_ms=config.retry_delay_ms,
            )
            self.network_result.emit(report)
        elif command == "preview":
            self.preview_enabled = payload
            if not payload:
                self._stop_preview()
        elif command == "preview_setting":
            self._stop_preview()
            self.preview_preset, config = payload
            self.preview_config = replace(
                config, sync_mode="software", auto_reference=False
            )
            self.message.emit(
                "선택한 촬영 그룹의 노출시간·게인을 미리보기에 적용했습니다."
            )
        elif command == "meter":
            self._stop_preview()
            config, event = payload
            if not self.backend or not self.backend.cameras:
                raise CaptureError("기준 카메라를 연결하세요.")
            self.message.emit("자동 노출 컨트롤 기준(노출시간·게인) 측광 중…")
            self.backend.configure_profiles(config)
            plan, reference = resolve_plan(
                self.backend, config, event, self.message.emit
            )
            presets = [next(iter(step_presets(step).values())) for step in plan]
            self.preview_preset = step_presets(plan[0])
            self.resolved.emit(presets, reference)
            self.message.emit(f"측광 완료 · 촬영 그룹 {len(plan)}개")
        elif command == "white_balance":
            self._stop_preview()
            config, event = payload
            self.cancel = event
            config.validate()
            check_cancel(event)
            if not self.backend or not self.backend.cameras:
                raise CaptureError("WB 기준 카메라를 연결하세요.")
            self.message.emit(
                "기준 카메라 WB 1회 측정 중 · 완료 후 전체 미리보기에 고정 적용합니다…"
            )
            try:
                report = self.backend.white_balance_once(config, event)
                if not report.get("raw_restored") or report.get("camera_auto") != "Off":
                    raise WhiteBalanceRestoreError(
                        "WB 측정 후 자동 조절 종료·RAW 복원을 확인하지 못했습니다. 카메라를 다시 연결하세요."
                    )
            except WhiteBalanceRestoreError:
                try:
                    self.backend.close()
                except Exception as error:
                    self.message.emit(f"WB 복원 실패 후 장치 정리: {error}")
                finally:
                    self.backend = None
                    self.last_frames = []
                    self.connected.emit([])
                raise
            check_cancel(event)
            red, blue = report["red"], report["blue"]
            if not all(
                math.isfinite(value) and 0.1 <= value <= 8 for value in (red, blue)
            ):
                raise CaptureError(
                    "측정한 WB 값이 0.1~8 범위 밖입니다. 기존 WB 값은 유지합니다."
                )
            red, blue = round(red, 4), round(blue, 4)
            report = {
                **report,
                "measured_red": report["red"],
                "measured_blue": report["blue"],
                "red": red,
                "blue": blue,
                "applied_serials": [camera.serial for camera in self.backend.cameras],
                "application": "shared preview gains; stored Bayer pixels unchanged",
                "measured_at_ns": time.time_ns(),
            }
            self.display_options = (red, blue, self.display_options[2])
            self.white_balance_result.emit(report)
            if self.last_frames:
                self._emit_images(self.last_frames)
            self.message.emit(
                f"WB 고정 완료 · 기준 {report['reference_serial']} · R {red:.4f} / B {blue:.4f} · 전체 {len(self.backend.cameras)}대 적용"
            )
        elif command == "nikon_probe":
            from .dslr import probe_nikon

            config, event = payload
            self.nikon_result.emit(probe_nikon(config, event))
        elif command == "nikon_preview":
            from .dslr import preview_nikon

            config, event = payload
            image = QImage.fromData(preview_nikon(config, event))
            if image.isNull():
                raise CaptureError("Z5 USB 미리보기 이미지를 읽지 못했습니다.")
            self.nikon_image.emit(image)
        elif command == 'noise_reference_probe':
            from pathlib import Path
            from .storage import write_json
            self._stop_preview()
            config,path=payload
            try:
                self.backend.configure_profiles(config)
                self.backend.prepare(Preset('timestamp_probe',24.,0.),config)
                self.backend.start()
                result=self.backend.probe_timestamp_position(config.timeout_ms,self.shutdown)
                write_json(Path(path),{'passed':True,**result})
                self.message.emit(f"독립 조명 기준 전환 검증 통과 · timestamp {result['position']}")
            except Exception as error:
                write_json(Path(path),{'passed':False,'error':str(error)})
                self.message.emit(f'독립 조명 기준 전환 검증 실패: {error}')
            finally:
                self.backend.stop()
        elif command in ("noise_dark", "noise_bright"):
            from .noise import run_noise_phase
            from .noise_analysis import analyze_noise_session, analyze_dark_session

            self._stop_preview()
            config, noise, session_path, event = payload
            self.cancel = event
            phase = "dark" if command == "noise_dark" else "bright"
            result = run_noise_phase(
                self.backend,
                config,
                noise,
                phase,
                session_path,
                event,
                self.progress.emit,
                self._emit_images,
            )
            self.noise_result.emit(result)
            if phase == "dark":
                self.noise_result.emit(
                    analyze_dark_session(result["path"], event, self.progress.emit)
                )
            if phase == "bright":
                self.noise_result.emit(
                    analyze_noise_session(result["path"], event, self.progress.emit)
                )
        elif command == "noise_analyze":
            from .noise_analysis import analyze_noise_session, analyze_dark_session
            from pathlib import Path

            self._stop_preview()
            path, event = payload
            self.cancel = event
            analyze = (
                analyze_noise_session
                if (Path(path) / "BRIGHT_COMPLETE.json").is_file()
                else analyze_dark_session
            )
            self.noise_result.emit(analyze(path, event, self.progress.emit))
        elif command == "noise_compare":
            from .noise_analysis import compare_noise_sessions

            self._stop_preview()
            paths, output, event = payload
            self.cancel = event
            self.noise_result.emit(compare_noise_sessions(paths, output, event))
        elif command == "capture":
            self._stop_preview()
            config, event = payload
            self.cancel = event
            result = run_capture(
                self.backend,
                config,
                event,
                self.progress.emit,
                self._emit_images,
                self.resolved.emit,
            )
            manifest = json.loads((result / "manifest.json").read_text())
            resolved_config = CaptureConfig.from_dict(manifest["config"])
            self.preview_preset = (
                step_presets(manifest["capture_plan"][0])
                if manifest.get("capture_plan")
                else resolved_config.presets[0]
            )
            spread = [
                group["trigger"].get("image_timestamp_spread_ns")
                for group in manifest["groups"]
            ]
            spread = [value for value in spread if value is not None]
            if spread:
                self.message.emit(
                    f"프레임 timestamp 시차: 최대 {max(spread) / 1000:.3f} µs"
                )
            self.result.emit(str(result))
        elif command == "verify":
            report = verify_dataset(payload)
            self.verified.emit(report)
            self.message.emit(
                f"재검사 통과: RAW {report['frames']}장 · {report['path']}"
            )
        elif command == "disconnect":
            self.last_frames = []
            self._stop_preview()
            if self.backend:
                self.backend.close()
                self.backend = None
            self.connected.emit([])
            self.message.emit("카메라 연결 해제")

    def run(self):
        next_preview = 0
        try:
            while not self.shutdown.is_set():
                try:
                    command, payload = self.commands.get(timeout=0.04)
                except Empty:
                    if (
                        self.backend
                        and self.backend.cameras
                        and self.preview_enabled
                        and time.monotonic() >= next_preview
                    ):
                        started = time.monotonic()
                        try:
                            self._preview_once()
                        except Cancelled:
                            pass
                        except Exception as error:
                            self._record_error("preview", error)
                            self.preview_enabled = False
                            try:
                                self._stop_preview()
                            except Exception as stop_error:
                                self.message.emit(f"미리보기 정리 오류: {stop_error}")
                            self.error.emit(f"미리보기 중단: {error}")
                        next_preview = max(
                            time.monotonic(), started + 1 / self.preview_fps
                        )
                    continue
                try:
                    self._handle(command, payload)
                except Cancelled as error:
                    if getattr(error, "noise_session_path", None):
                        self.noise_result.emit(
                            {"phase": "incomplete", "path": error.noise_session_path}
                        )
                    self.message.emit(str(error))
                    self.aborted.emit(str(error))
                except Exception as error:
                    if getattr(error, "noise_session_path", None):
                        self.noise_result.emit(
                            {"phase": "incomplete", "path": error.noise_session_path}
                        )
                    self._record_error(command, error)
                    self.error.emit(str(error))
                finally:
                    self.idle.emit(command)
        finally:
            if self.backend:
                try:
                    self.backend.close()
                except Exception as error:
                    self.message.emit(f"장치 정리 오류: {error}")

    def request_shutdown(self):
        self.cancel.set()
        self.shutdown.set()
