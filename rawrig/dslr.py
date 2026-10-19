"""Optional Nikon Z5 USB companion. USB PTP is Picture Transfer Protocol, not a clock."""
import io
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from threading import Event
import time

import numpy as np

from .models import CaptureError, check_cancel


def parse_widgets(output):
    widgets, current = {}, None
    for line in output.splitlines():
        if line.startswith("/"):
            current = {"path": line.strip(), "choices": []}
            widgets[line.strip().rsplit("/", 1)[-1]] = current
        elif current is not None and ":" in line:
            key, value = line.split(":", 1)
            value = value.strip()
            if key == "Choice":
                current["choices"].append(value.split(" ", 1)[1])
            else:
                current[key.lower().replace(" ", "_")] = value
    return widgets


def shutter_us(value):
    value = value.strip().replace('"', '').replace("s", "")
    if not re.fullmatch(r"\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)?", value):
        return None
    parts = value.split("/")
    result = float(parts[0]) / (float(parts[1]) if len(parts) == 2 else 1) * 1e6
    return result if result > 0 and math.isfinite(result) else None


def nearest_choice(choices, requested, parser, tolerance_ev):
    values = [(choice, parser(choice)) for choice in choices]
    values = [(c, v) for c, v in values if v is not None and v > 0]
    if not values:
        raise CaptureError("Z5에서 숫자로 해석 가능한 설정 선택지가 없습니다.")
    choice, actual = min(values, key=lambda pair: abs(math.log2(pair[1] / requested)))
    error = math.log2(actual / requested)
    if abs(error) > tolerance_ev + 1e-9:
        raise CaptureError(f"Z5 요청값 {requested:g}의 가장 가까운 지원값은 {choice} ({error:+.3f} EV)입니다. 설정 반올림 허용값을 확인하세요.")
    return choice, actual, error


def validate_nef(payload):
    if payload[:4] not in (b"II*\x00", b"MM\x00*"):
        raise CaptureError("Z5 파일이 TIFF 기반 NEF 형식이 아닙니다.")
    try:
        import rawpy
        import exifread
        tags = exifread.process_file(io.BytesIO(payload), details=False)
        model = str(tags.get("Image Model", ""))
        if model.strip().upper() != "NIKON Z 5":
            raise CaptureError(f"수신 NEF의 카메라 모델이 Z5와 다릅니다: {model}")
        exposure = tags.get("EXIF ExposureTime")
        iso = tags.get("EXIF ISOSpeedRatings")
        if exposure is None or iso is None:
            raise CaptureError("NEF 프레임의 실제 노출시간·ISO EXIF가 없습니다.")
        ratio = exposure.values[0]
        actual_us = float(ratio.num) / float(ratio.den) * 1e6
        actual_iso = float(iso.values[0])
        with rawpy.imread(io.BytesIO(payload)) as raw:
            pixels = raw.raw_image
            if raw.raw_pattern is None or pixels.ndim != 2 or pixels.dtype != np.uint16 or not pixels.size:
                raise CaptureError("Z5 NEF에 유효한 Bayer RAW 평면이 없습니다.")
            # Decode every sample, without demosaicing, white balance or RGB conversion.
            result = {"shape": list(pixels.shape), "dtype": str(pixels.dtype), "white_level": int(raw.white_level),
                "black_level_per_channel": list(raw.black_level_per_channel), "bayer_pattern": raw.raw_pattern.tolist(),
                "min": int(pixels.min()), "max": int(pixels.max()), "mean": float(pixels.mean()),
                "full_scale_fraction": float(np.count_nonzero(pixels >= raw.white_level) / pixels.size),
                "decoder": "LibRaw via rawpy; original NEF preserved", "status": "passed",
                "exif_model": model, "exposure_us": actual_us, "iso": actual_iso,
                "aperture": str(tags.get("EXIF FNumber", ""))}
            return result
    except ImportError as error:
        raise CaptureError("Z5 NEF 검증에는 rawpy·exifread가 필요합니다. pip install 'raw-rig[nikon]'으로 설치하세요.") from error
    except Exception as error:
        raise CaptureError(f"Z5 NEF RAW 디코딩 검사 실패: {error}") from error


class GPhoto:
    def __init__(self, port="", cancel=None):
        local = Path(__file__).resolve().parents[1] / ".runtime/gphoto/usr"
        self.executable = shutil.which("gphoto2") or str(local / "bin/gphoto2")
        if not Path(self.executable).is_file():
            raise CaptureError("gphoto2가 없습니다. 시스템의 gphoto2 패키지를 설치하세요.")
        self.env = {**os.environ, "LC_ALL": "C"}
        if str(local) in self.executable:
            self.env["LD_LIBRARY_PATH"] = str(local / "lib/x86_64-linux-gnu") + ":" + self.env.get("LD_LIBRARY_PATH", "")
        self.port = port
        self.cancel = cancel if cancel is not None else Event()

    def run(self, *args, timeout=20, cwd=None, ignore_cancel=False):
        command = [self.executable]
        if self.port:
            command += ["--port", self.port]
        command += list(args)
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=self.env, cwd=cwd)
        started = time.monotonic()
        try:
            while True:
                if not ignore_cancel:
                    check_cancel(self.cancel)
                if time.monotonic() - started > timeout:
                    raise CaptureError("Z5 USB 명령의 제한시간이 초과되었습니다.")
                try:
                    stdout, stderr = process.communicate(timeout=.1)
                    break
                except subprocess.TimeoutExpired:
                    continue
            if process.returncode:
                raise CaptureError("Z5 USB 통신 실패: " + stderr.decode(errors="replace").strip()[-1800:])
            return stdout.decode(errors="replace")
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.communicate(timeout=1)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.communicate()

    def detect(self):
        output = self.run("--auto-detect")
        return [{"model": m.group(1).strip(), "port": m.group(2)} for line in output.splitlines()
                if (m := re.match(r"(.+?)\s+(usb:\d+,\d+)\s*$", line))]

    def widgets(self):
        return parse_widgets(self.run("--list-all-config"))


def probe_nikon(config, cancel=None):
    api = GPhoto(config.dslr_port, cancel)
    found = [c for c in api.detect() if re.fullmatch(r"Nikon(?::|\s+)Z\s?5", c["model"], re.I)]
    if config.dslr_port:
        found = [c for c in found if c["port"] == config.dslr_port]
    if len(found) != 1:
        raise CaptureError(f"USB Nikon Z5가 정확히 1대 필요합니다. 발견: {len(found)}대. 전원·USB 데이터 케이블·다른 촬영 프로그램을 확인하세요.")
    api.port = found[0]["port"]
    widgets = api.widgets()
    return {"camera": found[0], "widgets": widgets, "usb_clock_sync": False,
            "note": "USB 촬영 명령 시각만 기록하며 노출 시작 시차는 측정하지 않습니다."}


def preview_nikon(config, cancel):
    report = probe_nikon(config, cancel)
    api = GPhoto(report["camera"]["port"], cancel)
    with tempfile.TemporaryDirectory(prefix="rawrig_nikon_preview_") as folder:
        api.run("--capture-preview", "--filename", "preview.jpg", cwd=folder, timeout=30)
        return (Path(folder) / "preview.jpg").read_bytes()


class NikonCompanion:
    def __init__(self, config, cancel):
        report = probe_nikon(config, cancel)
        self.report, self.widgets = report, report["widgets"]
        self.api = GPhoto(report["camera"]["port"], cancel)
        self.original = {}
        self.applied = {}
        self.restore_errors = []

    def key(self, *names, required=True):
        value = next((self.widgets[name] for name in names if name in self.widgets), None)
        if value is None and required:
            raise CaptureError(f"Z5가 {names[0]} USB 설정을 제공하지 않습니다.")
        return value

    def set(self, widget, value):
        if widget.get("readonly") == "1" and widget.get("current") != value:
            raise CaptureError(f"Z5의 {widget['path']} 설정이 잠겨 있습니다. 카메라를 사진 M 모드로 설정하세요.")
        self.original.setdefault(widget["path"], widget.get("current", ""))
        if widget.get("current") != value:
            self.api.run("--set-config-value", widget["path"] + "=" + value)
        readback = parse_widgets(widget["path"] + "\n" + self.api.run("--get-config", widget["path"]))
        actual = next(iter(readback.values()))["current"]
        if actual != value:
            raise CaptureError(f"Z5 설정 재확인 실패: {value} / {actual}")
        widget["current"] = actual
        self.applied[widget["path"]] = actual

    def preflight(self, plan, config):
        self.key("shutterspeed", "exptime")
        self.key("iso")
        self.key("imagequality")
        program = self.key("expprogram", "exposureprogram", required=False)
        if program and program.get("current", "").lower() not in ("m", "manual"):
            raise CaptureError("Z5 모드 다이얼을 M으로 설정하세요. 노출시간·ISO·조리개를 고정해 촬영합니다.")
        auto_iso = self.key("autoiso", "isoauto", required=False)
        if auto_iso:
            off = next((c for c in auto_iso["choices"] if c.lower() in ("off", "0", "disabled")), "0")
            self.set(auto_iso, off)
        quality = self.key("imagequality")
        raw = next((c for c in quality["choices"] if ("nef" in c.lower() or "raw" in c.lower()) and "+" not in c and "jpeg" not in c.lower()), None)
        if raw is None:
            raise CaptureError("Z5에서 RAW 단독 저장 선택지를 확인할 수 없습니다.")
        self.set(quality, raw)
        compression = self.key("rawcompression", required=False)
        if compression:
            lossless = next((c for c in compression["choices"] if "lossless" in c.lower() or "uncompressed" in c.lower()), None)
            if lossless is None:
                raise CaptureError("Z5에서 무손실 NEF 압축 설정을 확인할 수 없습니다.")
            self.set(compression, lossless)
        result = []
        for step in plan:
            serial = config.dslr_reference_serial or next(iter(step["applied"]))
            if serial not in step["applied"]:
                raise CaptureError("Z5가 따라갈 산업용 카메라가 연결되지 않았습니다.")
            baseline = step["applied"][serial]
            requested_us = baseline["exposure_us"] if config.dslr_follow else config.dslr_exposure_us
            requested_iso = config.dslr_iso * 10 ** (baseline["gain_db"] / 20) if config.dslr_follow else config.dslr_iso
            shutter, actual_us, shutter_ev = nearest_choice(self.key("shutterspeed", "exptime")["choices"], requested_us, shutter_us, config.dslr_max_quantization_ev)
            iso, actual_iso, iso_ev = nearest_choice(self.key("iso")["choices"], requested_iso,
                lambda c: float(c) if re.fullmatch(r"\d+", c) else None, config.dslr_max_quantization_ev)
            result.append({"shutter_choice": shutter, "iso_choice": iso, "exposure_us": actual_us, "iso": actual_iso,
                "requested_exposure_us": requested_us, "requested_iso": requested_iso,
                "shutter_rounding_ev": shutter_ev, "iso_rounding_ev": iso_ev, "follow_serial": serial if config.dslr_follow else None,
                "gain_to_iso": "nominal ISO_base * 10^(gain_dB/20); no radiometric equivalence" if config.dslr_follow else "fixed"})
        # Validate all writable settings before any dataset trigger.
        for setting in result:
            self.prepare(setting, config)
        return result

    def prepare(self, setting, config):
        self.set(self.key("shutterspeed", "exptime"), setting["shutter_choice"])
        self.set(self.key("iso"), setting["iso_choice"])
        for names, value in ((("f-number", "aperture"), config.dslr_aperture), (("whitebalance",), config.dslr_white_balance)):
            if value:
                self.set(self.key(*names), value)
        self.applied.update({w["path"]: w.get("current", "") for w in self.widgets.values()
                             if w["path"].rsplit("/", 1)[-1] in ("f-number", "aperture", "whitebalance", "serialnumber")})

    def capture(self, setting, config):
        if self.api.cancel.wait(max(0, config.dslr_dispatch_offset_ms) / 1000):
            check_cancel(self.api.cancel)
        started = time.monotonic_ns()
        with tempfile.TemporaryDirectory(prefix="rawrig_nef_") as folder:
            self.api.run("--capture-image-and-download", "--keep", "--filename", "capture.%C",
                         timeout=config.dslr_timeout_seconds, cwd=folder)
            paths = list(Path(folder).glob("*"))
            if len(paths) != 1 or paths[0].suffix.lower() != ".nef":
                raise CaptureError("Z5 RAW 단독 NEF 파일 1개가 수신되지 않았습니다.")
            payload = paths[0].read_bytes()
        decoded = validate_nef(payload)
        if abs(decoded["exposure_us"] - setting["exposure_us"]) > max(1., setting["exposure_us"] * .01) or decoded["iso"] != setting["iso"]:
            raise CaptureError("수신 NEF의 실제 노출시간·ISO가 요청과 다릅니다.")
        # Settings may have been overridden by the body (e.g. auto ISO); read them again after shooting.
        actual = self.api.widgets()
        for names, wanted in ((("shutterspeed", "exptime"), setting["shutter_choice"]), (("iso",), setting["iso_choice"])):
            widget = next((actual[n] for n in names if n in actual), None)
            if widget is None or widget.get("current") != wanted:
                raise CaptureError("Z5 촬영 후 노출시간·ISO 재확인이 실패했습니다.")
        return payload, {"camera": self.report["camera"], "settings": setting, "applied_settings": self.applied.copy(),
            "raw_validation": decoded, "original_format": "NEF", "timing": {"usb_command_started_monotonic_ns": started,
                "download_completed_monotonic_ns": time.monotonic_ns(), "dispatch_offset_ms": config.dslr_dispatch_offset_ms,
                "exposure_start_measured": False, "clock_comparable_to_gige_ptp": False}}

    def close(self):
        for path, value in reversed(list(self.original.items())):
            try:
                self.api.run("--set-config-value", path + "=" + value, ignore_cancel=True, timeout=5)
            except CaptureError as error:
                self.restore_errors.append(str(error))
        self.original.clear()
