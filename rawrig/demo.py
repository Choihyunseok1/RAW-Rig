from __future__ import annotations

import time

import numpy as np

from .models import CameraInfo, CaptureConfig, CaptureError, Frame, Preset, check_cancel


class DemoBackend:
    """Synthetic Bayer data. Every file and camera is explicitly marked simulated."""

    def __init__(self):
        self.cameras = []
        self.running = False
        self.preset = Preset("preview", 10000., 0.)
        self.config = CaptureConfig()
        self.counter = 0
        self.rng = np.random.default_rng(42)
        self.noise_phase = None

    def discover(self):
        return [CameraInfo(f"DEMO_{i+1}", f"가상 Bayer 카메라 {i+1}", True) for i in range(3)]

    def open(self, serials):
        available = {c.serial: c for c in self.discover()}
        if len(set(serials)) != len(serials) or not serials:
            raise CaptureError("카메라 선택이 비어 있거나 중복되었습니다.")
        self.cameras = [available[s] for s in serials]
        return self.cameras

    def limits(self):
        return [{"serial": c.serial, "exposure_us": [15., 30000000.], "gain_db": [0., 48.]} for c in self.cameras]

    def operating_presets(self):
        return {c.serial: self.preset[c.serial] if isinstance(self.preset, dict) else self.preset for c in self.cameras}

    def _settings(self, index):
        preset = self.preset[self.cameras[index].serial] if isinstance(self.preset, dict) else self.preset
        return {"serial": self.cameras[index].serial, "exposure_us": preset.exposure_us,
            "gain_db": preset.gain_db, "width": 640 + index * 32, "height": 480,
            "pixel_format": "BayerRG16", "pixel_max": 65535, "adc_bit_depth": "Bit12",
            "sensor_white_level": 65520,
            "bit_alignment": "12 significant bits shifted left 4 (demo)", "raw_processing": "disabled (demo)",
            "simulated": True}

    def prepare(self, preset, config):
        self.stop()
        presets = list(preset.values()) if isinstance(preset, dict) else [preset]
        for value in presets:
            value.validate()
        if config.sync_mode == "hardware" and config.master_serial not in {c.serial for c in self.cameras}:
            raise CaptureError("마스터 카메라를 선택하세요.")
        if any(not 15 <= p.exposure_us <= 30000000 or not 0 <= p.gain_db <= 48 for p in presets):
            raise CaptureError("가상 카메라 설정 범위 밖입니다.")
        self.preset, self.config = preset, config
        return [self._settings(i) for i in range(len(self.cameras))]

    def start(self):
        self.running = True

    def stop(self):
        self.running = False

    def capture(self, timeout_ms, cancel):
        if not self.running:
            raise CaptureError("카메라가 시작되지 않았습니다.")
        check_cancel(cancel)
        self.counter += 1
        frames = []
        for i, camera in enumerate(self.cameras):
            settings = self._settings(i)
            h, w = settings["height"], settings["width"]
            y, x = np.mgrid[:h, :w]
            if self.noise_phase:
                # Static flat field with known Poisson shot noise, read noise,
                # offset and pixel conversion variation. Never actual sensor data.
                gain = 2. * 10**(settings["gain_db"] / 20)
                prnu = 1 + .003 * np.sin(x * .43) * np.cos(y * .37)
                photons = np.full((h, w), settings["exposure_us"] / 100000 * 2500 * (1 + i * .08))
                if self.noise_phase == "dark":
                    photons[:] = settings["exposure_us"] / 1e6 * .2
                else:
                    photons[0::2, 0::2] *= .85
                    photons[1::2, 1::2] *= .65
                photoelectrons = self.rng.poisson(photons * prnu)
                read = self.rng.normal(0, 3., (h, w))
                offset = 64 + .3 * np.sin(x * .7)
                pixels = (np.clip(offset + gain * (photoelectrons + read), 0, 4095).astype(np.uint16) << 4)
                frames.append(Frame(camera, pixels, "BayerRG16", self.counter,
                    self.counter * 100000000 + i * 1000, time.monotonic_ns(), settings,
                    {"exposure_us": settings["exposure_us"], "gain_db": settings["gain_db"]},
                    {"incomplete": False, "crc_passed": None, "crc_status": "not_available (demo)"}))
                continue
            scene = (0.15 + .28 * x / w + .12 * y / h)
            circle = (x - w * (.45 + .08 * np.sin(self.counter / 25)))**2 + (y - h * .5)**2 < (h * .22)**2
            scene[circle] = .65
            signal = scene * (settings["exposure_us"] / 10000) * 10**(settings["gain_db"] / 20)
            signal[0::2, 0::2] *= .85
            signal[1::2, 1::2] *= .65
            signal += self.rng.normal(0, .002 * 10**(settings["gain_db"] / 20), (h, w))
            pixels = (np.clip(signal, 0, 1) * 4095).astype(np.uint16) << 4
            frames.append(Frame(camera, pixels, "BayerRG16", self.counter,
                self.counter * 100000000 + (i * 1000 if self.config.sync_mode == "ptp" else i * 100000000000), time.monotonic_ns(), settings,
                {"exposure_us": settings["exposure_us"], "gain_db": settings["gain_db"]},
                {"incomplete": False, "crc_passed": None, "crc_status": "not_available (demo)"}))
        return frames, {"mode": self.config.sync_mode, "simulated": True, "software_dispatch_span_ns": 0,
                        "exposure_skew_measured": False, "image_timestamp_spread_ns": (len(frames)-1)*1000 if self.config.sync_mode == "ptp" else None}

    def synchronize(self, config, cancel, progress):
        check_cancel(cancel)
        return {"mode": config.sync_mode, "simulated": True, "locked": config.sync_mode == "ptp"}

    def meter(self, config, cancel):
        self.stop()
        cancel.wait(min(config.meter_seconds, .4))
        check_cancel(cancel)
        serial = config.reference_serial or self.cameras[0].serial
        index = next(i for i, camera in enumerate(self.cameras) if camera.serial == serial)
        return {"reference_serial": serial,
                "exposure_us": 12000. * (index + 1), "gain_db": 3. * (index + 1), "stable": True, "simulated": True,
                "method": "synthetic continuous-auto reference"}

    def configure_profiles(self, config):
        return [{"serial": c.serial, "model": c.model, "declared": config.camera_profiles.get(c.serial, {}),
                 "simulated": True, "limits": limit} for c, limit in zip(self.cameras, self.limits())]

    def white_balance_once(self, config, cancel):
        self.stop()
        serial = config.white_balance_reference_serial or (self.cameras[0].serial if self.cameras else "")
        if serial not in {camera.serial for camera in self.cameras}:
            raise CaptureError("WB 기준 카메라가 연결되어 있지 않습니다.")
        cancel.wait(.04)
        check_cancel(cancel)
        return {"reference_serial": serial, "red": 1 / .85, "blue": 1 / .65,
            "method": "simulated camera BalanceWhiteAuto Once", "simulated": True,
            "profile": config.white_balance_profile, "camera_auto": "Off", "raw_restored": True}

    def close(self):
        self.stop()
        self.cameras = []
