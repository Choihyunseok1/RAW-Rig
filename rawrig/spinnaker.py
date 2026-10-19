"""Typed ctypes bridge to the vendor Spinnaker C API (no PySpin wheel required).

Signatures are taken from SpinnakerC.h / SpinnakerGenApiC.h, SDK 4.4.0.246.
Each acquired SDK image is copied as Bayer samples and released in a finally block.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, ExitStack
import ctypes as C
import ctypes.util
import math
from dataclasses import dataclass
from ipaddress import IPv4Address
import os
from pathlib import Path
import secrets
import socket
import sys
from threading import Barrier, Event
import time

import numpy as np

from .models import (CameraInfo, CaptureConfig, CaptureError, RetryableCaptureError,
                     WhiteBalanceRestoreError, Frame, Preset, RAW_FORMATS, check_cancel)
from .recovery import capture_checked
from .network import NetworkCamera, ipv4, normalized_mac
from .sync import validate_ptp_snapshot, validate_ptp_frames


H, SZ, I64, U64, BOOL, DOUBLE = C.c_void_p, C.c_size_t, C.c_int64, C.c_uint64, C.c_uint8, C.c_double
P = C.POINTER


class ActionResult(C.Structure):
    _fields_ = [("DeviceAddress", C.c_uint), ("Status", C.c_int)]


class SDKError(CaptureError):
    def __init__(self, name, code, message):
        self.code = code
        super().__init__(f"{name}: {message} (SDK {code})")


def find_library():
    candidates = [os.environ.get("SPINNAKER_C_LIBRARY", ""),
        str(Path(__file__).resolve().parents[1] / ".runtime/spinnaker/opt/spinnaker/lib/libSpinnaker_C.so"),
        "/opt/spinnaker/lib/libSpinnaker_C.so", ctypes.util.find_library("Spinnaker_C")]
    for candidate in candidates:
        if candidate and (Path(candidate).is_file() or "/" not in candidate):
            return candidate
    raise CaptureError("Spinnaker C 라이브러리가 없습니다. scripts/setup_sdk.sh 또는 SPINNAKER_C_LIBRARY를 사용하세요.")


class SpinAPI:
    def __init__(self):
        try:
            self.lib = C.CDLL(find_library())
        except OSError as error:
            raise CaptureError(f"Spinnaker 로딩 실패: {error}") from error
        signatures = {
            "spinErrorGetLastMessage": [C.c_char_p, P(SZ)],
            "spinSystemGetInstance": [P(H)], "spinSystemReleaseInstance": [H],
            "spinSystemGetCameras": [H, H], "spinCameraListCreateEmpty": [P(H)],
            "spinSystemGetCamerasEx": [H, BOOL, BOOL, H],
            "spinSystemUpdateCamerasEx": [H, BOOL, P(BOOL)],
            "spinSystemGetInterfaces": [H, H], "spinInterfaceListCreateEmpty": [P(H)],
            "spinInterfaceListDestroy": [H], "spinInterfaceListClear": [H],
            "spinInterfaceListGetSize": [H, P(SZ)], "spinInterfaceListGet": [H, SZ, P(H)],
            "spinInterfaceGetTLNodeMap": [H, P(H)], "spinInterfaceRelease": [H],
            "spinCameraListDestroy": [H], "spinCameraListClear": [H],
            "spinCameraListGetSize": [H, P(SZ)], "spinCameraListGet": [H, SZ, P(H)],
            "spinCameraListGetBySerial": [H, C.c_char_p, P(H)], "spinCameraRelease": [H],
            "spinCameraInit": [H], "spinCameraDeInit": [H],
            "spinCameraBeginAcquisition": [H], "spinCameraEndAcquisition": [H],
            "spinCameraGetNodeMap": [H, P(H)], "spinCameraGetTLDeviceNodeMap": [H, P(H)],
            "spinCameraGetTLStreamNodeMap": [H, P(H)], "spinCameraGetNextImageEx": [H, U64, P(H)],
            "spinNodeMapGetNode": [H, C.c_char_p, P(H)], "spinNodeMapReleaseNode": [H, H],
            "spinNodeIsReadable": [H, P(BOOL)], "spinNodeIsWritable": [H, P(BOOL)],
            "spinNodeIsAvailable": [H, P(BOOL)], "spinNodeToString": [H, C.c_char_p, P(SZ)],
            "spinEnumerationGetEntryByName": [H, C.c_char_p, P(H)],
            "spinEnumerationEntryGetIntValue": [H, P(I64)],
            "spinEnumerationReleaseNode": [H, H], "spinEnumerationSetIntValue": [H, I64],
            "spinFloatSetValue": [H, DOUBLE], "spinFloatGetValue": [H, P(DOUBLE)],
            "spinFloatGetMin": [H, P(DOUBLE)], "spinFloatGetMax": [H, P(DOUBLE)],
            "spinIntegerSetValue": [H, I64], "spinIntegerGetValue": [H, P(I64)],
            "spinIntegerGetMin": [H, P(I64)], "spinIntegerGetMax": [H, P(I64)], "spinIntegerGetInc": [H, P(I64)],
            "spinBooleanSetValue": [H, BOOL], "spinBooleanGetValue": [H, P(BOOL)], "spinCommandExecute": [H],
            "spinImageRelease": [H], "spinImageGetData": [H, P(H)],
            "spinImageGetWidth": [H, P(SZ)], "spinImageGetHeight": [H, P(SZ)],
            "spinImageGetStride": [H, P(SZ)], "spinImageGetBufferSize": [H, P(SZ)],
            "spinImageGetFrameID": [H, P(U64)], "spinImageGetTimeStamp": [H, P(U64)],
            "spinImageGetPixelFormatName": [H, C.c_char_p, P(SZ)],
            "spinImageIsIncomplete": [H, P(BOOL)], "spinImageHasCRC": [H, P(BOOL)], "spinImageCheckCRC": [H, P(BOOL)],
            "spinImageChunkDataGetFloatValue": [H, C.c_char_p, P(DOUBLE)],
        }
        for name, args in signatures.items():
            fn = getattr(self.lib, name)
            fn.argtypes, fn.restype = args, C.c_int
        # 4.4.0.246 Linux exports this C-wrapper function with a C++ symbol and
        # uint32 count/key parameters, despite size_t declarations in its C header.
        # Bind the actual exported ABI explicitly; never pass size_t* to uint32*.
        try:
            self.action = getattr(self.lib, "spinSystemSendActionCommand")
            self.action.argtypes = [H, SZ, SZ, SZ, SZ, BOOL, P(SZ), P(ActionResult)]
            self.action_count_type = SZ
        except AttributeError:
            symbol = "_Z27spinSystemSendActionCommandPvjjjybPjP20_actionCommandResult"
            self.action = getattr(self.lib, symbol)
            self.action.argtypes = [H, C.c_uint, C.c_uint, C.c_uint, U64, C.c_bool, P(C.c_uint), P(ActionResult)]
            self.action_count_type = C.c_uint
        self.action.restype = C.c_int

    def send_action(self, system, device_key, group_key, target, size, results):
        code = self.action(system, device_key, group_key, 0xFFFFFFFF, target, True, C.byref(size), results)
        if code:
            raise SDKError("spinSystemSendActionCommand", code, "예약 Action Command 전송 실패")

    def call(self, name, *args):
        code = getattr(self.lib, name)(*args)
        if code:
            buffer, size = C.create_string_buffer(4096), SZ(4096)
            self.lib.spinErrorGetLastMessage(buffer, C.byref(size))
            raise SDKError(name, code, buffer.value.decode("utf-8", "replace"))

    def value(self, name, handle, dtype, *args):
        result = dtype()
        self.call(name, handle, *args, C.byref(result))
        return result.value

    def handle(self, name, *args):
        handle = H()
        self.call(name, *args, C.byref(handle))
        return handle

    def string(self, name, handle):
        buffer, size = C.create_string_buffer(16384), SZ(16384)
        self.call(name, handle, buffer, C.byref(size))
        return buffer.value.decode("utf-8", "replace")


class NodeMap:
    def __init__(self, api: SpinAPI, handle):
        self.api, self.handle = api, handle
        self.cache = {}

    def node(self, name):
        if name not in self.cache:
            self.cache[name] = self.api.handle("spinNodeMapGetNode", self.handle, name.encode())
        return self.cache[name]

    def has(self, name):
        try:
            node = self.node(name)
            return bool(node.value and self.api.value("spinNodeIsAvailable", node, BOOL))
        except SDKError:
            return False

    def readable(self, name):
        return self.has(name) and bool(self.api.value("spinNodeIsReadable", self.node(name), BOOL))

    def writable(self, name):
        return self.has(name) and bool(self.api.value("spinNodeIsWritable", self.node(name), BOOL))

    def text(self, name):
        return self.api.string("spinNodeToString", self.node(name))

    def enum_supported(self, name, value):
        if not self.has(name):
            return False
        entry = None
        try:
            entry = self.api.handle("spinEnumerationGetEntryByName", self.node(name), value.encode())
            return bool(entry.value and self.api.value("spinNodeIsReadable", entry, BOOL))
        except SDKError:
            return False
        finally:
            if entry and entry.value:
                self.api.call("spinEnumerationReleaseNode", self.node(name), entry)

    def enum(self, name, value, optional=False):
        if optional and not self.has(name):
            return None
        if self.readable(name) and self.text(name) == value:
            return value
        if not self.writable(name) or not self.enum_supported(name, value):
            raise CaptureError(f"{name}={value} 설정을 지원하지 않거나 다른 프로그램이 카메라를 사용 중입니다.")
        entry = self.api.handle("spinEnumerationGetEntryByName", self.node(name), value.encode())
        try:
            number = self.api.value("spinEnumerationEntryGetIntValue", entry, I64)
            self.api.call("spinEnumerationSetIntValue", self.node(name), number)
        finally:
            self.api.call("spinEnumerationReleaseNode", self.node(name), entry)
        if self.text(name) != value:
            raise CaptureError(f"{name} 설정을 읽어 확인하지 못했습니다.")
        return value

    def get_float(self, name):
        return self.api.value("spinFloatGetValue", self.node(name), DOUBLE)

    def get_int(self, name):
        return self.api.value("spinIntegerGetValue", self.node(name), I64)

    def bounds(self, name, integer=False):
        kind, dtype = ("Integer", I64) if integer else ("Float", DOUBLE)
        return [self.api.value(f"spin{kind}GetMin", self.node(name), dtype),
                self.api.value(f"spin{kind}GetMax", self.node(name), dtype)]

    def floating(self, name, value, optional=False, tolerance=1e-6):
        if optional and not self.has(name):
            return None
        low, high = self.bounds(name)
        if not low <= value <= high:
            raise CaptureError(f"{name}={value} 범위 초과. 지원 범위: {low}~{high}")
        if self.writable(name):
            self.api.call("spinFloatSetValue", self.node(name), value)
        actual = self.get_float(name)
        if abs(actual - value) > tolerance:
            raise CaptureError(f"{name}: 요청값 {value} / 실제값 {actual}. 동일 설정을 적용할 수 없습니다.")
        return actual

    def resolve_float(self, name, requested, tolerance):
        """Resolve a supported setting before acquisition, without relaxing frame checks."""
        low, high = self.bounds(name)
        if not math.isfinite(requested) or not low <= requested <= high:
            raise CaptureError(f"{name}: 노이즈 촬영 요청값이 지원 범위를 벗어납니다.")
        def write_read(value):
            if not self.writable(name):
                raise CaptureError(f"{name}: 카메라 설정 쓰기 권한이 없습니다.")
            self.api.call("spinFloatSetValue", self.node(name), value)
            actual = self.get_float(name)
            if not math.isfinite(actual) or not low <= actual <= high:
                raise CaptureError(f"{name}: 실제 읽기값이 유효하지 않습니다: {actual}")
            return actual
        actual = write_read(requested)
        if abs(actual - requested) > tolerance:
            # Verify that writes respond on both sides of the requested value.
            # An ignored write cannot pass just because its current value is stable.
            step = max(2 * abs(actual-requested), tolerance, .01)
            direction = -1 if actual > requested else 1
            responsive = False
            for _ in range(8):
                target = min(high, max(low, requested + direction * step))
                other = write_read(target)
                if actual != other and min(actual, other) <= requested <= max(actual, other):
                    responsive = True
                    break
                step *= 2
            if not responsive:
                raise CaptureError(f"{name}: 센서의 설정 반올림을 검증하지 못했습니다: {requested} / {actual} / {other}")
        readback = write_read(actual)
        if abs(readback - actual) > max(1e-6, abs(actual) * 1e-9):
            raise CaptureError(f"{name}: 실제 적용값을 재적용해 확인하지 못했습니다: {actual} / {readback}")
        return actual

    def integer(self, name, value, optional=False, write_only=False):
        if optional and not self.has(name):
            return None
        if self.writable(name):
            self.api.call("spinIntegerSetValue", self.node(name), value)
        elif write_only:
            raise CaptureError(f"{name} 쓰기 권한이 없습니다.")
        if not write_only and self.get_int(name) != value:
            raise CaptureError(f"{name}={value} 설정에 실패했습니다.")
        return value

    def boolean(self, name, value, optional=False):
        if optional and not self.has(name):
            return None
        if self.writable(name):
            self.api.call("spinBooleanSetValue", self.node(name), value)
        actual = bool(self.api.value("spinBooleanGetValue", self.node(name), BOOL))
        if actual != value:
            raise CaptureError(f"{name}={value} 설정에 실패했습니다.")
        return actual

    def execute(self, name):
        if not self.writable(name):
            raise CaptureError(f"{name} 명령이 사용 불가능합니다.")
        self.api.call("spinCommandExecute", self.node(name))

    def close(self):
        for handle in self.cache.values():
            if handle.value:
                self.api.call("spinNodeMapReleaseNode", self.handle, handle)
        self.cache.clear()


@dataclass
class Device:
    info: CameraInfo
    handle: H
    nodes: NodeMap
    stream: NodeMap
    settings: dict
    running: bool = False


class SpinnakerBackend:
    def __init__(self):
        self.api = SpinAPI()
        self.system = self.api.handle("spinSystemGetInstance")
        self.camera_list = H()
        self.devices: list[Device] = []
        self.cameras = []
        self.config = CaptureConfig()
        self.pool = ThreadPoolExecutor(max_workers=3, thread_name_prefix="raw-camera")
        self.action_device_key = secrets.randbits(31) | 1
        self.action_group_key = secrets.randbits(31) | 1

    @contextmanager
    def _network_interfaces(self):
        if self.devices:
            raise CaptureError("주소 설정 전에 카메라 연결을 해제해야 합니다.")
        with ExitStack() as cleanup:
            # USB Ethernet may arrive after System was created. Refresh the
            # interface list before enumerating its transport-layer node maps.
            self.api.value("spinSystemUpdateCamerasEx", self.system, BOOL, True)
            interfaces = self.api.handle("spinInterfaceListCreateEmpty")
            cleanup.callback(self.api.call, "spinInterfaceListDestroy", interfaces)
            cleanup.callback(self.api.call, "spinInterfaceListClear", interfaces)
            self.api.call("spinSystemGetInterfaces", self.system, interfaces)
            maps = []
            for index in range(self.api.value("spinInterfaceListGetSize", interfaces, SZ)):
                handle = self.api.handle("spinInterfaceListGet", interfaces, index)
                cleanup.callback(self.api.call, "spinInterfaceRelease", handle)
                nodes = NodeMap(self.api, self.api.handle("spinInterfaceGetTLNodeMap", handle))
                cleanup.callback(nodes.close)
                maps.append(nodes)
            yield maps

    @staticmethod
    def _inventory_interface(nodes):
        if not nodes.readable("GevInterfaceMACAddress") or not nodes.get_int("GevInterfaceMACAddress"):
            return []
        host_mac = normalized_mac(nodes.get_int("GevInterfaceMACAddress"))
        host_conflict = nodes.readable("GevInterfaceIsIPConflict") and bool(nodes.api.value(
            "spinBooleanGetValue", nodes.node("GevInterfaceIsIPConflict"), BOOL))
        nodes.execute("DeviceUpdateList")
        devices = []
        for compatible, prefix, selector, count_node in ((True, "", "DeviceSelector", "DeviceCount"),
                (False, "Incompatible", "IncompatibleDeviceSelector", "IncompatibleDeviceCount")):
            count = nodes.get_int(count_node) if nodes.readable(count_node) else 0
            for index in range(count):
                nodes.integer(selector, index)
                def text(name, default=""):
                    return nodes.text(prefix + name) if nodes.readable(prefix + name) else default
                def number(name):
                    return nodes.get_int(prefix + name)
                devices.append(NetworkCamera(text("DeviceSerialNumber"), text("DeviceModelName"),
                    normalized_mac(number("GevDeviceMACAddress")), ipv4(number("GevDeviceIPAddress")),
                    ipv4(number("GevDeviceSubnetMask")), host_mac, text("DeviceAccessStatus", "Unknown"), compatible, host_conflict))
        return devices

    def network_inventory(self):
        with self._network_interfaces() as interfaces:
            return [camera for nodes in interfaces for camera in self._inventory_interface(nodes)]

    def force_ip(self, camera, address, mask, gateway):
        with self._network_interfaces() as interfaces:
            for nodes in interfaces:
                if not nodes.readable("GevInterfaceMACAddress") or not nodes.get_int("GevInterfaceMACAddress"):
                    continue
                if normalized_mac(nodes.get_int("GevInterfaceMACAddress")) != camera.interface_mac:
                    continue
                nodes.execute("DeviceUpdateList")
                for index in range(nodes.get_int("DeviceCount")):
                    nodes.integer("DeviceSelector", index)
                    if normalized_mac(nodes.get_int("GevDeviceMACAddress")) != camera.mac:
                        continue
                    if nodes.text("DeviceSerialNumber") != camera.serial:
                        raise CaptureError("할당 직전에 카메라의 MAC·시리얼 조합이 바뀌었습니다.")
                    access = nodes.text("DeviceAccessStatus")
                    if access not in ("ReadWrite", "NoAccess", "Unknown"):
                        raise CaptureError(f"카메라가 사용 중이어서 주소를 변경하지 않습니다: {access}")
                    for name, value in (("GevDeviceForceIPAddress", address), ("GevDeviceForceSubnetMask", mask),
                                        ("GevDeviceForceGateway", gateway)):
                        if not nodes.writable(name):
                            raise CaptureError(f"{name}: ForceIP를 지원하지 않습니다.")
                        nodes.integer(name, int(IPv4Address(value)))
                    nodes.execute("GevDeviceForceIP")
                    return
        raise CaptureError("MAC 주소로 할당 대상 카메라를 다시 찾지 못했습니다.")

    def validate_network_camera(self, camera, event, config=None):
        self.open([camera.serial])
        try:
            device = self.devices[0]
            tl = NodeMap(self.api, self.api.handle("spinCameraGetTLDeviceNodeMap", device.handle))
            try:
                if normalized_mac(tl.get_int("GevDeviceMACAddress")) != camera.mac or ipv4(tl.get_int("GevDeviceIPAddress")) != camera.ip:
                    raise CaptureError("연결된 카메라의 MAC·IP가 재검색 결과와 다릅니다.")
            finally:
                tl.close()
            limits = self.limits()[0]
            preset = Preset("network_validation", min(max(10000., limits["exposure_us"][0]), limits["exposure_us"][1]),
                            max(0., limits["gain_db"][0]))
            from dataclasses import replace
            config = replace(config or CaptureConfig(), expected_cameras=1, sync_mode="software", auto_reference=False,
                auto_retry_count=0)
            self.prepare(preset, config)
            self.start()
            if event.wait(.1):
                check_cancel(event)
            ids, samples = {}, []
            for _ in range(3):
                frames, trigger = capture_checked(self, preset, config, event, ids)
                samples.append({"frame_id": frames[0].frame_id, "crc_passed": frames[0].transport.get("crc_passed"),
                    "recovery": trigger.get("recovery")})
            return {"raw_verified": True, "pixel_format": frames[0].pixel_format,
                "shape": list(frames[0].pixels.shape), "frame_id": frames[0].frame_id,
                "verified_samples": samples, "consecutive_samples": len(samples),
                "transport_configuration": device.settings.get("transport_configuration", {}),
                "crc_passed": frames[0].transport.get("crc_passed"),
                "ptp_supported": device.nodes.has("GevIEEE1588"),
                "scheduled_action_supported": device.nodes.enum_supported("TriggerSource", "Action0")}
        finally:
            self._release_devices()

    def discover(self):
        if self.devices:
            raise CaptureError("카메라 연결을 해제한 뒤 다시 검색하세요.")
        if self.camera_list.value:
            self.api.call("spinCameraListClear", self.camera_list)
        else:
            self.camera_list = self.api.handle("spinCameraListCreateEmpty")
        self.api.call("spinSystemGetCamerasEx", self.system, True, True, self.camera_list)
        found = []
        for i in range(self.api.value("spinCameraListGetSize", self.camera_list, SZ)):
            cam = self.api.handle("spinCameraListGet", self.camera_list, i)
            nodes = None
            try:
                nodes = NodeMap(self.api, self.api.handle("spinCameraGetTLDeviceNodeMap", cam))
                found.append(CameraInfo(nodes.text("DeviceSerialNumber"), nodes.text("DeviceModelName")))
            finally:
                if nodes:
                    nodes.close()
                self.api.call("spinCameraRelease", cam)
        return sorted(found, key=lambda c: c.serial)

    def open(self, serials):
        if not serials or len(set(serials)) != len(serials) or len(serials) > 3:
            raise CaptureError("1~3대의 서로 다른 카메라를 선택하세요.")
        found = {c.serial: c for c in self.discover()}
        try:
            for serial in serials:
                if serial not in found:
                    raise CaptureError(f"카메라 {serial}를 찾을 수 없습니다.")
                handle = self.api.handle("spinCameraListGetBySerial", self.camera_list, serial.encode())
                initialized, nodes, stream = False, None, None
                try:
                    self.api.call("spinCameraInit", handle)
                    initialized = True
                    nodes = NodeMap(self.api, self.api.handle("spinCameraGetNodeMap", handle))
                    stream = NodeMap(self.api, self.api.handle("spinCameraGetTLStreamNodeMap", handle))
                    device = Device(found[serial], handle, nodes, stream, {})
                    self._configure_raw(device)
                    self.devices.append(device)
                except Exception:
                    if nodes:
                        nodes.close()
                    if stream:
                        stream.close()
                    if initialized:
                        self.api.call("spinCameraDeInit", handle)
                    self.api.call("spinCameraRelease", handle)
                    raise
            self.cameras = [d.info for d in self.devices]
            return self.cameras
        except Exception as error:
            self._release_devices()
            raise CaptureError(f"카메라 연결 실패. SpinView의 획득을 멈추고 장치 연결을 해제해 주세요.\n{error}") from error

    def _configure_raw(self, device):
        n = device.nodes
        n.enum("TriggerMode", "Off")
        n.enum("SequencerMode", "Off", optional=True)
        n.enum("ExposureAuto", "Off")
        n.enum("GainAuto", "Off")
        n.enum("BalanceWhiteAuto", "Off", optional=True)
        processing = {}
        for channel in ("Red", "Blue"):
            if n.enum_supported("BalanceRatioSelector", channel):
                n.enum("BalanceRatioSelector", channel)
                processing[f"BalanceRatio{channel}"] = n.floating("BalanceRatio", 1., optional=True, tolerance=.005)
        for name in ("GammaEnable", "LUTEnable", "SharpnessEnabled", "SharpeningEnable", "ColorTransformationEnable"):
            processing[name] = n.boolean(name, False, optional=True)
        for name in ("DefectCorrectStaticEnable", "DefectCorrectDynamicEnable"):
            processing[name] = n.boolean(name, False, optional=True)
        # ISP bypass also disables pipeline color corrections on supported cameras.
        processing["IspEnable"] = n.boolean("IspEnable", False, optional=True)
        n.enum("ImageCompressionMode", "Off", optional=True)
        n.boolean("GevSCPSBigEndian", False, optional=True)
        n.enum("TestPattern", "Off", optional=True)
        n.boolean("ReverseX", False, optional=True)
        n.boolean("ReverseY", False, optional=True)
        for name in ("BinningHorizontal", "BinningVertical", "DecimationHorizontal", "DecimationVertical"):
            n.integer(name, 1, optional=True)
        n.integer("OffsetX", 0, optional=True)
        n.integer("OffsetY", 0, optional=True)
        for name in ("Width", "Height"):
            low, high = n.bounds(name, integer=True)
            inc = self.api.value("spinIntegerGetInc", n.node(name), I64)
            n.integer(name, low + (high - low) // max(1, inc) * max(1, inc))
        for bits in ("Bit16", "Bit14", "Bit12", "Bit10", "Bit8"):
            if n.enum_supported("AdcBitDepth", bits):
                n.enum("AdcBitDepth", bits)
                break
        formats = [f"Bayer{p}16" for p in ("RG", "GR", "GB", "BG")]
        pixel_format = next((f for f in formats if n.enum_supported("PixelFormat", f)), None)
        if not pixel_format:
            raise CaptureError(f"{device.info.serial}: unpacked 16-bit Bayer 형식이 없습니다. RAW 정밀도를 낮추지 않고 중단합니다.")
        n.enum("PixelFormat", pixel_format)
        n.enum("GainSelector", "All", optional=True)
        n.enum("ExposureMode", "Timed")
        n.enum("AcquisitionMode", "Continuous")
        n.boolean("AcquisitionFrameRateEnable", False, optional=True)
        # Three 30 MB/s streams fit within a shared 1 GbE link; snapshots are not continuous recording.
        throughput = None
        if n.has("DeviceLinkThroughputLimit"):
            n.enum("DeviceLinkThroughputLimitMode", "On", optional=True)
            low, high = n.bounds("DeviceLinkThroughputLimit", integer=True)
            inc = self.api.value("spinIntegerGetInc", n.node("DeviceLinkThroughputLimit"), I64)
            value = min(max(30000000, low), high)
            value = low + (value - low) // max(1, inc) * max(1, inc)
            throughput = n.integer("DeviceLinkThroughputLimit", value)
        device.stream.enum("StreamBufferHandlingMode", "OldestFirst")
        transport_configuration = {}
        # Keep packets within a standard Ethernet MTU and request SDK packet
        # resends where available. Optional optimizations must not prohibit RAW.
        for nodes, name, value in ((device.stream, "StreamPacketResendEnable", True),
                                    (n, "GevSCPSPacketSize", 1500)):
            if not nodes.has(name):
                continue
            try:
                if isinstance(value, bool):
                    transport_configuration[name] = nodes.boolean(name, value)
                else:
                    low, high = nodes.bounds(name, integer=True)
                    inc = max(1, self.api.value("spinIntegerGetInc", nodes.node(name), I64))
                    target = low + (min(max(value, low), high) - low) // inc * inc
                    transport_configuration[name] = nodes.integer(name, target)
            except CaptureError as error:
                transport_configuration[name] = {"warning": str(error)}
        chunks = []
        if n.has("ChunkModeActive"):
            n.boolean("ChunkModeActive", True)
            for name in ("ExposureTime", "Gain", "FrameID", "Timestamp", "CRC"):
                if n.enum_supported("ChunkSelector", name):
                    n.enum("ChunkSelector", name)
                    n.boolean("ChunkEnable", True)
                    chunks.append(name)
            n.enum("ChunkGainSelector", "All", optional=True)
        device.settings = {"serial": device.info.serial, "width": n.get_int("Width"), "height": n.get_int("Height"),
            "pixel_format": n.text("PixelFormat"), "adc_bit_depth": n.text("AdcBitDepth") if n.readable("AdcBitDepth") else None,
            "pixel_max": n.get_int("PixelDynamicRangeMax") if n.readable("PixelDynamicRangeMax") else 65535,
            "bit_alignment": "camera delivered samples unchanged; see model technical reference",
            "raw_processing": processing, "enabled_chunks": chunks, "link_throughput_bytes_per_second": throughput,
            "transport_configuration": transport_configuration,
            "black_level_percent": n.get_float("BlackLevel") if n.readable("BlackLevel") else None,
            "black_level_clamping": n.text("BlackLevelClampingEnable") if n.readable("BlackLevelClampingEnable") else None,
            "defect_pixel_correction": n.text("DefectCorrectStaticEnable") if n.readable("DefectCorrectStaticEnable") else None}
        if n.readable("GainConversionMode"):
            device.settings["gain_conversion_mode"] = n.text("GainConversionMode")
        if device.info.model.startswith("Blackfly S BFS-PGE") and device.settings["adc_bit_depth"]:
            adc_bits = int(device.settings["adc_bit_depth"].replace("Bit", ""))
            device.settings["sensor_white_level"] = ((1 << adc_bits) - 1) << (16 - adc_bits)
        if n.readable("GevCurrentIPAddress"):
            device.settings["device_ip_integer"] = n.get_int("GevCurrentIPAddress")

    def limits(self):
        return [{"serial": d.info.serial, "exposure_us": d.nodes.bounds("ExposureTime"),
                 "gain_db": d.nodes.bounds("Gain")} for d in self.devices]

    def configure_profiles(self, config):
        self.stop()
        result = []
        for device in self.devices:
            n = device.nodes
            declared = config.camera_profiles.get(device.info.serial, {})
            if 'link_throughput_bytes_per_second' in declared:
                requested = int(declared['link_throughput_bytes_per_second'])
                low, high = n.bounds('DeviceLinkThroughputLimit', integer=True)
                inc = max(1, self.api.value('spinIntegerGetInc', n.node('DeviceLinkThroughputLimit'), I64))
                if not low <= requested <= high:
                    raise CaptureError('요청한 전송 대역폭이 카메라 지원 범위 밖입니다.')
                actual = n.integer('DeviceLinkThroughputLimit', low + (requested-low)//inc*inc)
                device.settings['link_throughput_bytes_per_second'] = actual
            if 'black_level_clamping' in declared:
                n.boolean('BlackLevelClampingEnable', declared['black_level_clamping'])
            if 'black_level_percent' in declared:
                n.enum('BlackLevelSelector', 'All')
                n.floating('BlackLevel', float(declared['black_level_percent']), tolerance=.005)
            device.settings['black_level_percent'] = n.get_float('BlackLevel') if n.readable('BlackLevel') else None
            device.settings['black_level_clamping'] = n.text('BlackLevelClampingEnable') if n.readable('BlackLevelClampingEnable') else None
            conversion = declared.get("gain_conversion_mode", "preserve")
            if conversion != "preserve":
                if not n.enum_supported("GainConversionMode", conversion):
                    raise CaptureError(f"{device.info.serial}: 변환 게인 {conversion}을 지원하지 않습니다.")
                n.enum("GainConversionMode", conversion)
            modes = {name: n.text(name) for name in ("GainConversionMode", "SensorShutterMode", "HDRMode", "HDREnable", "SensorHDRMode") if n.readable(name)}
            if "GainConversionMode" in modes:
                device.settings["gain_conversion_mode"] = modes["GainConversionMode"]
            device.settings["sensor_modes"] = modes
            device.settings["declared_profile"] = declared
            result.append({"serial": device.info.serial, "model": device.info.model, "declared": declared,
                "readback": modes, "raw": dict(device.settings)})
        return result

    def operating_presets(self):
        return {d.info.serial: Preset("noise_fixed", d.nodes.get_float("ExposureTime"), d.nodes.get_float("Gain")) for d in self.devices}

    def resolve_noise_plan(self, plan, config, cancel):
        """Preflight all steps against sensor timing before the long dark stack."""
        self.stop()
        original = self.operating_presets()
        resolved = []
        previous = {}
        try:
            for point in plan:
                check_cancel(cancel)
                actual = {}
                for device in self.devices:
                    n = device.nodes
                    n.enum("ExposureAuto", "Off")
                    n.enum("GainAuto", "Off")
                    t = n.resolve_float("ExposureTime", point["exposure_us"], max(1., point["exposure_us"] * .005))
                    g = n.resolve_float("Gain", point["gain_db"], .05)
                    if device.info.serial in previous and t <= previous[device.info.serial]:
                        raise CaptureError(f"{device.info.serial}: 센서 시간 간격 때문에 서로 다른 {len(plan)}개 노출시간을 적용할 수 없습니다.")
                    previous[device.info.serial] = t
                    actual[device.info.serial] = {"name": f"noise_{point['index']:03d}", "exposure_us": t, "gain_db": g}
                resolved.append({**point, "camera_presets": actual})
            return resolved
        finally:
            primary_error = sys.exc_info()[1]
            try:
                self.prepare(original, config)
            except Exception as restore_error:
                if primary_error is None:
                    raise
                if hasattr(primary_error, "add_note"):
                    primary_error.add_note(f"노이즈 설정 사전 검증 후 복원 실패: {restore_error}")

    def prepare(self, preset, config):
        self.stop()
        self.light_phase=None
        self.light_phase_fraction=None
        self.light_phase_latency_ns=0
        self.light_phase_latencies={}
        presets = preset if isinstance(preset, dict) else {c.serial: preset for c in self.cameras}
        if set(presets) != {c.serial for c in self.cameras}:
            raise CaptureError("카메라별 노출 컨트롤 설정에 누락/중복이 있습니다.")
        for value in presets.values():
            value.validate()
        self.config = config
        master = config.master_serial
        if config.sync_mode == "hardware" and master not in {c.serial for c in self.cameras}:
            raise CaptureError("하드웨어 동기 촬영에는 선택된 마스터 카메라가 필요합니다.")
        delay_bounds = [d.nodes.bounds("TriggerDelay") for d in self.devices if d.nodes.has("TriggerDelay")]
        common_delay = max((bounds[0] for bounds in delay_bounds), default=0.)
        if any(common_delay > bounds[1] for bounds in delay_bounds):
            raise CaptureError("카메라들의 공통 TriggerDelay 범위가 없습니다.")
        settings = []
        for device in self.devices:
            preset = presets[device.info.serial]
            n = device.nodes
            n.enum("TriggerMode", "Off")
            n.enum("ExposureAuto", "Off")
            n.enum("GainAuto", "Off")
            device.settings["exposure_us"] = n.floating("ExposureTime", preset.exposure_us, tolerance=max(1., preset.exposure_us * .005))
            device.settings["gain_db"] = n.floating("Gain", preset.gain_db, tolerance=.05)
            n.enum("TriggerSelector", "FrameStart")
            n.enum("TriggerOverlap", "Off", optional=True)
            device.settings["trigger_delay_us"] = n.floating("TriggerDelay", common_delay, optional=True, tolerance=1.)
            if config.sync_mode == "ptp":
                n.enum("ActionSelector", "Action0")
                n.integer("ActionDeviceKey", self.action_device_key, write_only=True)
                n.integer("ActionGroupKey", self.action_group_key)
                n.integer("ActionGroupMask", 0xFFFFFFFF)
                n.enum("ActionUnconditionalMode", "Off", optional=True)
                n.execute("ActionQueueEmpty")
                n.enum("TriggerSource", "Action0")
                n.enum("TriggerActivation", "RisingEdge", optional=True)
            elif config.sync_mode == "software" or device.info.serial == master:
                n.enum("TriggerSource", "Software")
                if config.sync_mode == "hardware":
                    n.enum("LineSelector", config.output_line)
                    n.enum("LineMode", "Output")
                    n.boolean("LineInverter", False, optional=True)
                    n.enum("LineSource", "ExposureActive")
            else:
                n.enum("LineSelector", config.input_line)
                n.enum("LineMode", "Input")
                n.enum("TriggerSource", config.input_line)
                n.enum("TriggerActivation", config.trigger_edge)
            n.enum("TriggerMode", "On")
            settings.append(dict(device.settings))
        return settings

    def start(self):
        # Slaves are armed first, then the master. A settle wait follows in the capture engine.
        order = sorted(self.devices, key=lambda d: d.info.serial == self.config.master_serial)
        try:
            for device in order:
                if not device.running:
                    self.api.call("spinCameraBeginAcquisition", device.handle)
                    device.running = True
        except Exception:
            self.stop()
            raise

    def stop(self):
        errors = []
        for device in self.devices:
            if device.nodes.writable("ActionQueueEmpty"):
                try:
                    device.nodes.execute("ActionQueueEmpty")
                except Exception as error:
                    errors.append(error)
            if device.running:
                try:
                    self.api.call("spinCameraEndAcquisition", device.handle)
                    device.running = False
                except Exception as error:
                    errors.append(error)
        if errors:
            raise errors[0]

    def _ptp_snapshot(self):
        result = []
        for device in self.devices:
            n = device.nodes
            n.execute("GevIEEE1588DataSetLatch")
            result.append({"serial": device.info.serial, "status": n.text("GevIEEE1588StatusLatched"),
                "clock_id": n.get_int("GevIEEE1588ClockId"),
                "parent_clock_id": n.get_int("GevIEEE1588ParentClockIdLatched"),
                "offset_ns": n.get_int("GevIEEE1588OffsetFromMasterLatched")})
        return result

    def synchronize(self, config, cancel, progress):
        if config.sync_mode != "ptp":
            return {"mode": config.sync_mode, "locked": False}
        for device in self.devices:
            n = device.nodes
            required = ("GevIEEE1588", "GevIEEE1588DataSetLatch", "GevIEEE1588StatusLatched",
                "GevIEEE1588ParentClockIdLatched", "GevIEEE1588OffsetFromMasterLatched", "TimestampLatch", "TimestampLatchValue", "ActionQueueEmpty")
            if not all(n.has(name) for name in required) or not n.enum_supported("TriggerSource", "Action0"):
                raise CaptureError(f"{device.info.serial}: PTP + 예약 Action Command를 지원하지 않습니다.")
            if n.get_int("GevTimestampTickFrequency") != 1_000_000_000:
                raise CaptureError("이 버전은 1 GHz timestamp 카메라만 PTP 예약 촬영을 지원합니다.")
            n.enum("GevIEEE1588Mode", "Auto", optional=True)
            n.boolean("GevIEEE1588", True)
        deadline, consecutive = time.monotonic() + config.ptp_wait_seconds, 0
        last_error = None
        while time.monotonic() < deadline:
            check_cancel(cancel)
            snapshot = self._ptp_snapshot()
            try:
                validate_ptp_snapshot(snapshot, config.ptp_max_offset_us)
                consecutive += 1
                if consecutive >= 3:
                    progress("PTP 동기화 완료: " + " / ".join(f"{s['serial']} {s['status']} {s['offset_ns']/1000:.3f} µs" for s in snapshot))
                    return {"mode": "ptp", "locked": True, "initial_snapshot": snapshot,
                        "max_offset_us": config.ptp_max_offset_us, "max_image_timestamp_spread_us": config.ptp_max_timestamp_spread_us,
                        "note": "Common scheduled Action0; timestamp spread is checked for every group. Optical exposure skew is not independently measured."}
            except CaptureError as error:
                consecutive, last_error = 0, error
            progress("PTP 안정화 대기: " + " / ".join(f"{s['serial']} {s['status']} {s['offset_ns']/1000:.1f} µs" for s in snapshot))
            cancel.wait(.5)
        raise CaptureError(f"PTP 동기화 시간 초과: {last_error}")

    def _schedule_action(self, cancel):
        check_cancel(cancel)
        snapshot = self._ptp_snapshot()
        validate_ptp_snapshot(snapshot, self.config.ptp_max_offset_us)
        reference = self.devices[0].nodes
        reference.execute("TimestampLatch")
        now = reference.get_int("TimestampLatchValue")
        target = now + self.config.action_lead_ms * 1_000_000
        size = self.api.action_count_type(len(self.devices))
        results = (ActionResult * len(self.devices))()
        self.api.send_action(self.system, self.action_device_key, self.action_group_key, target, size, results)
        # SDK's uint32 response address is in network byte order in host memory.
        acks = [{"device_ip_integer": socket.ntohl(results[i].DeviceAddress), "status": results[i].Status} for i in range(min(size.value, len(self.devices)))]
        expected = {d.settings["device_ip_integer"] for d in self.devices}
        if size.value != len(self.devices) or {a["device_ip_integer"] for a in acks} != expected or any(a["status"] != 0 for a in acks):
            raise RetryableCaptureError(f"예약 Action Command 응답 누락/거절/늦음: {acks}")
        return {"mode": "ptp", "scheduled_action_time_ns": target, "action_lead_ms": self.config.action_lead_ms,
            "ptp_before": snapshot, "acknowledgements": acks, "exposure_skew_measured": False}

    def _read(self, device: Device, timeout_ms: int, cancel: Event):
        deadline = time.monotonic() + timeout_ms / 1000
        image = H()
        while True:
            check_cancel(cancel)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RetryableCaptureError(f"{device.info.serial}: 수신 시간 초과. 카메라 응답 또는 전송이 지연되었습니다.")
            try:
                self.api.call("spinCameraGetNextImageEx", device.handle, min(200, max(1, int(remaining * 1000))), C.byref(image))
                break
            except SDKError as error:
                if error.code != -1011:
                    raise
        try:
            incomplete = bool(self.api.value("spinImageIsIncomplete", image, BOOL))
            if incomplete:
                raise RetryableCaptureError(f"{device.info.serial}: 네트워크/전송에서 불완전 프레임 발생.")
            pixel_format = self.api.string("spinImageGetPixelFormatName", image)
            if pixel_format not in RAW_FORMATS:
                raise CaptureError(f"{device.info.serial}: Bayer RAW가 아닌 프레임 수신: {pixel_format}")
            width = self.api.value("spinImageGetWidth", image, SZ)
            height = self.api.value("spinImageGetHeight", image, SZ)
            stride = self.api.value("spinImageGetStride", image, SZ)
            buffer_size = self.api.value("spinImageGetBufferSize", image, SZ)
            bpp = 2 if pixel_format.endswith("16") else 1
            if not width or not height or stride < width * bpp or buffer_size < stride * height:
                raise CaptureError("SDK RAW 버퍼의 크기/stride가 올바르지 않습니다.")
            data = self.api.handle("spinImageGetData", image)
            if not data.value:
                raise CaptureError("SDK RAW 버퍼가 비어 있습니다.")
            view = (C.c_uint8 * (stride * height)).from_address(data.value)
            # Preserve every pixel value; only transport row padding is omitted from the dataset.
            pixels = np.ndarray((height, width), dtype="<u2" if bpp == 2 else "u1", buffer=view, strides=(stride, bpp)).copy()
            chunk = {}
            for name, key in (("ExposureTime", "exposure_us"), ("Gain", "gain_db")):
                if name in device.settings["enabled_chunks"]:
                    value = DOUBLE()
                    # API expects GenICam chunk node names, not selector enum names.
                    self.api.call("spinImageChunkDataGetFloatValue", image, ("Chunk" + name).encode(), C.byref(value))
                    chunk[key] = value.value
            has_crc = bool(self.api.value("spinImageHasCRC", image, BOOL))
            crc = bool(self.api.value("spinImageCheckCRC", image, BOOL)) if has_crc else None
            if crc is False:
                raise RetryableCaptureError(f"{device.info.serial}: 카메라 CRC 실패.")
            return Frame(device.info, pixels, pixel_format,
                self.api.value("spinImageGetFrameID", image, U64),
                self.api.value("spinImageGetTimeStamp", image, U64), time.monotonic_ns(), dict(device.settings), chunk,
                {"incomplete": False, "crc_passed": crc, "crc_status": "checked" if has_crc else "not_available",
                 "sdk_stride_bytes": stride, "sdk_buffer_bytes": buffer_size})
        finally:
            primary_error = sys.exc_info()[1]
            try:
                self.api.call("spinImageRelease", image)
            except Exception as error:
                if primary_error is None:
                    raise
                if hasattr(primary_error, "add_note"):
                    primary_error.add_note(f"이미지 버퍼 정리 오류: {error}")

    def probe_timestamp_position(self, timeout_ms, cancel):
        """Measure timestamp dependence on exposure; do not assume a convention."""
        d=self.devices[0]
        observations=[]
        for requested in [24.,50000.]*8:
            check_cancel(cancel)
            before=time.monotonic_ns()
            d.nodes.execute('TimestampLatch')
            after=time.monotonic_ns()
            clock=d.nodes.get_int('TimestampLatchValue')
            frame_group,trigger=self.capture_light_reference(requested,timeout_ms,cancel)
            f=frame_group[0]
            command=trigger['command_intervals_monotonic_ns'][f.camera.serial][0]
            estimated=clock+command-(before+after)//2
            observations.append({'exposure_us':f.chunk['exposure_us'],'frame_id':f.frame_id,
                'camera_timestamp_ns':f.camera_timestamp_ns,'host_received_monotonic_ns':f.host_received_ns,
                'timestamp_minus_estimated_command_ns':f.camera_timestamp_ns-estimated,
                'crc_passed':f.transport.get('crc_passed')})
        low=np.array([o['timestamp_minus_estimated_command_ns'] for o in observations[::2]])
        high=np.array([o['timestamp_minus_estimated_command_ns'] for o in observations[1::2]])
        scale=(np.median(high)-np.median(low))/((observations[1]['exposure_us']-observations[0]['exposure_us'])*1000)
        if abs(scale)<.2:
            position='exposure_start'
        elif abs(scale-1)<.2:
            position='exposure_end'
        else:
            raise CaptureError(f'카메라 timestamp 기준을 구분하지 못했습니다: 노출시간 의존도 {scale:.3f}')
        ids=[o['frame_id'] for o in observations]
        if any(b<=a for a,b in zip(ids,ids[1:])):
            raise CaptureError('독립 조명 기준 전환 중 FrameID 순서 검증 실패')
        result={'position':position,'normalized_exposure_time_dependence':float(scale),
                'method':'16 actual frames; alternating 24us and 50ms; hardware clock latch and software command intervals. Fixed timestamp offset is absorbed by illumination-phase fit.',
                'observations':observations}
        self.camera_timestamp_position=position
        self.timestamp_position_probe=result
        return result

    def capture_light_reference(self, exposure_us, timeout_ms, cancel):
        """Independent illumination probe, retaining acquisition/FrameID sequence.

        ExposureTime is restored before returning. Probe frames are never added
        to a noise stack and carry their real exposure-time chunk values.
        """
        if len(self.devices)!=1 or self.config.sync_mode!='software':
            raise CaptureError('독립 조명 기준 촬영은 단일 소프트웨어 트리거 카메라가 필요합니다.')
        d=self.devices[0]
        original=d.settings['exposure_us']
        try:
            actual=d.nodes.floating('ExposureTime',exposure_us,tolerance=max(1.,exposure_us*.005))
            d.settings['exposure_us']=actual
            frames,trigger=self.capture(timeout_ms,cancel)
            for frame in frames:
                reported=frame.chunk.get('exposure_us')
                if reported is None or abs(reported-actual)>max(1.,actual*.005):
                    raise CaptureError('독립 조명 기준의 실제 노출시간 chunk 검증 실패')
                frame.settings['analysis_excluded']='independent illumination reference'
            return frames,trigger
        finally:
            restored=d.nodes.floating('ExposureTime',original,tolerance=max(1.,original*.005))
            d.settings['exposure_us']=restored

    def capture(self, timeout_ms, cancel):
        check_cancel(cancel)
        if not all(d.running for d in self.devices):
            raise CaptureError("모든 카메라가 시작되지 않았습니다.")
        if self.config.sync_mode == "ptp":
            trigger_info = self._schedule_action(cancel)
            timeout_ms += self.config.action_lead_ms
        elif self.config.sync_mode == "software":
            phase_timing = self._wait_light_phase(cancel)
            barrier = Barrier(len(self.devices))
            def trigger(device):
                barrier.wait(timeout=3)
                before = time.monotonic_ns()
                device.nodes.execute("TriggerSoftware")
                return before, time.monotonic_ns()
            futures = [self.pool.submit(trigger, d) for d in self.devices]
            timings, errors = [], []
            for future in futures:
                try:
                    timings.append(future.result())
                except Exception as error:
                    errors.append(error)
            if errors:
                raise errors[0]
            trigger_info = {"mode": "software", "software_dispatch_span_ns": max(t[0] for t in timings) - min(t[0] for t in timings),
                "command_intervals_monotonic_ns": {d.info.serial: list(t) for d, t in zip(self.devices, timings)},
                "exposure_skew_measured": False}
            if phase_timing is not None:
                trigger_info['illumination_phase']=phase_timing
        else:
            master = next(d for d in self.devices if d.info.serial == self.config.master_serial)
            start = time.monotonic_ns()
            master.nodes.execute("TriggerSoftware")
            trigger_info = {"mode": "hardware", "master_serial": master.info.serial,
                "master_output": self.config.output_line, "slave_input": self.config.input_line,
                "slave_edge": self.config.trigger_edge, "signal": "ExposureActive", "host_command_ns": start,
                "exposure_skew_measured": False, "note": "Slave starts follow master's exposure through GPIO; propagation not calibrated."}
        futures = [self.pool.submit(self._read, d, timeout_ms, cancel) for d in self.devices]
        frames, errors = [], []
        # Drain every future before ending acquisition or releasing device handles.
        for future in futures:
            try:
                frames.append(future.result())
            except Exception as error:
                errors.append(error)
        if errors:
            raise errors[0]
        if trigger_info.get('illumination_phase'):
            target=trigger_info['illumination_phase']['target_camera_timestamp_ns']
            error=frames[0].camera_timestamp_ns-target
            trigger_info['illumination_phase']['actual_phase_error_ns']=error
            period=self.light_phase['period_ns']
            circular_error=(error+period/2)%period-period/2
            key=frames[0].settings['exposure_us']
            latencies=getattr(self,'light_phase_latencies',{})
            latency=latencies.get(key,self.light_phase_latency_ns)
            self.light_phase_latency_ns=int(max(-period/2,min(period/2,latency+circular_error*.2)))
            latencies[key]=self.light_phase_latency_ns
            self.light_phase_latencies=latencies
        if self.config.sync_mode == "ptp":
            after = self._ptp_snapshot()
            validate_ptp_snapshot(after, self.config.ptp_max_offset_us)
            trigger_info["ptp_after"] = after
            trigger_info["image_timestamp_spread_ns"] = validate_ptp_frames(frames, trigger_info["scheduled_action_time_ns"], self.config.ptp_max_timestamp_spread_us)
        return frames, trigger_info

    def set_light_phase(self, model, phase_fraction=None):
        self.light_phase = None
        if len(self.devices)!=1 or self.config.sync_mode!='software':
            return False
        n=self.devices[0].nodes
        if not n.has('TimestampLatch') or not n.has('TimestampLatchValue'):
            return False
        if not n.has('GevTimestampTickFrequency') or n.get_int('GevTimestampTickFrequency') != 1000000000:
            return False
        self.light_phase=model
        self.light_phase_fraction=phase_fraction
        if not hasattr(self,'light_phase_latency_ns'):
            self.light_phase_latency_ns=0
        return True

    def _wait_light_phase(self, cancel):
        model=getattr(self,'light_phase',None)
        if model is None or len(self.devices)!=1:
            return None
        from .light_phase import phase_target_ns
        n=self.devices[0].nodes
        key=getattr(self.devices[0],'settings',{}).get('exposure_us',0)
        latency=getattr(self,'light_phase_latencies',{}).get(key,self.light_phase_latency_ns)
        before=time.monotonic_ns()
        n.execute('TimestampLatch')
        after=time.monotonic_ns()
        clock=n.get_int('TimestampLatchValue')
        host_reference=(before+after)//2
        estimated_now=clock+time.monotonic_ns()-host_reference
        target=phase_target_ns(estimated_now,model,lead_ns=2000000+max(latency,0),phase_fraction=getattr(self,'light_phase_fraction',None))
        host_target=host_reference+(target-clock)-latency
        wait_ns=host_target-time.monotonic_ns()-250000
        if wait_ns>0 and cancel.wait(wait_ns/1e9):
            check_cancel(cancel)
        while time.monotonic_ns()<host_target:
            check_cancel(cancel)
        return {'target_camera_timestamp_ns':target,'frequency_hz':model['frequency_hz'],
                'phase_fraction':getattr(self,'light_phase_fraction',None),
                'method':'camera clock latch; software trigger timing servo; verified photometry required'}

    def white_balance_once(self, config, cancel):
        """Use camera AWB Once for calibration, then restore neutral Bayer acquisition."""
        self.stop()
        check_cancel(cancel)
        serial = config.white_balance_reference_serial or (self.cameras[0].serial if self.cameras else "")
        device = next((device for device in self.devices if device.info.serial == serial), None)
        if device is None:
            raise CaptureError("WB 기준 카메라가 연결되어 있지 않습니다.")
        n = device.nodes
        original_isp = n.text("IspEnable") if n.readable("IspEnable") else None
        original_trigger = n.text("TriggerMode")
        original_selector = n.text("BalanceRatioSelector") if n.readable("BalanceRatioSelector") else None
        original_profile = n.text("BalanceWhiteAutoProfile") if n.readable("BalanceWhiteAutoProfile") else None
        started = time.monotonic()
        samples = 0
        try:
            n.enum("TriggerMode", "Off")
            n.enum("ExposureAuto", "Off")
            n.enum("GainAuto", "Off")
            if original_isp is not None:
                n.boolean("IspEnable", True)
            # Some models expose AWB nodes only while their ISP statistics are enabled.
            if not n.enum_supported("BalanceWhiteAuto", "Once") or not all(
                    n.enum_supported("BalanceRatioSelector", channel) for channel in ("Red", "Blue")):
                raise CaptureError(f"{serial}: WB 1회 자동 조절을 지원하지 않습니다. 다른 기준 카메라를 선택하세요.")
            if original_selector is None:
                original_selector = n.text("BalanceRatioSelector")
            if original_profile is None and n.readable("BalanceWhiteAutoProfile"):
                original_profile = n.text("BalanceWhiteAutoProfile")
            n.enum("BalanceWhiteAuto", "Off")
            if original_profile is not None:
                n.enum("BalanceWhiteAutoProfile", config.white_balance_profile)
            n.enum("BalanceWhiteAuto", "Once")
            self.api.call("spinCameraBeginAcquisition", device.handle)
            device.running = True
            deadline = started + config.white_balance_timeout_seconds
            while True:
                check_cancel(cancel)
                if time.monotonic() >= deadline:
                    raise CaptureError("WB 1회 측정이 제한시간 안에 완료되지 않았습니다. 장면 밝기를 확인하고 다시 누르세요. 기존 WB 값은 유지합니다.")
                image = H()
                try:
                    self.api.call("spinCameraGetNextImageEx", device.handle, 200, C.byref(image))
                    if not self.api.value("spinImageIsIncomplete", image, BOOL):
                        samples += 1
                except SDKError as error:
                    if error.code != -1011:
                        raise
                finally:
                    if image.value:
                        self.api.call("spinImageRelease", image)
                if samples and n.text("BalanceWhiteAuto") == "Off":
                    break
            ratios = {}
            for channel in ("Red", "Blue"):
                n.enum("BalanceRatioSelector", channel)
                ratios[channel.lower()] = n.get_float("BalanceRatio")
            if not all(math.isfinite(value) and .1 <= value <= 8 for value in ratios.values()):
                raise CaptureError("측정한 WB 값이 지원 표시 범위 0.1~8 밖입니다. 기존 WB 값은 유지합니다.")
            return {"reference_serial": serial, **ratios, "method": "camera BalanceWhiteAuto Once",
                "profile": config.white_balance_profile if original_profile is not None else "camera default",
                "samples": samples, "elapsed_seconds": time.monotonic() - started,
                "exposure_us": n.get_float("ExposureTime"), "gain_db": n.get_float("Gain"),
                "camera_auto": "Off", "raw_restored": True}
        finally:
            primary_error = sys.exc_info()[1]
            errors = []
            def restore(callback, *args, **kwargs):
                try:
                    callback(*args, **kwargs)
                except Exception as error:
                    errors.append(str(error))
            restore(self.stop)
            restore(n.enum, "BalanceWhiteAuto", "Off", optional=True)
            for channel in ("Red", "Blue"):
                if n.enum_supported("BalanceRatioSelector", channel):
                    restore(n.enum, "BalanceRatioSelector", channel)
                    restore(n.floating, "BalanceRatio", 1.)
            if original_selector is not None:
                restore(n.enum, "BalanceRatioSelector", original_selector)
            if original_profile is not None:
                restore(n.enum, "BalanceWhiteAutoProfile", original_profile)
            if original_isp is not None:
                restore(n.boolean, "IspEnable", original_isp.lower() in ("true", "1"))
            restore(n.enum, "TriggerMode", original_trigger)
            if errors:
                raise WhiteBalanceRestoreError("WB 측정 후 RAW 설정 복원 실패: " + " / ".join(errors)) from primary_error

    def meter(self, config, cancel):
        self.stop()
        serial = config.reference_serial or self.cameras[0].serial
        if serial not in {d.info.serial for d in self.devices}:
            raise CaptureError("자동 측광 기준 카메라가 연결되어 있지 않습니다.")
        device = next(d for d in self.devices if d.info.serial == serial)
        n = device.nodes
        n.enum("TriggerMode", "Off")
        n.enum("ExposureAuto", "Off")
        n.enum("GainAuto", "Off")
        # ISP is temporarily enabled for the camera's auto statistics, then RAW bypass is restored.
        original_isp = n.text("IspEnable") if n.readable("IspEnable") else None
        try:
            if original_isp is not None:
                n.boolean("IspEnable", True)
            auto_controls = {}
            n.enum("AutoExposureTargetGreyValueAuto", "Off", optional=True)
            # All references receive the same requested metering constraints. Missing nodes are recorded.
            for name, value in (("AutoExposureTargetGreyValue", config.meter_target_percent),
                                ("AutoExposureExposureTimeUpperLimit", config.meter_max_exposure_us),
                                ("AutoExposureGainUpperLimit", config.meter_max_gain_db)):
                if n.writable(name):
                    low, high = n.bounds(name)
                    actual = min(max(value, low), high)
                    n.floating(name, actual, tolerance=max(.01, actual * .005))
                    auto_controls[name] = {"requested": value, "applied": actual}
                else:
                    auto_controls[name] = {"requested": value, "supported": False}
            n.enum("ExposureAuto", "Continuous")
            n.enum("GainAuto", "Continuous")
            self.api.call("spinCameraBeginAcquisition", device.handle)
            device.running = True
            history = []
            deadline = time.monotonic() + config.meter_seconds
            while time.monotonic() < deadline:
                check_cancel(cancel)
                image = H()
                try:
                    self.api.call("spinCameraGetNextImageEx", device.handle, 200, C.byref(image))
                    if not self.api.value("spinImageIsIncomplete", image, BOOL):
                        history.append((n.get_float("ExposureTime"), n.get_float("Gain")))
                except SDKError as error:
                    if error.code != -1011:
                        raise
                finally:
                    if image.value:
                        self.api.call("spinImageRelease", image)
            self.stop()
            n.enum("ExposureAuto", "Off")
            n.enum("GainAuto", "Off")
            exposure, gain = n.get_float("ExposureTime"), n.get_float("Gain")
            recent = history[-5:]
            stable = len(recent) == 5 and all(abs(e - exposure) <= max(2., exposure * .03) and abs(g - gain) <= .3 for e, g in recent)
            return {"reference_serial": serial, "exposure_us": exposure, "gain_db": gain, "stable": stable,
                "samples": len(history), "last_samples": recent, "meter_seconds": config.meter_seconds,
                "target_grey_value": n.get_float("AutoExposureTargetGreyValue") if n.readable("AutoExposureTargetGreyValue") else None,
                "auto_controls": auto_controls,
                "method": "reference camera ExposureAuto/GainAuto Continuous; frozen before dataset sequence"}
        finally:
            self.stop()
            n.enum("ExposureAuto", "Off")
            n.enum("GainAuto", "Off")
            if original_isp is not None:
                n.boolean("IspEnable", original_isp.lower() in ("true", "1"))

    def _release_devices(self):
        primary_error = sys.exc_info()[1]
        errors = []
        try:
            self.stop()
        except Exception as error:
            errors.append(error)
        for device in self.devices:
            for cleanup in (device.nodes.close, device.stream.close,
                lambda d=device: self.api.call("spinCameraDeInit", d.handle),
                lambda d=device: self.api.call("spinCameraRelease", d.handle)):
                try:
                    cleanup()
                except Exception as error:
                    errors.append(error)
        self.devices.clear()
        self.cameras.clear()
        if errors:
            if primary_error is not None:
                if hasattr(primary_error, "add_note"):
                    for error in errors:
                        primary_error.add_note(f"카메라 핸들 정리 오류: {error}")
            else:
                raise errors[0]

    def close(self):
        try:
            self._release_devices()
        finally:
            self.pool.shutdown(wait=True)
            if self.camera_list.value:
                self.api.call("spinCameraListClear", self.camera_list)
                self.api.call("spinCameraListDestroy", self.camera_list)
                self.camera_list = H()
            if self.system.value:
                self.api.call("spinSystemReleaseInstance", self.system)
                self.system = H()
