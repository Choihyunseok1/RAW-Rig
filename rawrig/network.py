"""GigE address repair with identity checks and fail-closed neighbour probing."""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from ipaddress import IPv4Address, IPv4Interface, IPv4Network
import json
from pathlib import Path
import socket
import subprocess
from threading import Event
import time

from .models import Cancelled, CaptureConfig, CaptureError
from .recovery import is_transient


class NetworkIssue(CaptureError):
    def __init__(self, message, kind, action):
        super().__init__(message)
        self.kind, self.action = kind, action


def network_diagnosis(error, stage):
    if isinstance(error, NetworkIssue):
        return {"category": error.kind, "reason": str(error), "action": error.action, "stage": stage}
    code = getattr(error, "code", None)
    cause = error.__cause__
    if cause is not None and code is None:
        code = getattr(cause, "code", None)
    if code in (-1004, -1005, -1022):
        category, action = "카메라 제어 권한 대기", "SpinView 등에서 해당 카메라의 획득과 장치 연결을 해제하세요."
    elif is_transient(error):
        category, action = "RAW 전송 복구 미완료", "자동 재시도 후에도 수신이 불안정합니다. 카메라 전원·PoE·유선 링크를 확인하세요."
    elif stage == "raw_validation":
        category, action = "RAW 설정 확인 필요", "아래 원인의 지원 형식·설정값을 확인하세요. 카메라 모델과 진단 기록이 자동 저장되었습니다."
    else:
        category, action = "카메라 주소 확인 필요", "아래 원인과 현재 IP를 확인하세요. 진단 기록이 자동 저장되었습니다."
    return {"category": category, "reason": str(error), "action": action, "stage": stage, "sdk_code": code}


@dataclass(frozen=True)
class HostInterface:
    name: str
    mac: str
    address: str
    prefix: int
    up: bool = True
    wired: bool = True

    @property
    def network(self):
        return IPv4Interface(f"{self.address}/{self.prefix}").network


@dataclass(frozen=True)
class NetworkCamera:
    serial: str
    model: str
    mac: str
    ip: str
    mask: str
    interface_mac: str
    access: str = "Unknown"
    compatible: bool = True
    interface_ip_conflict: bool = False


def normalized_mac(value):
    if isinstance(value, int):
        if not 0 < value < 1 << 48:
            raise CaptureError("유효하지 않은 카메라 MAC 주소입니다.")
        value = f"{value:012x}"
    value = str(value).replace(":", "").replace("-", "").lower()
    if len(value) != 12 or any(char not in "0123456789abcdef" for char in value) or int(value, 16) == 0:
        raise CaptureError("유효하지 않은 MAC 주소입니다.")
    return ":".join(value[index:index+2] for index in range(0, 12, 2))


def ipv4(value):
    return str(IPv4Address(value))


def cancelled(event):
    if event.is_set():
        raise Cancelled("카메라 주소 설정을 중지했습니다. 이미 적용된 임시 주소는 유지됩니다.")


def command_json(arguments):
    try:
        result = subprocess.run(arguments, capture_output=True, text=True, check=True, timeout=3)
        return json.loads(result.stdout)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        raise CaptureError(f"네트워크 상태를 확인하지 못했습니다: {error}") from error


def host_interfaces():
    result = []
    for interface in command_json(["ip", "-j", "address", "show"]):
        name = interface["ifname"]
        if interface.get("link_type") != "ether":
            continue
        sys_path = Path("/sys/class/net") / name
        wired = (sys_path / "device").exists() and not (sys_path / "wireless").exists()
        up = "UP" in interface.get("flags", []) and "LOWER_UP" in interface.get("flags", [])
        for address in interface.get("addr_info", []):
            if address["family"] == "inet":
                result.append(HostInterface(name, normalized_mac(interface["address"]),
                    address["local"], address["prefixlen"], up, wired))
    return result


class LinuxNeighbours:
    """Trigger the kernel's ARP resolution without root/raw sockets.

    Silence from ping is never interpreted as a free address. A completed ARP
    FAILED state after an active on-link UDP probe is required. Unknown states
    block allocation; entries with a MAC always reserve the address.
    """
    def __init__(self, timeout=4.):
        self.timeout = timeout

    def entries(self, interface, address):
        return command_json(["ip", "-j", "neigh", "show", "to", address, "dev", interface.name])

    def probe(self, interface, address, event):
        cancelled(event)
        if IPv4Address(address) not in interface.network:
            raise CaptureError("주소 충돌 검사는 같은 서브넷에서만 실행할 수 있습니다.")
        entries = self.entries(interface, address)
        if any(entry.get("lladdr") for entry in entries):
            return "occupied"
        routes = command_json(["ip", "-j", "route", "get", address, "from", interface.address])
        if len(routes) != 1 or routes[0].get("dev") != interface.name or routes[0].get("gateway"):
            return "unknown"
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_DONTROUTE, 1)
                sock.bind((interface.address, 0))
                sock.sendto(b"RAW-RIG-address-probe", (address, 9))
        except OSError:
            return "unknown"
        deadline = time.monotonic() + self.timeout
        saw_pending = False
        while time.monotonic() < deadline:
            cancelled(event)
            entries = self.entries(interface, address)
            for entry in entries:
                if entry.get("lladdr"):
                    return "occupied"
                states = entry.get("state", [])
                if isinstance(states, str):
                    states = [states]
                saw_pending |= "INCOMPLETE" in states
                if "FAILED" in states and saw_pending:
                    return "free"
            event.wait(.1)
        return "unknown"

    def mac(self, interface, address):
        entries = self.entries(interface, address)
        return next((normalized_mac(entry["lladdr"]) for entry in entries if entry.get("lladdr")), None)


def addressing_problem(camera, interface):
    ip = IPv4Address(camera.ip)
    network = interface.network
    try:
        camera_network = IPv4Network(f"{camera.ip}/{camera.mask}", strict=False)
    except ValueError:
        return "서브넷 마스크 오류"
    if ip not in network or ip in (network.network_address, network.broadcast_address) or ip == IPv4Address(interface.address):
        return "PC와 통신할 수 없는 IP"
    if camera_network != network:
        return "PC와 서브넷 마스크 불일치"
    return ""


def candidate_addresses(interface, reserved):
    network = interface.network
    # Bounded even for /8 networks. Small host numbers first are convenient on
    # dedicated camera adapters; every candidate still needs an ARP check.
    stop = min(int(network.broadcast_address), int(network.network_address) + 257)
    for value in range(int(network.network_address) + 2, stop):
        address = str(IPv4Address(value))
        if address not in reserved and address != interface.address:
            yield address


def configure_network(adapter, config: CaptureConfig, event: Event, progress,
                      interfaces=None, neighbours=None):
    report = {"simulated": False, "passed": False, "interfaces": [], "cameras": [], "errors": [],
              "address_mode": "temporary_force_ip", "warnings": [], "scan_attempts": []}
    config.validate()
    host_deadline = time.monotonic() + config.network_rediscover_seconds
    if interfaces is None:
        interfaces = host_interfaces()
        while not any(i.wired and i.up and (not config.network_interface or i.name == config.network_interface) for i in interfaces) and time.monotonic() < host_deadline:
            cancelled(event)
            progress("USB 유선 인터페이스·링크·IPv4 준비 대기 중…")
            event.wait(.5)
            interfaces = host_interfaces()
    report["interfaces"] = [asdict(interface) for interface in interfaces]
    eligible = [interface for interface in interfaces if interface.wired and interface.up
                and (not config.network_interface or interface.name == config.network_interface)]
    if not eligible:
        report["errors"].append("사용할 유선 IPv4 연결이 없습니다. USB 이더넷 어댑터·케이블·링크와 PC IPv4 주소를 확인하세요.")
        return report
    neighbours = neighbours or LinuxNeighbours(config.network_probe_seconds)
    progress("유선 네트워크의 GigE 카메라를 검색합니다.")
    cameras = []
    scan_deadline = time.monotonic() + config.network_rediscover_seconds
    attempt = 0
    while True:
        cancelled(event)
        try:
            cameras = adapter.network_inventory()
            report["scan_attempts"].append({"attempt": attempt + 1, "found": len(cameras)})
            if any(c.interface_mac in {i.mac for i in eligible} for c in cameras):
                break
        except Cancelled:
            raise
        except Exception as error:
            report["scan_attempts"].append({"attempt": attempt + 1, "error": str(error)})
            if not is_transient(error) or time.monotonic() >= scan_deadline:
                report["errors"].append(f"SDK 카메라 검색 실패: {error}")
                return report
        if time.monotonic() >= scan_deadline:
            break
        attempt += 1
        progress(f"카메라 응답 대기 · 인터페이스 갱신 후 자동 재검색 {attempt}")
        event.wait(max(.1, config.retry_delay_ms / 1000))
    reserved = {interface.address for interface in interfaces}
    reserved.update(camera.ip for camera in cameras)
    duplicate_ips = Counter((camera.interface_mac, camera.ip) for camera in cameras)
    serial_macs = {}
    for camera in cameras:
        serial_macs.setdefault(camera.serial, set()).add(camera.mac)
    processed = set()
    for camera in cameras:
        cancelled(event)
        choices = [interface for interface in eligible if interface.mac == camera.interface_mac]
        if not choices:
            continue
        # If a NIC has multiple IPv4 networks, only a unique matching network is
        # automatically selected. Do not guess which subnet to move a camera to.
        matching = [interface for interface in choices if IPv4Address(camera.ip) in interface.network]
        if matching:
            choices = matching
        networks = {(interface.address, interface.prefix) for interface in choices}
        if len(networks) != 1:
            report["errors"].append(f"{camera.mac}: PC 인터페이스에 여러 IPv4 주소가 있어 자동 선택할 수 없습니다.")
            continue
        interface = choices[0]
        if camera.mac in processed:
            continue
        processed.add(camera.mac)
        row = asdict(camera)
        row.update(interface=interface.name, original_ip=camera.ip, allocation_attempted=False,
                   assigned=False, verified=False, error="", warnings=[], attempts=[], stage="address_check")
        report["cameras"].append(row)
        try:
            if camera.interface_ip_conflict:
                raise CaptureError("PC 유선 인터페이스의 IP 충돌이 감지되었습니다. PC 주소를 먼저 확인하세요.")
            if not camera.compatible or not camera.serial:
                raise CaptureError("SDK가 연결 가능한 장치로 식별하지 못했습니다. 주소를 변경하지 않습니다.")
            if len(serial_macs[camera.serial]) != 1:
                raise CaptureError("서로 다른 MAC에 같은 시리얼이 발견되었습니다.")
            reason = addressing_problem(camera, interface)
            if duplicate_ips[(camera.interface_mac, camera.ip)] > 1:
                reason = "카메라 사이의 IP 충돌"
            owner = neighbours.mac(interface, camera.ip) if not reason else None
            if owner and owner != camera.mac:
                reason = "다른 장치와 IP 충돌"
            # OpenReadOnly/OpenReadWrite are handles opened by this GenTL host,
            # not proof of ownership by another application. On a valid IP,
            # CameraInit + setting readback decide actual control access.
            # Never ForceIP an opened or explicitly busy device.
            if camera.access in ("Busy", "ReadOnly") or (reason and camera.access not in ("ReadWrite", "NoAccess", "Unknown")):
                raise NetworkIssue(f"장치 사용 권한: {camera.access}. 다른 프로그램의 연결을 해제하세요.",
                    "카메라 제어 권한 대기", "SpinView 등에서 해당 카메라의 획득과 장치 연결을 해제한 뒤 다시 검사하세요.")
            target = camera.ip
            if reason:
                progress(f"{camera.serial}: {reason} · 빈 주소를 검사합니다.")
                target = None
                unknown = []
                for count, candidate in enumerate(candidate_addresses(interface, reserved)):
                    if count >= 16:
                        break
                    cancelled(event)
                    progress(f"{camera.serial}: 주소 충돌 검사 {count + 1}/16 · {candidate}")
                    status = neighbours.probe(interface, candidate, event)
                    if status == "unknown":
                        unknown.append(candidate)
                        continue
                    if status == "free":
                        target = candidate
                        break
                    reserved.add(candidate)
                if target is None:
                    if unknown:
                        raise NetworkIssue("ARP 충돌 검사를 완료하지 못했습니다. 빈 주소라고 추정하여 할당하지 않습니다.",
                            "주소 충돌 검사 대기", "검증할 수 없는 주소는 건너뛰었습니다. PC 유선 IPv4·라우팅·케이블을 확인하세요.")
                    raise CaptureError("검사한 범위에서 비어 있는 IPv4 주소를 찾지 못했습니다.")
                cancelled(event)
                reserved.add(target)
                # The adapter reselects by MAC and confirms identity immediately
                # before applying the three ForceIP fields as one command.
                progress(f"{camera.serial}: {camera.ip} → {target} 주소 할당")
                row.update(allocation_attempted=True, requested_ip=target, reason=reason, ip="", mask="")
                row["stage"] = "address_assignment"
                try:
                    adapter.force_ip(camera, target, str(interface.network.netmask), "0.0.0.0")
                except Cancelled:
                    raise
                except CaptureError as error:
                    # An ACK can be lost after the camera changed its address.
                    # Confirm the result by identity without issuing ForceIP again.
                    row["warnings"].append(f"주소 명령 응답 미확인: {error}")
                    progress(f"{camera.serial}: 주소 명령 응답 미확인 · MAC으로 적용 결과를 확인합니다.")
            cancelled(event)
            row["stage"] = "rediscovery"
            deadline = time.monotonic() + config.network_rediscover_seconds
            verified_device = None
            while time.monotonic() < deadline:
                cancelled(event)
                try:
                    inventory = adapter.network_inventory()
                except CaptureError as error:
                    if not is_transient(error):
                        raise
                    row["warnings"].append(f"재검색 응답 지연: {error}")
                    event.wait(.25)
                    continue
                identical = [item for item in inventory if item.mac == camera.mac and item.interface_mac == camera.interface_mac]
                if len(identical) == 1:
                    device = identical[0]
                    if device.serial != camera.serial:
                        raise CaptureError("재검색한 장치의 MAC·시리얼 조합이 일치하지 않습니다.")
                    row.update(ip=device.ip, mask=device.mask)
                    if device.ip == target and device.mask == str(interface.network.netmask):
                        contenders = [item for item in inventory if item.interface_mac == camera.interface_mac and item.ip == target]
                        if len({item.mac for item in contenders}) == 1:
                            verified_device = device
                            break
                event.wait(.25)
            if verified_device is None:
                raise CaptureError("재검색에서 요청한 IP·마스크·MAC과 중복 여부를 검증하지 못했습니다.")
            row["assigned"] = row["allocation_attempted"]
            owner = neighbours.mac(interface, target)
            if owner and owner != camera.mac:
                raise CaptureError("할당 후 다른 장치의 MAC이 같은 IP에서 발견되었습니다.")
            progress(f"{camera.serial}: 주소 확인 · 카메라 초기화와 Bayer RAW 수신 검증")
            row["stage"] = "raw_validation"
            for attempt in range(config.auto_retry_count + 1):
                cancelled(event)
                try:
                    validation = adapter.validate_network_camera(verified_device, event)
                    row["attempts"].append({"attempt": attempt + 1, "passed": bool(validation.get("raw_verified"))})
                    break
                except Cancelled:
                    raise
                except Exception as error:
                    row["attempts"].append({"attempt": attempt + 1, "passed": False, "error": str(error)})
                    if not is_transient(error) or attempt == config.auto_retry_count:
                        raise
                    progress(f"{camera.serial}: RAW 수신 자동 복구 {attempt + 1}/{config.auto_retry_count} · {error}")
                    event.wait(config.retry_delay_ms / 1000)
            if not validation.get("raw_verified"):
                raise CaptureError("Bayer RAW 수신 검증을 통과하지 못했습니다.")
            row.update(ip=verified_device.ip, mask=verified_device.mask, verified=True, validation=validation, stage="ready")
            row["recovered"] = len(row["attempts"]) > 1 or bool(row["warnings"])
            progress(f"{camera.serial}: {row['ip']} · MAC·주소·RAW 검증 통과")
        except Cancelled:
            raise
        except Exception as error:
            row["error"] = str(error)
            row["diagnosis"] = network_diagnosis(error, row["stage"])
            report["errors"].append(f"{camera.serial or camera.mac}: {error}")
            progress(f"{camera.serial or camera.mac}: {row['diagnosis']['category']} · {error}")
    if not report["cameras"] and not report["errors"]:
        report["errors"].append("선택한 유선 네트워크에서 GigE 카메라를 찾지 못했습니다. 카메라 전원·PoE·케이블을 확인하세요.")
    report["passed"] = bool(report["cameras"]) and not report["errors"] and all(row["verified"] for row in report["cameras"])
    return report


class DemoNetworkAdapter:
    """Explicitly synthetic addresses; no operating-system network writes."""
    interface = HostInterface("demo_eth0", "02:00:00:00:10:01", "10.42.0.1", 24)

    def __init__(self, backend):
        from dataclasses import replace
        self.backend = backend
        self.replace = replace
        self.devices = [NetworkCamera(camera.serial, camera.model, f"02:00:00:00:20:0{index+1}",
            "192.168.1.10" if index == 2 else f"10.42.0.{125-index}", "255.255.255.0",
            self.interface.mac, "ReadWrite") for index, camera in enumerate(backend.discover())]

    def network_inventory(self):
        return list(self.devices)

    def force_ip(self, camera, address, mask, gateway):
        self.devices = [self.replace(device, ip=address, mask=mask) if device.mac == camera.mac else device for device in self.devices]

    def probe(self, interface, address, event):
        cancelled(event)
        return "occupied" if any(device.ip == address for device in self.devices) else "free"

    def mac(self, interface, address):
        return next((device.mac for device in self.devices if device.ip == address), None)

    def validate_network_camera(self, camera, event):
        from .models import Preset, validate_frame
        self.backend.open([camera.serial])
        try:
            preset = Preset("network_validation", 10000., 0.)
            self.backend.prepare(preset, CaptureConfig(expected_cameras=1, sync_mode="software", auto_reference=False))
            self.backend.start()
            frames, _ = self.backend.capture(3000, event)
            validate_frame(frames[0], preset)
            return {"raw_verified": True, "simulated": True, "pixel_format": frames[0].pixel_format,
                "shape": list(frames[0].pixels.shape), "ptp_supported": True, "scheduled_action_supported": True}
        finally:
            self.backend.close()


def setup_network(backend, config, event, progress, demo=False):
    if demo:
        from dataclasses import replace
        adapter = DemoNetworkAdapter(backend)
        report = configure_network(adapter, replace(config, network_interface=""), event,
                                   progress, [adapter.interface], adapter)
        report["simulated"] = True
        return report
    return configure_network(backend, config, event, progress)
