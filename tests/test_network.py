from contextlib import contextmanager
from dataclasses import replace
from ipaddress import IPv4Address
from threading import Event

import pytest

from rawrig import network
from rawrig.models import CaptureConfig, CaptureError, Cancelled, RetryableCaptureError
from rawrig.network import HostInterface, NetworkCamera, configure_network, LinuxNeighbours
from rawrig.spinnaker import SpinnakerBackend, H


HOST = HostInterface("eth_camera", "02:00:00:00:10:01", "10.42.0.1", 24)


def camera(index, ip, **values):
    return NetworkCamera(f"camera_{index}", "Blackfly S", f"02:00:00:00:20:{index:02x}",
        ip, values.pop("mask", "255.255.255.0"), HOST.mac, **values)


class Adapter:
    def __init__(self, cameras):
        self.cameras = cameras
        self.writes, self.validated = [], []
        self.bad_readback = False
        self.fail_raw = False

    def network_inventory(self):
        return list(self.cameras)

    def force_ip(self, cam, address, mask, gateway):
        self.writes.append((cam.mac, address, mask, gateway))
        self.cameras = [replace(item, ip="10.42.0.230" if self.bad_readback else address, mask=mask)
                        if item.mac == cam.mac else item for item in self.cameras]

    def validate_network_camera(self, cam, event):
        self.validated.append(cam)
        if self.fail_raw:
            raise CaptureError("RAW CRC mismatch")
        return {"raw_verified": True, "pixel_format": "BayerRG16"}


class Neighbours:
    def __init__(self, occupied=None, unknown=False):
        self.occupied = occupied or {}
        self.unknown = unknown

    def mac(self, interface, address):
        return self.occupied.get(address)

    def probe(self, interface, address, event):
        return "unknown" if self.unknown else "occupied" if address in self.occupied else "free"


def run(adapter, neighbours=None, interfaces=None, event=None, config=None):
    return configure_network(adapter, config or CaptureConfig(network_rediscover_seconds=2),
        event or Event(), lambda _: None, [HOST] if interfaces is None else interfaces, neighbours or Neighbours())


def test_new_wrong_subnet_camera_skips_occupied_address_preserves_existing():
    old, new = camera(1, "10.42.0.125"), camera(2, "192.168.10.20")
    adapter = Adapter([old, new])
    report = run(adapter, Neighbours({"10.42.0.2": "02:00:00:00:30:01"}))
    assert report["passed"]
    assert adapter.writes == [(new.mac, "10.42.0.3", "255.255.255.0", "0.0.0.0")]
    assert adapter.cameras[0] == old
    assert [item.ip for item in adapter.validated] == ["10.42.0.125", "10.42.0.3"]
    assert report["cameras"][1]["original_ip"] == "192.168.10.20"
    assert report["cameras"][1]["assigned"] and report["cameras"][1]["verified"]


def test_duplicate_camera_ips_become_distinct_by_mac():
    adapter = Adapter([camera(1, "10.42.0.125"), camera(2, "10.42.0.125")])
    report = run(adapter)
    assert report["passed"]
    assert len({item.ip for item in adapter.cameras}) == 2
    assert len({write[0] for write in adapter.writes}) == 2


def test_conflict_with_other_device_detected_by_mac():
    cam = camera(1, "10.42.0.125")
    adapter = Adapter([cam])
    report = run(adapter, Neighbours({cam.ip: "02:00:00:00:99:01"}))
    assert report["passed"]
    assert report["cameras"][0]["reason"] == "다른 장치와 IP 충돌"
    assert adapter.writes[0][1] != cam.ip


@pytest.mark.parametrize("interface", [replace(HOST, wired=False), replace(HOST, up=False)])
def test_missing_wired_link_fails_without_writes(interface):
    adapter = Adapter([camera(1, "192.168.1.10")])
    report = run(adapter, interfaces=[interface])
    assert not report["passed"]
    assert "유선" in report["errors"][0]
    assert not adapter.writes and not adapter.validated


def test_arp_unknown_never_means_free():
    adapter = Adapter([camera(1, "192.168.1.10")])
    report = run(adapter, Neighbours(unknown=True))
    assert not report["passed"] and not adapter.writes
    assert "ARP" in report["errors"][0]


@pytest.mark.parametrize("access", ["Busy", "OpenReadWrite", "OpenReadOnly", "ReadOnly"])
def test_device_owned_by_another_program_not_readdressed(access):
    adapter = Adapter([camera(1, "192.168.1.10", access=access)])
    report = run(adapter)
    assert not report["passed"] and not adapter.writes and not adapter.validated


def test_ip_command_success_with_wrong_readback_is_failure():
    adapter = Adapter([camera(1, "192.168.1.10")])
    adapter.bad_readback = True
    report = run(adapter)
    assert not report["passed"] and not adapter.validated
    row = report["cameras"][0]
    assert row["allocation_attempted"] and not row["assigned"] and not row["verified"]
    assert row["requested_ip"] == "10.42.0.2"
    assert "재검색" in row["error"]


def test_raw_receive_failure_does_not_mark_camera_ready():
    adapter = Adapter([camera(1, "10.42.0.125")])
    adapter.fail_raw = True
    report = run(adapter)
    assert not report["passed"] and not report["cameras"][0]["verified"]
    assert "CRC" in report["errors"][0]


def test_multiple_host_subnets_does_not_guess_target():
    adapter = Adapter([camera(1, "192.168.1.10")])
    report = run(adapter, interfaces=[HOST, replace(HOST, address="10.43.0.1")])
    assert not report["passed"] and not adapter.writes
    assert "여러 IPv4" in report["errors"][0]


def test_cancelled_before_assignment():
    adapter = Adapter([camera(1, "192.168.1.10")])
    event = Event()
    event.set()
    with pytest.raises(Cancelled):
        run(adapter, event=event)
    assert not adapter.writes


def test_unsupported_device_not_mutated():
    adapter = Adapter([camera(1, "192.168.1.10", compatible=False)])
    report = run(adapter)
    assert not report["passed"] and not adapter.writes


def test_duplicate_serials_block_all_writes():
    adapter = Adapter([camera(1, "192.168.1.10"), replace(camera(2, "192.168.1.11"), serial="camera_1")])
    report = run(adapter)
    assert not report["passed"] and not adapter.writes
    assert len(report["errors"]) == 2


def test_pc_ip_conflict_blocks_camera_assignment():
    adapter = Adapter([camera(1, "192.168.1.10", interface_ip_conflict=True)])
    report = run(adapter)
    assert not report["passed"] and not adapter.writes
    assert "PC 유선 인터페이스" in report["errors"][0]


def test_lost_forceip_ack_confirms_by_mac_without_reissuing_command():
    class LostAck(Adapter):
        def force_ip(self, *args):
            super().force_ip(*args)
            raise CaptureError("ForceIP ACK timeout")
    adapter = LostAck([camera(1, "192.168.1.10")])
    report = run(adapter)
    row = report["cameras"][0]
    assert report["passed"] and row["verified"]
    assert row["allocation_attempted"] and row["assigned"] and row["recovered"]
    assert row["requested_ip"] == row["ip"] == "10.42.0.2"
    assert adapter.cameras[0].ip == "10.42.0.2"
    assert len(adapter.writes) == 1 and row["warnings"]


@pytest.mark.parametrize("access", ["OpenReadOnly", "OpenReadWrite"])
def test_open_status_on_valid_ip_is_checked_by_real_control_access(access):
    adapter = Adapter([camera(1, "10.42.0.125", access=access)])
    report = run(adapter)
    assert report["passed"] and len(adapter.validated) == 1
    assert not adapter.writes


def test_transient_raw_failure_recovers_without_user_or_address_changes():
    class Flaky(Adapter):
        failures = 2
        def validate_network_camera(self, cam, event):
            if self.failures:
                self.failures -= 1
                raise RetryableCaptureError("RAW CRC mismatch")
            return super().validate_network_camera(cam, event)
    adapter = Flaky([camera(1, "10.42.0.125")])
    report = run(adapter, config=CaptureConfig(retry_delay_ms=0))
    row = report["cameras"][0]
    assert report["passed"] and row["recovered"] and len(row["attempts"]) == 3
    assert not adapter.writes and not report["errors"]


def test_persistent_transport_problem_has_actionable_diagnosis():
    class Broken(Adapter):
        def validate_network_camera(self, cam, event):
            self.validated.append(cam)
            raise RetryableCaptureError("RAW CRC mismatch")
    adapter = Broken([camera(1, "10.42.0.125")])
    report = run(adapter, config=CaptureConfig(auto_retry_count=1, retry_delay_ms=0))
    row = report["cameras"][0]
    assert not report["passed"] and not row["verified"]
    assert len(adapter.validated) == len(row["attempts"]) == 2
    assert row["diagnosis"]["category"] == "RAW 전송 복구 미완료"
    assert "전원" in row["diagnosis"]["action"]


def test_unknown_address_is_skipped_while_verified_free_address_can_be_used():
    class PartialProbe(Neighbours):
        def probe(self, interface, address, event):
            return "unknown" if address == "10.42.0.2" else "free"
    adapter = Adapter([camera(1, "192.168.1.10")])
    report = run(adapter, PartialProbe())
    assert report["passed"] and adapter.writes[0][1] == "10.42.0.3"


def test_camera_powerup_uses_bounded_automatic_rescan():
    class Booting(Adapter):
        reads = 0
        def network_inventory(self):
            self.reads += 1
            return [] if self.reads <= 2 else super().network_inventory()
    adapter = Booting([camera(1, "10.42.0.125")])
    report = run(adapter, config=CaptureConfig(retry_delay_ms=0))
    assert report["passed"] and [r["found"] for r in report["scan_attempts"]] == [0, 0, 1]


def test_stale_failed_neighbour_alone_is_not_fresh_proof(monkeypatch):
    class Socket:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def setsockopt(self, *args): pass
        def bind(self, address): pass
        def sendto(self, payload, target): pass
    monkeypatch.setattr(network, "command_json", lambda args: [{"dev": HOST.name}] if "route" in args else [{"state": ["FAILED"]}])
    monkeypatch.setattr(network.socket, "socket", lambda *args: Socket())
    assert LinuxNeighbours(timeout=.01).probe(HOST, "10.42.0.3", Event()) == "unknown"


def test_forceip_targets_mac_after_sdk_device_order_changes():
    target = camera(2, "192.168.1.10")
    class Nodes:
        selected = 0
        fields = {}
        commands = []
        def readable(self, name): return True
        def writable(self, name): return True
        def integer(self, name, value):
            if name == "DeviceSelector":
                self.selected = value
            else:
                self.fields[name] = value
        def get_int(self, name):
            if name == "GevInterfaceMACAddress":
                return int(HOST.mac.replace(":", ""), 16)
            if name == "DeviceCount":
                return 2
            return int((camera(1, "10.42.0.125").mac if self.selected == 0 else target.mac).replace(":", ""), 16)
        def text(self, name):
            return ("camera_1" if self.selected == 0 else target.serial) if name == "DeviceSerialNumber" else "NoAccess"
        def execute(self, name): self.commands.append(name)
    nodes = Nodes()
    backend = object.__new__(SpinnakerBackend)
    @contextmanager
    def interfaces(): yield [nodes]
    backend._network_interfaces = interfaces
    backend.force_ip(target, "10.42.0.3", "255.255.255.0", "0.0.0.0")
    assert nodes.selected == 1
    assert nodes.fields == {"GevDeviceForceIPAddress": int(IPv4Address("10.42.0.3")),
        "GevDeviceForceSubnetMask": 0xFFFFFF00, "GevDeviceForceGateway": 0}
    assert nodes.commands[-1] == "GevDeviceForceIP"


def test_sdk_interface_references_released_on_failure():
    class API:
        calls = []
        def handle(self, name, *args):
            return H(10 + args[1]) if name == "spinInterfaceListGet" else H(1)
        def value(self, *args): return 2
        def call(self, name, *args): self.calls.append((name, args))
    backend = object.__new__(SpinnakerBackend)
    backend.api, backend.devices, backend.system = API(), [], H(2)
    with pytest.raises(ValueError):
        with backend._network_interfaces():
            raise ValueError("simulated command failure")
    released = [args[0].value for name, args in backend.api.calls if name == "spinInterfaceRelease"]
    assert released == [11, 10]
    assert [name for name, _ in backend.api.calls][-2:] == ["spinInterfaceListClear", "spinInterfaceListDestroy"]


def test_linux_arp_requires_resolution_failure_or_known_owner(monkeypatch):
    entries = iter([[], [{"state": ["INCOMPLETE"]}], [{"state": ["FAILED"]}]])
    sent = []
    class Socket:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def setsockopt(self, *args): pass
        def bind(self, address): assert address[0] == HOST.address
        def sendto(self, payload, target): sent.append(target)
    def commands(args):
        return [{"dev": HOST.name}] if "route" in args else next(entries)
    monkeypatch.setattr(network, "command_json", commands)
    monkeypatch.setattr(network.socket, "socket", lambda *args: Socket())
    assert LinuxNeighbours().probe(HOST, "10.42.0.3", Event()) == "free"
    assert sent == [("10.42.0.3", 9)]


def test_linux_arp_route_to_other_interface_blocks_probe(monkeypatch):
    monkeypatch.setattr(network, "command_json", lambda args: [{"dev": "wrong_eth"}] if "route" in args else [])
    monkeypatch.setattr(network.socket, "socket", lambda *args: pytest.fail("wrong interface must not send"))
    assert LinuxNeighbours().probe(HOST, "10.42.0.3", Event()) == "unknown"


def test_late_hotplug_empty_inventory_is_retried_beyond_raw_retry_count():
    class Hotplug(Adapter):
        def __init__(self):
            super().__init__([camera(1, '10.42.0.125')])
            self.scans = 0
        def network_inventory(self):
            self.scans += 1
            return [] if self.scans < 4 else super().network_inventory()
    adapter = Hotplug()
    report = run(adapter, config=CaptureConfig(auto_retry_count=0, retry_delay_ms=0, network_rediscover_seconds=2))
    assert report['passed'] and len(report['scan_attempts']) == 4
    assert not adapter.writes and len(adapter.validated) == 1


def test_host_ipv4_hotplug_is_waited_for_before_sdk_search(monkeypatch):
    calls = []
    snapshots = iter([[], [replace(HOST, up=False)], [HOST]])
    monkeypatch.setattr(network, 'host_interfaces', lambda: next(snapshots))
    class FastEvent(Event):
        def wait(self, seconds):
            calls.append(seconds)
            return self.is_set()
    adapter = Adapter([camera(1, '10.42.0.125')])
    report = configure_network(adapter, CaptureConfig(network_rediscover_seconds=2), FastEvent(), lambda _:None, neighbours=Neighbours())
    assert report['passed'] and len(calls) == 2 and report['interfaces'][0]['up']


def test_hotplug_cancel_is_prompt_and_never_changes_an_address():
    event = Event()
    class Missing(Adapter):
        def network_inventory(self):
            event.set()
            return []
    adapter = Missing([])
    with pytest.raises(Cancelled):
        run(adapter, event=event)
    assert not adapter.writes and not adapter.validated
