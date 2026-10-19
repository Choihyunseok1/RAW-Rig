from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import signal
import sys
from threading import Event

from .demo import DemoBackend
from .engine import run_capture
from .models import CaptureConfig, CaptureError
from .network import setup_network
from .storage import verify_dataset


def main(argv=None):
    parser = argparse.ArgumentParser(description="3-camera Bayer RAW dataset capture")
    parser.add_argument("command", nargs="?", default="gui", choices=("gui", "list", "capture", "verify", "network", "nikon"))
    parser.add_argument("path", nargs="?", help="verify: dataset directory")
    parser.add_argument("--demo", action="store_true", help="synthetic cameras; never use as real training data")
    parser.add_argument("--config", help="JSON config file")
    parser.add_argument("--output", help="dataset output root")
    parser.add_argument("--serials", nargs="+", help="selected camera serials in stable order")
    parser.add_argument("--camera-count", type=int, default=None, choices=(1, 2, 3), help="default 3; explicitly use 2 for hardware testing")
    parser.add_argument("--sync", choices=("ptp", "software", "hardware"))
    parser.add_argument("--manual", action="store_true", help="skip auto reference metering; use config absolute presets")
    parser.add_argument("--network-interface", help="camera Ethernet interface; default auto by SDK interface MAC")
    parser.add_argument("--strategy", choices=("bracket", "cross_reference", "per_camera"))
    parser.add_argument("--cross-bracket", action="store_true", help="four settings per reference camera")
    parser.add_argument("--nikon", action="store_true", help="include a USB Nikon Z5 NEF per capture group")
    parser.add_argument("--nikon-port", help="optional USB port, e.g. usb:001,003")
    args = parser.parse_args(argv)
    try:
        config = CaptureConfig.from_dict(json.loads(Path(args.config).read_text(encoding="utf-8"))) if args.config else CaptureConfig()
        overrides = {}
        for key, value in (("output_dir", args.output), ("expected_cameras", args.camera_count),
                           ("sync_mode", args.sync), ("network_interface", args.network_interface),
                           ("capture_strategy", args.strategy), ("dslr_port", args.nikon_port)):
            if value is not None:
                overrides[key] = value
        if args.manual:
            overrides["auto_reference"] = False
        if args.cross_bracket:
            overrides["cross_bracket"] = True
        if args.nikon:
            overrides["dslr_enabled"] = True
        if args.demo and not args.output and not args.config:
            overrides["output_dir"] = "demo_output"
        config = replace(config, **overrides)
        config.validate()
        if args.command == "nikon":
            from .dslr import probe_nikon
            print(json.dumps(probe_nikon(config), ensure_ascii=False, indent=2))
            return 0
        if args.command == "verify":
            if not args.path:
                parser.error("verify requires a dataset path")
            print(json.dumps(verify_dataset(args.path), ensure_ascii=False, indent=2))
            return 0
        if args.command == "gui":
            from .gui import launch_gui
            return launch_gui(args.demo, config, config.expected_cameras)
        if args.demo:
            backend = DemoBackend()
        else:
            from .spinnaker import SpinnakerBackend
            backend = SpinnakerBackend()
        try:
            if args.command == "network":
                cancel = Event()
                old_handler = signal.signal(signal.SIGINT, lambda *_: cancel.set())
                try:
                    report = setup_network(backend, config, cancel,
                        lambda message: print(message, file=sys.stderr, flush=True), args.demo)
                    print(json.dumps(report, ensure_ascii=False, indent=2))
                    return 0 if report["passed"] else 1
                finally:
                    signal.signal(signal.SIGINT, old_handler)
            cameras = backend.discover()
            if args.command == "list":
                print(json.dumps([asdict(c) for c in cameras], ensure_ascii=False, indent=2))
                return 0
            serials = args.serials or [c.serial for c in cameras[:config.expected_cameras]]
            if len(serials) != config.expected_cameras:
                raise CaptureError(f"촬영에는 {config.expected_cameras}대가 필요합니다. 현재 발견한 카메라: {len(cameras)}대")
            backend.open(serials)
            cancel = Event()
            old_handler = signal.signal(signal.SIGINT, lambda *_: cancel.set())
            try:
                result = run_capture(backend, config, cancel, lambda message, done, total: print(f"[{done}/{total}] {message}", flush=True))
                print(json.dumps(verify_dataset(result), ensure_ascii=False, indent=2))
                return 0
            finally:
                signal.signal(signal.SIGINT, old_handler)
        finally:
            backend.close()
    except (CaptureError, OSError, ValueError, TypeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
