from threading import Event
import time

import numpy as np
import pytest

from rawrig.demo import DemoBackend
from rawrig.engine import run_capture
from rawrig.models import (CameraInfo, CaptureConfig, CaptureError, Cancelled,
                           WhiteBalanceRestoreError)
from rawrig.spinnaker import SpinnakerBackend, Device, H, SDKError
from rawrig.worker import CameraWorker


class Nodes:
    def __init__(self, mode):
        self.mode = mode
        self.values = {'IspEnable': 'False', 'TriggerMode': 'On', 'BalanceRatioSelector': 'Blue',
            'BalanceWhiteAutoProfile': 'Outdoor', 'BalanceWhiteAuto': 'Off',
            'ExposureAuto': 'Off', 'GainAuto': 'Off', 'ExposureTime': 10000., 'Gain': 3.}
        self.ratios = {'Red': 1., 'Blue': 1.}
        self.writes = []

    def readable(self, name):
        return name in self.values and not (self.mode == 'delayed_nodes' and name.startswith('Balance')
                                           and self.values['IspEnable'] == 'False')
    def text(self, name): return self.values[name]
    def enum_supported(self, name, value):
        if self.mode == 'delayed_nodes' and name.startswith('Balance') and self.values['IspEnable'] == 'False':
            return False
        return not (self.mode == 'unsupported' and name == 'BalanceWhiteAuto' and value == 'Once')
    def enum(self, name, value, optional=False):
        self.values[name] = value
        self.writes.append((name, value))
    def boolean(self, name, value):
        self.values[name] = str(value)
        self.writes.append((name, value))
    def floating(self, name, value):
        channel = self.values['BalanceRatioSelector']
        if self.mode == 'restore_failed' and channel == 'Red' and value == 1.:
            raise CaptureError('BalanceRatio restore failure')
        self.ratios[channel] = value
    def get_float(self, name):
        return self.ratios[self.values['BalanceRatioSelector']] if name == 'BalanceRatio' else self.values[name]


class API:
    def __init__(self, nodes, cancel):
        self.nodes, self.cancel = nodes, cancel
        self.frames, self.released = 0, 0
    def call(self, name, *args):
        if name == 'spinCameraGetNextImageEx':
            self.frames += 1
            args[-1]._obj.value = self.frames
            if self.nodes.mode == 'sdk_error':
                raise SDKError(name, -1010, 'IO failure')
            if self.nodes.mode == 'cancelled':
                self.cancel.set()
            if self.nodes.mode == 'timeout':
                time.sleep(.01)
            elif self.frames >= 2:
                self.nodes.values['BalanceWhiteAuto'] = 'Off'
                self.nodes.ratios = {'Red': 0. if self.nodes.mode == 'bad_ratio' else 1.8, 'Blue': 2.3}
        elif name == 'spinImageRelease':
            self.released += 1
    def value(self, name, *args):
        assert name == 'spinImageIsIncomplete'
        return False


def sdk_rig(mode):
    backend = object.__new__(SpinnakerBackend)
    camera = CameraInfo('TEST_1', 'Test camera', True)
    event = Event()
    nodes = Nodes(mode)
    backend.cameras = [camera]
    backend.api = API(nodes, event)
    backend.devices = [Device(camera, H(1), nodes, None, {})]
    def stop():
        for device in backend.devices:
            device.running = False
    backend.stop = stop
    return backend, nodes, event


@pytest.mark.parametrize('mode', ['success', 'delayed_nodes'])
def test_sdk_uses_once_only_and_restores_raw_with_exposure_and_gain_unchanged(mode):
    backend, nodes, event = sdk_rig(mode)
    report = backend.white_balance_once(CaptureConfig(white_balance_reference_serial='TEST_1'), event)
    assert (report['red'], report['blue']) == (1.8, 2.3)
    assert report['samples'] == 2 and report['raw_restored'] and report['camera_auto'] == 'Off'
    assert ('BalanceWhiteAuto', 'Once') in nodes.writes
    assert not any(value == 'Continuous' for _, value in nodes.writes)
    assert nodes.values['BalanceWhiteAuto'] == 'Off'
    assert nodes.values['IspEnable'] == 'False'
    assert nodes.values['TriggerMode'] == 'On'
    assert nodes.values['BalanceWhiteAutoProfile'] == 'Outdoor'
    assert nodes.values['BalanceRatioSelector'] == 'Blue'
    assert nodes.ratios == {'Red': 1., 'Blue': 1.}
    assert (nodes.values['ExposureTime'], nodes.values['Gain']) == (10000., 3.)
    assert backend.api.frames == backend.api.released
    assert not backend.devices[0].running


@pytest.mark.parametrize('mode,error', [('cancelled', Cancelled), ('timeout', CaptureError),
    ('sdk_error', SDKError), ('bad_ratio', CaptureError), ('unsupported', CaptureError)])
def test_sdk_failure_and_cancel_leave_auto_off_neutral_raw_and_release_every_image(mode, error):
    backend, nodes, event = sdk_rig(mode)
    with pytest.raises(error):
        backend.white_balance_once(CaptureConfig(white_balance_timeout_seconds=1), event)
    assert nodes.values['BalanceWhiteAuto'] == 'Off'
    assert nodes.values['IspEnable'] == 'False' and nodes.values['TriggerMode'] == 'On'
    assert nodes.ratios == {'Red': 1., 'Blue': 1.}
    assert backend.api.frames == backend.api.released
    assert not backend.devices[0].running


def test_restore_failure_still_restores_remaining_camera_controls():
    backend, nodes, event = sdk_rig('restore_failed')
    with pytest.raises(WhiteBalanceRestoreError, match='restore failure'):
        backend.white_balance_once(CaptureConfig(), event)
    assert nodes.values['BalanceWhiteAuto'] == 'Off'
    assert nodes.values['IspEnable'] == 'False' and nodes.values['TriggerMode'] == 'On'
    assert nodes.values['BalanceWhiteAutoProfile'] == 'Outdoor'
    assert nodes.ratios['Blue'] == 1.
    assert backend.api.frames == backend.api.released


def test_worker_once_applies_identical_gains_and_rerenders_paused_images_without_touching_raw():
    worker = CameraWorker(demo=True)
    worker.backend = DemoBackend()
    worker.backend.open(['DEMO_1', 'DEMO_2', 'DEMO_3'])
    worker._preview_once()
    originals = [frame.pixels.copy() for frame in worker.last_frames]
    frames = worker.backend.counter
    worker.preview_enabled = False
    worker.display_options = (2., 3., .4)
    reports, images = [], []
    worker.white_balance_result.connect(reports.append)
    worker.images.connect(images.append)
    worker._handle('white_balance', (CaptureConfig(white_balance_reference_serial='DEMO_2'), Event()))
    assert len(reports) == 1 and len(images) == 1 and len(images[0]) == 3
    assert reports[0]['reference_serial'] == 'DEMO_2'
    assert reports[0]['applied_serials'] == ['DEMO_1', 'DEMO_2', 'DEMO_3']
    assert worker.display_options == (1.1765, 1.5385, .4)
    assert not worker.preview_enabled and worker.backend.counter == frames
    for frame, original in zip(worker.last_frames, originals):
        np.testing.assert_array_equal(frame.pixels, original)
    worker.backend.close()


def test_worker_restore_failure_disconnects_rig_before_any_more_raw_acquisition():
    worker = CameraWorker(demo=True)
    worker.backend = DemoBackend()
    worker.backend.open(['DEMO_1', 'DEMO_2', 'DEMO_3'])
    previous = worker.display_options
    def broken(*args): raise WhiteBalanceRestoreError('Unsafe RAW state')
    worker.backend.white_balance_once = broken
    connected = []
    worker.connected.connect(connected.append)
    with pytest.raises(WhiteBalanceRestoreError):
        worker._handle('white_balance', (CaptureConfig(), Event()))
    assert worker.backend is None and connected == [[]]
    assert worker.display_options == previous


def test_calibration_is_recorded_in_dataset_without_changing_stored_bayer_pixels(tmp_path):
    import json
    from rawrig.storage import verify_dataset
    backend = DemoBackend()
    backend.open(['DEMO_1', 'DEMO_2', 'DEMO_3'])
    calibration = backend.white_balance_once(CaptureConfig(white_balance_reference_serial='DEMO_2'), Event())
    originals = []
    config = CaptureConfig(output_dir=str(tmp_path), auto_reference=False, sync_mode='software',
        warmup_frames=0, settle_ms=0, preview_red=calibration['red'], preview_blue=calibration['blue'],
        white_balance_calibration=calibration)
    result = run_capture(backend, config, preview=originals.extend)
    manifest = json.loads((result / 'manifest.json').read_text())
    assert manifest['config']['white_balance_calibration'] == calibration
    assert verify_dataset(result)['frames'] == 12
    for record, frame in zip(manifest['frames'], originals):
        metadata = json.loads((result / record['metadata']).read_text())
        saved = np.fromfile(result / record['raw'], dtype=metadata['dtype']).reshape(metadata['shape'])
        np.testing.assert_array_equal(saved, frame.pixels)
    backend.close()


@pytest.mark.parametrize('values', [{'white_balance_timeout_seconds': 0}, {'white_balance_timeout_seconds': float('nan')},
    {'white_balance_profile': 'Continuous'}, {'white_balance_reference_serial': 'wrong/serial'},
    {'white_balance_calibration': []}])
def test_invalid_white_balance_config_is_rejected(values):
    with pytest.raises(CaptureError):
        CaptureConfig(**values).validate()
