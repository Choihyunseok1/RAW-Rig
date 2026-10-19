import json
import os
from pathlib import Path
import time

os.environ["QT_QPA_PLATFORM"] = "offscreen"

import pytest
from PySide6.QtCore import Qt, QPoint, QPointF
from PySide6.QtGui import QWheelEvent
from PySide6.QtTest import QTest, QSignalSpy
from PySide6.QtWidgets import QApplication, QPushButton, QComboBox, QDoubleSpinBox, QSpinBox, QScrollArea

from rawrig.gui import MainWindow
from rawrig.models import CaptureConfig, Preset
from rawrig.storage import verify_dataset
from rawrig.widgets import ScrollSafeComboBox, ScrollSafeDoubleSpinBox, ScrollSafeSpinBox


def wait_until(predicate, timeout=12):
    end = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= end:
            raise AssertionError("Qt operation did not finish")
        QApplication.processEvents()
        time.sleep(.01)  # Let the Python camera worker run while Qt keeps processing UI events.


@pytest.fixture
def window():
    app = QApplication.instance() or QApplication([])
    window = MainWindow(demo=True)
    window.show()
    wait_until(lambda: window.camera_list.count() == 3 and not window.busy)
    yield window
    window.worker.request_shutdown()
    wait_until(lambda: not window.worker.isRunning())
    window.close()
    app.processEvents()


def connect(window):
    QTest.mouseClick(window.connect_button, Qt.MouseButton.LeftButton)
    wait_until(lambda: len(window.cameras) == 3 and not window.busy and window.cards[0].saved_image is not None)


def test_resize_click_zoom_and_fullscreen_restore(window):
    connect(window)
    QTest.qWait(150)
    assert window.overview.columns == 2
    assert window.pages.currentWidget() is window.workspace
    assert window.workspace.size() == window.pages.size()
    assert window.inspector_scroll.horizontalScrollBar().maximum() == 0
    QTest.mouseClick(window.cards[0].image, Qt.MouseButton.LeftButton)
    QTest.qWait(60)
    assert window.zoomed_index == 0
    assert window.focus_image.image is not None
    assert not window.inspector_scroll.isVisible()
    QTest.mouseClick(window.focus_image, Qt.MouseButton.LeftButton)
    assert window.zoomed_index is None
    assert window.inspector_scroll.isVisible()
    original_size = window.size()
    QTest.mouseClick(window.fullscreen_button, Qt.MouseButton.LeftButton)
    QTest.qWait(60)
    assert window.isFullScreen()
    QTest.keyClick(window, Qt.Key.Key_Escape)
    QTest.qWait(60)
    assert not window.isFullScreen()
    assert window.size() == original_size
    window.resize(900, 700)
    QTest.qWait(150)
    assert window.overview.columns == 2
    assert window.sidebar.width() == 72
    assert not window.inspector_scroll.isVisible()
    window.resize(640, 520)
    QTest.qWait(150)
    assert window.overview.columns == 2
    for card in window.cards:
        assert window.overview.rect().contains(card.geometry())


def test_buttons_have_visible_hover_and_help(window):
    connect(window)
    window.resize(740, 650)
    QTest.qWait(100)
    assert all(button.toolTip() for button in window.findChildren(QPushButton))
    for button in (window.fullscreen_button, window.nav_buttons[0], window.nav_buttons[1], window.capture_button, window.network_button):
        if button is window.network_button:
            QTest.mouseClick(window.inspector_toggle, Qt.MouseButton.LeftButton)
            window.inspector_scroll.ensureWidgetVisible(button)
            QTest.qWait(80)
        QTest.mouseMove(window, QPoint(10, 10))
        QTest.qWait(30)
        before = button.grab().toImage().pixelColor(5, button.height() // 2)
        QTest.mouseMove(button, button.rect().center())
        QTest.qWait(30)
        after = button.grab().toImage().pixelColor(5, button.height() // 2)
        assert before != after, button.text()


def test_network_button_assigns_new_camera_and_verifies_then_connects(window):
    # Address setup is independent of an unfinished capture preset name.
    window.preset_editors[0][0].setText("")
    QTest.mouseClick(window.network_button, Qt.MouseButton.LeftButton)
    wait_until(lambda: window.last_network_report is not None and not window.busy)
    report = window.last_network_report
    assert report["passed"] and report["simulated"]
    assert len(window.cameras) == 3
    assert report["connected_serials"] == [camera.serial for camera in window.cameras]
    changed = [row for row in report["cameras"] if row["assigned"]]
    assert len(changed) == 1
    assert changed[0]["original_ip"] == "192.168.1.10"
    assert changed[0]["ip"] == "10.42.0.2"
    assert all(row["verified"] and row["validation"]["raw_verified"] for row in report["cameras"])
    assert window.capture_button.isEnabled()
    assert window.network_save.isEnabled()
    assert "10.42.0.2" in window.network_report_view.toPlainText()
    assert window.network_table.rowCount() == 3
    assert window.network_table.item(2, 4).text() == "10.42.0.2"
    assert Path(report["diagnostic_path"]).is_file()


def test_network_failure_explains_action_and_keeps_healthy_cameras_connected(window, monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    from rawrig.network import DemoNetworkAdapter
    original = DemoNetworkAdapter.validate_network_camera
    def unsupported(self, camera, event):
        if camera.serial == "DEMO_3":
            from rawrig.models import CaptureError
            raise CaptureError("Bayer16 출력 미지원")
        return original(self, camera, event)
    monkeypatch.setattr(DemoNetworkAdapter, "validate_network_camera", unsupported)
    QTest.mouseClick(window.network_button, Qt.MouseButton.LeftButton)
    wait_until(lambda: window.last_network_report is not None and not window.busy)
    report = window.last_network_report
    assert not report["passed"] and len(window.cameras) == 2
    assert "다음 조치:" in window.network_status.text()
    assert "Bayer16 출력 미지원" in window.network_table.item(2, 5).text()
    failed = next(window.camera_list.item(i) for i in range(window.camera_list.count())
        if window.camera_list.item(i).data(Qt.ItemDataRole.UserRole) == "DEMO_3")
    assert "RAW 설정 확인 필요" in failed.text() and "검사 실패" not in failed.text()
    assert failed.checkState() == Qt.CheckState.Unchecked
    assert Path(report["diagnostic_path"]).is_file()
    assert "진단 자동 저장:" in window.network_summary.text()


def test_new_cameras_get_preview_inside_common_limits(window):
    window.worker.preview_preset = Preset("old_preview", 50000000., 60.)
    QTest.mouseClick(window.network_button, Qt.MouseButton.LeftButton)
    wait_until(lambda: window.last_network_report is not None and not window.busy)
    assert window.last_network_report["passed"]
    assert window.worker.preview_preset.exposure_us == 30000000.
    assert window.worker.preview_preset.gain_db == 48.
    wait_until(lambda: window.cards[0].saved_image is not None)


def test_all_config_fields_roundtrip_and_independent_manual_presets(window, tmp_path):
    config = CaptureConfig(output_dir=str(tmp_path / "raw"), expected_cameras=2, repeats=3,
        sync_mode="hardware", master_serial="DEMO_2", reference_serial="DEMO_1",
        output_line="Line3", input_line="Line1", trigger_edge="FallingEdge", settle_ms=320,
        warmup_frames=2, timeout_ms=6200, preview_red=1.4, preview_blue=1.6,
        preview_brightness_ev=2.5, preview_fps=6.5, auto_reference=False, meter_seconds=2.5,
        low_exposure_factor=.5, high_exposure_factor=3., low_gain_offset_db=-2., high_gain_offset_db=7.,
        ptp_wait_seconds=22., ptp_max_offset_us=8.5, ptp_max_timestamp_spread_us=75., action_lead_ms=340,
        presets=[Preset(f"custom_{i}", 1234.5 * (i+1), 1.25 * i) for i in range(4)])
    window._load_config(config)
    assert window._config().to_dict() == config.to_dict()
    unchanged = window._editor_presets()[1:]
    window.preset_editors[0][1].setValue(1789.5)
    assert window.base_exp.value() == 1789.5
    assert window._editor_presets()[1:] == unchanged
    path = tmp_path / "settings.json"
    saved = window.save_settings(path).to_dict()
    window._load_config(CaptureConfig())
    window.load_settings(path)
    assert window._config().to_dict() == saved


def test_one_click_auto_capture_uses_baseline_as_setting_one_and_verifies(window, tmp_path):
    config = CaptureConfig(output_dir=str(tmp_path), meter_seconds=.5, settle_ms=0, warmup_frames=0)
    window._load_config(config)
    connect(window)
    window.preset_editors[0][0].setText("custom_baseline")
    assert window.capture_button.isEnabled()
    QTest.mouseClick(window.capture_button, Qt.MouseButton.LeftButton)
    assert window.busy, window.log.toPlainText()
    try:
        wait_until(lambda: bool(window.last_result) and not window.busy, timeout=40)
    except AssertionError:
        pytest.fail(window.log.toPlainText())
    path = Path(window.last_result)
    assert verify_dataset(path)["frames"] == 12
    manifest = json.loads((path / "manifest.json").read_text())
    presets = manifest["config"]["presets"]
    reference = manifest["auto_reference"]
    assert presets[0]["exposure_us"] == reference["exposure_us"]
    assert presets[0]["gain_db"] == reference["gain_db"]
    assert presets[0]["name"] == "custom_baseline"
    assert presets[1]["exposure_us"] == presets[0]["exposure_us"] * 4
    assert presets[2]["gain_db"] == presets[0]["gain_db"] + 12
    assert window.resolved_presets[0].exposure_us == window.base_exp.value()
    QTest.mouseClick(window.verify_button, Qt.MouseButton.LeftButton)
    wait_until(lambda: not window.busy)
    assert "검사 통과" in window.verify_status.text()


@pytest.mark.parametrize("mode,groups,frames", [("cross_reference", 3, 9), ("per_camera", 1, 3)])
def test_reference_modes_gui_table_capture_and_preview(window, tmp_path, mode, groups, frames):
    window._load_config(CaptureConfig(output_dir=str(tmp_path), capture_strategy=mode, settle_ms=0, warmup_frames=0, meter_seconds=.5))
    connect(window)
    assert f"RAW {frames}장" in window.summary.text()
    QTest.mouseClick(window.capture_button, Qt.MouseButton.LeftButton)
    wait_until(lambda: bool(window.last_result) and not window.busy, timeout=35)
    assert verify_dataset(window.last_result)["frames"] == frames
    assert window.plan_table.rowCount() == groups * 3
    assert window.preview_preset.count() == groups
    QTest.mouseClick(window.apply_preview, Qt.MouseButton.LeftButton)
    wait_until(lambda: not window.busy)
    assert isinstance(window.worker.preview_preset, dict)
    if mode == "per_camera":
        assert {p.exposure_us for p in window.worker.preview_preset.values()} == {12000, 24000, 36000}
    window.resize(640, 520)
    QTest.qWait(100)
    assert window.workspace.size() == window.pages.size()


def wheel(widget, delta=-120, *, pixels=False):
    point = widget.rect().center()
    event = QWheelEvent(QPointF(point), QPointF(widget.mapToGlobal(point)),
                        QPoint(0, delta // 3) if pixels else QPoint(), QPoint(0, delta),
                        Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier, Qt.ScrollPhase.ScrollUpdate, False)
    QApplication.sendEvent(widget, event)
    QTest.qWait(30)


@pytest.mark.parametrize("focused", [False, True])
@pytest.mark.parametrize("pixels", [False, True])
def test_wheel_over_settings_scrolls_without_changing_values(window, focused, pixels):
    window.resize(1540, 960)
    window.high_factor.setValue(6.001)
    window.repeats.setValue(2)
    QTest.qWait(100)
    config = window._config().to_dict()
    positions = [card.mapTo(window.workspace, QPoint()) for card in window.cards]
    cases = [(0, window.high_factor), (0, window.gain_increase), (0, window.strategy),
             (0, window.sync), (0, window.preview_preset), (1, window.repeats),
             (1, window.config_widgets["preview_red"]), (1, window.config_widgets["range_policy"]),
             (1, window.profile_rows["DEMO_1"][1])]
    for page, field in cases:
        window._select_page(page)
        area = window.inspector_scroll if page == 0 else window.pages.widget(1)
        bar = area.verticalScrollBar()
        changes = QSignalSpy(field.currentIndexChanged if isinstance(field, QComboBox) else field.valueChanged)
        targets = [field] + ([field.lineEdit()] if field.lineEdit() is not None else [])
        for target in targets:
            area.ensureWidgetVisible(field, 20, 80)
            field.setFocus() if focused else window.capture_button.setFocus()
            QTest.qWait(20)
            assert field.hasFocus() == focused
            before = bar.value()
            assert before < bar.maximum()
            wheel(target, -120, pixels=pixels)
            down = bar.value()
            assert down > before, (field, target, focused, pixels)
            wheel(target, 120, pixels=pixels)
            assert bar.value() < down
            assert changes.count() == 0
            assert window._config().to_dict() == config
    window._select_page(0)
    assert [card.mapTo(window.workspace, QPoint()) for card in window.cards] == positions


def test_all_numeric_and_selection_controls_use_scroll_safe_input(window):
    # Includes dynamically created camera-profile controls, not just the quick settings.
    for base, safe in ((QDoubleSpinBox, ScrollSafeDoubleSpinBox), (QSpinBox, ScrollSafeSpinBox),
                       (QComboBox, ScrollSafeComboBox)):
        fields = window.findChildren(base)
        assert fields
        assert all(isinstance(field, safe) for field in fields)


def test_scroll_safe_controls_still_accept_explicit_edits(window):
    field = window.high_factor
    field.setFocus()
    field.lineEdit().selectAll()
    QTest.keyClicks(field.lineEdit(), "6.5")
    QTest.keyClick(field, Qt.Key.Key_Return)
    assert field.value() == 6.5
    QTest.keyClick(field, Qt.Key.Key_Up)
    assert field.value() > 6.5
    QTest.mouseClick(field, Qt.MouseButton.LeftButton, pos=QPoint(field.width() - 8, field.height() // 4))
    assert field.value() > 7.5
    window._select_page(1)
    window.pages.widget(1).ensureWidgetVisible(window.repeats)
    window.repeats.setFocus()
    QTest.keyClick(window.repeats, Qt.Key.Key_Down)
    assert window.repeats.value() == 1
    QTest.keyClick(window.repeats, Qt.Key.Key_Up)
    assert window.repeats.value() == 2
    window._select_page(0)
    window.inspector_scroll.ensureWidgetVisible(window.strategy)
    window.strategy.setFocus()
    QTest.keyClick(window.strategy, Qt.Key.Key_Down)
    assert window.strategy.currentData() == "cross_reference"
    window.strategy.showPopup()
    QTest.qWait(30)
    view = window.strategy.view()
    QTest.mouseClick(view.viewport(), Qt.MouseButton.LeftButton,
                    pos=view.visualRect(view.model().index(2, 0)).center())
    assert window.strategy.currentData() == "per_camera"


@pytest.mark.parametrize("size", [(2560, 1440), (1920, 995), (1540, 960), (1280, 720), (900, 700), (640, 520), (640, 480)])
def test_main_wheel_never_moves_and_inspector_scroll_is_independent(window, size):
    connect(window)
    window.resize(*size)
    QTest.qWait(150)
    assert window.workspace.size() == window.pages.size()
    assert window.overview.columns == 2
    assert [window.overview.grid.getItemPosition(i)[:2] for i in range(len(window.cards))] == [(0, 0), (0, 1), (1, 0)]
    positions = [card.mapTo(window.workspace, QPoint()) for card in window.cards]
    for card in window.cards:
        assert window.overview.rect().contains(card.geometry())
        assert card.image.height() >= 48
        for _ in range(3):
            wheel(card.image)
    assert [card.mapTo(window.workspace, QPoint()) for card in window.cards] == positions
    assert window.workspace.rect().contains(window.stage.geometry())
    assert window.capture_button.isVisible()
    assert window.rect().contains(window.capture_button.mapTo(window, QPoint()))
    if window.inspector_toggle.isVisible():
        QTest.mouseClick(window.inspector_toggle, Qt.MouseButton.LeftButton)
        QTest.qWait(80)
        assert window.inspector_scroll.isVisible()
        assert [card.mapTo(window.workspace, QPoint()) for card in window.cards] == positions
    bar = window.inspector_scroll.verticalScrollBar()
    assert bar.maximum() > 0
    bar.setValue(0)
    wheel(window.inspector_scroll.viewport(), -600)
    assert bar.value() > 0
    assert [card.mapTo(window.workspace, QPoint()) for card in window.cards] == positions
    for _ in range(3):
        wheel(window.cards[0].image)
    assert [card.mapTo(window.workspace, QPoint()) for card in window.cards] == positions
    if window.inspector_close.isVisible():
        bar.setValue(0)
        QTest.mouseClick(window.inspector_close, Qt.MouseButton.LeftButton)
        assert not window.inspector_scroll.isVisible()
        assert [card.mapTo(window.workspace, QPoint()) for card in window.cards] == positions
        QTest.mouseClick(window.inspector_toggle, Qt.MouseButton.LeftButton)
        QTest.keyClick(window, Qt.Key.Key_Escape)
        assert not window.inspector_scroll.isVisible()
        assert [card.mapTo(window.workspace, QPoint()) for card in window.cards] == positions


def test_new_settings_and_sparse_profiles_roundtrip(window, tmp_path):
    config = CaptureConfig(capture_strategy="cross_reference", cross_bracket=True, range_policy="clip",
        auto_retry_count=3, retry_delay_ms=80,
        meter_target_percent=12.5, meter_max_exposure_us=250000, meter_max_gain_db=18,
        camera_profiles={"DEMO_1": {"sensor_hint": "IMX490", "raw_mode": "hdr", "notes": "HDR test"}},
        dslr_enabled=True, dslr_port="usb:001,003", dslr_follow=True, dslr_reference_serial="DEMO_2",
        dslr_exposure_us=25000, dslr_iso=200, dslr_aperture="5.6", dslr_white_balance="Daylight",
        dslr_max_quantization_ev=.25, dslr_dispatch_offset_ms=-100, dslr_timeout_seconds=75)
    window._load_config(config)
    assert window._config().to_dict() == config.to_dict()
    window.save_settings(tmp_path / "new_settings.json")
    window._load_config(CaptureConfig())
    window.load_settings(tmp_path / "new_settings.json")
    assert window._config().to_dict() == config.to_dict()
    assert "NEF 12장" in window.summary.text()


@pytest.mark.parametrize("size,orientation,columns,preset_columns", [
    ((1920, 1000), Qt.Orientation.Horizontal, 3, 4),
    ((2560, 1200), Qt.Orientation.Horizontal, 4, 4),
    ((1540, 960), Qt.Orientation.Horizontal, 2, 4),
    ((1280, 720), Qt.Orientation.Vertical, 2, 2),
    ((900, 700), Qt.Orientation.Vertical, 1, 2),
    ((640, 480), Qt.Orientation.Vertical, 1, 1),
])
def test_history_and_settings_use_available_width_without_moving_main(window, size, orientation, columns, preset_columns):
    window._network_setup()
    wait_until(lambda: window.last_network_report is not None and not window.busy)
    config = window._config().to_dict()
    window.resize(*size)
    window._select_page(2)
    QTest.qWait(150)
    assert window.size().toTuple() == size
    assert window.history_page.size() == window.pages.size()
    assert not isinstance(window.history_page, QScrollArea)
    splitter = window.history_splitter
    assert splitter.orientation() == orientation
    assert all(value > 0 for value in splitter.sizes())
    for child in (window.history_results, window.history_log_panel):
        assert splitter.rect().contains(child.geometry())
    if orientation == Qt.Orientation.Horizontal:
        assert window.history_results.geometry().right() < window.history_log_panel.geometry().left()
    else:
        assert window.history_results.geometry().bottom() < window.history_log_panel.geometry().top()
    # A small result table has no unused rows, and top cards don't grow into empty blocks.
    table = window.network_table
    content_height = table.horizontalHeader().height() + sum(table.rowHeight(i) for i in range(table.rowCount()))
    assert table.height() <= content_height + table.frameWidth() * 2 + table.horizontalScrollBar().sizeHint().height()
    assert window.history_network.y() - window.history_top.geometry().bottom() <= 20
    positions = [child.geometry() for child in (window.history_results, window.history_log_panel)]
    for i in range(100):
        window._log(f"검증 진행 {i}")
    bar = window.log.verticalScrollBar()
    bar.setValue(0)
    wheel(window.log.viewport(), -600)
    assert bar.value() > 0
    wheel(window.history_page, -600)
    assert [child.geometry() for child in (window.history_results, window.history_log_panel)] == positions
    window.history_results.ensureWidgetVisible(table)
    QTest.mouseClick(table.viewport(), Qt.MouseButton.LeftButton,
                    pos=table.visualItemRect(table.item(1, 0)).center())
    assert "DEMO_2" in window.network_camera_details.text()
    window._select_page(1)
    QTest.qWait(150)
    assert window.settings_columns.columns == columns
    assert max(window.preset_grid.getItemPosition(i)[1] for i in range(4)) + 1 == preset_columns
    groups_by_column = {}
    for group in window.setting_groups:
        assert window.settings_columns.rect().contains(group.geometry())
        groups_by_column.setdefault(group.x(), []).append(group)
    for groups in groups_by_column.values():
        groups.sort(key=lambda group: group.y())
        for before, after in zip(groups, groups[1:]):
            assert 0 < after.y() - before.geometry().bottom() <= 20
    assert window._config().to_dict() == config


def test_history_splitter_drag_and_orientation_restore_keep_settings(window):
    window.resize(1920, 1000)
    window._select_page(2)
    QTest.qWait(100)
    splitter = window.history_splitter
    handle = splitter.handle(1)
    original = splitter.sizes()
    point = handle.rect().center()
    QTest.mousePress(handle, Qt.MouseButton.LeftButton, pos=point)
    QTest.mouseMove(handle, point + QPoint(-160, 0), delay=30)
    QTest.mouseRelease(handle, Qt.MouseButton.LeftButton, pos=point + QPoint(-160, 0))
    QTest.qWait(60)
    dragged = splitter.sizes()
    assert dragged[0] < original[0]
    window.resize(900, 700)
    QTest.qWait(100)
    assert splitter.orientation() == Qt.Orientation.Vertical
    assert all(size > 0 for size in splitter.sizes())
    window.resize(1920, 1000)
    QTest.qWait(100)
    assert splitter.orientation() == Qt.Orientation.Horizontal
    assert abs(splitter.sizes()[0] / sum(splitter.sizes()) - dragged[0] / sum(dragged)) < .01


def test_white_balance_button_once_shares_reference_gains_and_preserves_paused_raw(window, tmp_path, monkeypatch):
    import numpy as np
    from rawrig.demo import DemoBackend
    calls = []
    original = DemoBackend.white_balance_once
    def track(backend, config, event):
        calls.append(config.white_balance_reference_serial)
        return original(backend, config, event)
    monkeypatch.setattr(DemoBackend, "white_balance_once", track)
    window._load_config(CaptureConfig(capture_strategy="cross_reference", white_balance_reference_serial="DEMO_2"))
    assert not window.wb_button.isEnabled()
    connect(window)
    assert window.wb_button.isEnabled() and not window.reference.isEnabled()
    QTest.mouseClick(window.preview_check, Qt.MouseButton.LeftButton)
    wait_until(lambda: not window.busy and not window.worker.preview_active)
    originals = [frame.pixels.copy() for frame in window.worker.last_frames]
    before = window._config().to_dict()
    QTest.mouseClick(window.wb_button, Qt.MouseButton.LeftButton)
    assert window.busy and not window.wb_button.isEnabled()
    wait_until(lambda: bool(window.last_white_balance) and not window.busy)
    assert calls == ["DEMO_2"]
    assert window.worker.display_options[:2] == (1.1765, 1.5385)
    assert window.last_white_balance["applied_serials"] == ["DEMO_1", "DEMO_2", "DEMO_3"]
    assert window.last_white_balance["camera_auto"] == "Off"
    assert "DEMO_2" in window.wb_status.text() and "고정" in window.wb_status.text()
    for frame, original_raw in zip(window.worker.last_frames, originals):
        np.testing.assert_array_equal(frame.pixels, original_raw)
    after = window._config().to_dict()
    for key in ("preview_red", "preview_blue", "white_balance_calibration"):
        before.pop(key)
        after.pop(key)
    assert before == after
    saved = window.save_settings(tmp_path / "wb.json").to_dict()
    window._load_config(CaptureConfig())
    window.load_settings(tmp_path / "wb.json")
    assert window._config().to_dict() == saved
    QTest.mouseClick(window.preview_check, Qt.MouseButton.LeftButton)
    wait_until(lambda: not window.busy and window.worker.preview_active)
    QTest.qWait(200)
    assert calls == ["DEMO_2"]  # Live preview resumes with fixed gains, never another AWB call.
    window.config_widgets["preview_red"].setValue(1.3)
    assert not window.last_white_balance
    assert "수동 WB" in window.wb_status.text()


@pytest.mark.parametrize("cancel", [False, True])
def test_white_balance_failure_or_stop_keeps_previous_values_and_measurement(window, monkeypatch, cancel):
    from rawrig.demo import DemoBackend
    from rawrig.models import CaptureError, check_cancel
    connect(window)
    QTest.mouseClick(window.wb_button, Qt.MouseButton.LeftButton)
    wait_until(lambda: bool(window.last_white_balance) and not window.busy)
    previous = window._config().to_dict()
    def failed(backend, config, event):
        event.wait(.15)
        check_cancel(event)
        raise CaptureError("WB 1회 측정 시간 초과")
    monkeypatch.setattr(DemoBackend, "white_balance_once", failed)
    QTest.mouseClick(window.wb_button, Qt.MouseButton.LeftButton)
    if cancel:
        QTest.mouseClick(window.stop_button, Qt.MouseButton.LeftButton)
    wait_until(lambda: not window.busy)
    assert window._config().to_dict() == previous
    assert "기존 값 유지" in window.wb_status.text()
    assert window.wb_button.isEnabled()


def test_search_keeps_focus_scroll_anchor_and_unrelated_idle_cannot_unlock(window, monkeypatch):
    window.resize(1540, 720)
    QTest.qWait(100)
    window.inspector_scroll.ensureWidgetVisible(window.network_button)
    window.network_button.setFocus()
    QTest.qWait(30)
    bar = window.inspector_scroll.verticalScrollBar()
    anchor = bar.value()
    positions = [c.mapTo(window.workspace, QPoint()) for c in window.cards]
    monkeypatch.setattr(window.worker, 'submit', lambda *args:None)
    QTest.mouseClick(window.network_button, Qt.MouseButton.LeftButton)
    assert window.busy and window.inspector_scroll.scroll_locked
    window._idle('display')
    assert window.busy and window.active_command == 'network'
    assert window.inspector_scroll.scroll_locked
    window.inspector_scroll.ensureWidgetVisible(window.wb_reference)
    for i in range(10):
        window._progress('인터페이스 재검색 ' + '상세 상태 ' * 50, i, 10)
        QApplication.processEvents()
        assert bar.value() == anchor
        assert [c.mapTo(window.workspace, QPoint()) for c in window.cards] == positions
    # The user can still scroll intentionally during the operation.
    bar.setValue(anchor)
    wheel(window.inspector_scroll.viewport(), -600)
    assert bar.value() > anchor
    new_anchor = bar.value()
    window._idle('network')
    QTest.qWait(20)
    assert not window.busy and not window.inspector_scroll.scroll_locked
    assert bar.value() == new_anchor


@pytest.mark.parametrize('count', [1, 2, 3])
def test_only_connected_camera_cards_are_laid_out_in_capture_and_noise_pages(window, count):
    for i in range(window.camera_list.count()):
        window.camera_list.item(i).setCheckState(Qt.CheckState.Checked if i < count else Qt.CheckState.Unchecked)
    QTest.mouseClick(window.connect_button, Qt.MouseButton.LeftButton)
    wait_until(lambda: not window.busy and len(window.cameras) == count)
    wait_until(lambda: window.noise_panel.camera_cards[0].saved_image is not None)
    for size in ((1920, 995), (640, 480)):
        window.resize(*size)
        QTest.qWait(100)
        for page, grid, cards in ((0, window.overview, window.cards), (3, window.noise_panel.live_grid, window.noise_panel.camera_cards)):
            window._select_page(page)
            QTest.qWait(60)
            assert grid.grid.count() == count
            assert sum(not c.isHidden() for c in cards) == count
            assert grid.columns == (1 if count == 1 else 2)
            assert [grid.grid.getItemPosition(i)[:2] for i in range(count)] == [(i // grid.columns, i % grid.columns) for i in range(count)]
            for card in cards[:count]:
                assert grid.rect().contains(card.geometry())
            before = grid.pos()
            wheel(cards[0].image, -600)
            assert grid.pos() == before
    panel = window.noise_panel
    assert panel.report is None and panel.display_deck.currentWidget() is panel.live_grid
    QTest.mouseClick(panel.camera_cards[0].image, Qt.MouseButton.LeftButton)
    assert panel.display_deck.currentWidget() is panel.live_focus and not panel.inspector.isVisible()
    QTest.mouseClick(panel.live_focus, Qt.MouseButton.LeftButton)
    assert panel.display_deck.currentWidget() is panel.live_grid


def test_second_gui_instance_activates_existing_window_and_releases_camera_lock(tmp_path, monkeypatch):
    from rawrig.instance import GuiInstance
    from PySide6.QtCore import QStandardPaths
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(QStandardPaths, 'writableLocation', lambda _:str(tmp_path))
    activated = []
    first, second = GuiInstance(), GuiInstance()
    first.name = second.name = 'raw-rig-test-' + str(os.getpid())
    try:
        assert first.acquire(lambda:activated.append(True))
        assert not second.acquire(lambda:None)
        QTest.qWait(30)
        assert activated == [True]
    finally:
        first.close()
        second.close()
    third = GuiInstance()
    third.name = first.name
    try:
        assert third.acquire(lambda:None)
    finally:
        third.close()
    app.processEvents()
