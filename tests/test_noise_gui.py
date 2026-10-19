import os

os.environ["QT_QPA_PLATFORM"] = "offscreen"
from dataclasses import asdict
from pathlib import Path
import time
from unittest.mock import patch

import pytest
from PySide6.QtCore import Qt, QPoint, QPointF
from PySide6.QtGui import QWheelEvent, QImage
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from rawrig.gui import MainWindow
from rawrig.demo import DemoBackend
from rawrig.noise import NoiseConfig, verify_noise_session


class SmallDemo(DemoBackend):
    def _settings(self, index):
        return {**super()._settings(index), "width": 64, "height": 64}


def wait_until(predicate, timeout=30):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("Qt operation timed out")
        QApplication.processEvents()
        time.sleep(0.01)


@pytest.fixture
def window(monkeypatch):
    monkeypatch.setattr("rawrig.worker.DemoBackend", SmallDemo)
    app = QApplication.instance() or QApplication([])
    w = MainWindow(demo=True)
    w.show()
    wait_until(lambda: not w.busy and w.camera_list.count() == 3)
    w.camera_list.item(1).setCheckState(Qt.CheckState.Unchecked)
    w.camera_list.item(2).setCheckState(Qt.CheckState.Unchecked)
    QTest.mouseClick(w.connect_button, Qt.MouseButton.LeftButton)
    wait_until(lambda: len(w.cameras) == 1 and not w.busy)
    yield w
    w.worker.request_shutdown()
    wait_until(lambda: not w.worker.isRunning())
    w.close()
    app.processEvents()


def wheel(widget, delta=-120):
    point = widget.rect().center()
    event = QWheelEvent(
        QPointF(point),
        QPointF(widget.mapToGlobal(point)),
        QPoint(),
        QPoint(0, delta),
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.NoScrollPhase,
        False,
    )
    QApplication.sendEvent(widget, event)
    QTest.qWait(20)


@pytest.mark.parametrize("size", [(1920, 995), (1280, 720), (900, 700), (640, 480)])
def test_noise_workspace_is_fixed_with_contained_actions_and_independent_inspector(
    window, size
):
    window._select_page(3)
    window.resize(*size)
    QTest.qWait(120)
    panel = window.noise_panel
    assert panel.size() == window.pages.size()
    assert not window.capture_button.isVisible()
    for button in (panel.dark_button, panel.bright_button, panel.stop_button):
        assert panel.rect().contains(button.mapTo(panel, button.rect().topLeft()))
        assert panel.rect().contains(button.mapTo(panel, button.rect().bottomRight()))
    position = panel.display_deck.mapTo(panel, QPoint())
    wheel(panel.camera_cards[0].image, -600)
    assert panel.display_deck.mapTo(panel, QPoint()) == position
    if panel.toggle.isVisible():
        QTest.mouseClick(panel.toggle, Qt.MouseButton.LeftButton)
        QTest.qWait(60)
    assert panel.inspector.isVisible()
    assert panel.display_deck.mapTo(panel, QPoint()) == position
    panel.inspector.ensureWidgetVisible(panel.protocol_choice)
    before = panel.protocol_choice.currentIndex()
    scroll = panel.inspector.verticalScrollBar()
    old = scroll.value()
    panel.protocol_choice.setFocus()
    wheel(panel.protocol_choice, -600)
    assert panel.protocol_choice.currentIndex() == before
    assert scroll.value() > old
    assert panel.display_deck.mapTo(panel, QPoint()) == position
    if panel.toggle.isVisible():
        QTest.keyClick(window, Qt.Key.Key_Escape)
        assert not panel.inspector.isVisible()


def test_two_buttons_do_not_auto_start_bright_and_then_auto_analyze_with_config_preserved(
    window, tmp_path
):
    panel = window.noise_panel
    noise = NoiseConfig.fixed()
    panel.load_config(asdict(noise))
    window.output.setText(str(tmp_path))
    window._select_page(3)
    assert panel.dark_button.isEnabled() and not panel.bright_button.isEnabled()
    base_exp = window.base_exp.value()
    wb = window.worker.display_options

    # Numerical/full graph coverage is in test_noise; here exercise UI thread routing.
    def small_plots(folder, result, maps, plots, cancel, **kwargs):
        image = QImage(64, 64, QImage.Format.Format_RGB888)
        image.fill(Qt.GlobalColor.red)
        image.save(str(folder / "test.png"))
        plots.append(
            {
                "label": "Synthetic graph",
                "path": f"{folder.name}/test.png",
                "svg": f"{folder.name}/test.svg",
            }
        )
        (folder / "test.svg").write_text('<svg xmlns="http://www.w3.org/2000/svg"/>')

    with (
        patch("rawrig.noise_analysis.create_camera_plots", side_effect=small_plots),
        patch("rawrig.noise_analysis.create_fixed_plots", side_effect=small_plots),
    ):
        QTest.mouseClick(panel.dark_button, Qt.MouseButton.LeftButton)
        wait_until(lambda: not window.busy and panel.dark_complete)
        path = Path(panel.session_path)
        assert path.name.endswith(".partial")
        assert not (path / "BRIGHT_COMPLETE.json").exists()
        assert panel.analyze_button.isEnabled()
        assert (
            panel.report is not None
            and panel.report["cameras"][0]["channels"]["R"]["curve"] == []
        )
        assert panel.bright_button.isEnabled()
        panel.protocol_choice.setCurrentIndex(0)
        assert not panel.bright_button.isEnabled()
        panel.protocol_choice.setCurrentIndex(1)
        QTest.mouseClick(panel.bright_button, Qt.MouseButton.LeftButton)
        wait_until(lambda: not window.busy and panel.report is not None)
    assert verify_noise_session(panel.session_path)["passed"]
    assert not Path(panel.session_path).name.endswith(".partial")
    assert panel.canvas.image is not None
    assert panel.summary.rowCount() == 4
    assert panel.report["simulated"]
    assert panel.bright_button.isEnabled()
    assert "재촬영" in panel.bright_button.text()
    assert window.base_exp.value() == base_exp and window.worker.display_options == wb
    QTest.mouseClick(panel.canvas, Qt.MouseButton.LeftButton)
    assert panel.graph_zoomed and not panel.inspector.isVisible()
    QTest.keyClick(window, Qt.Key.Key_Escape)
    assert not panel.graph_zoomed
    window.save_settings(tmp_path / "config.json")
    panel.protocol_choice.setCurrentIndex(0)
    window.load_settings(tmp_path / "config.json")
    assert asdict(panel.settings()) == asdict(noise)
