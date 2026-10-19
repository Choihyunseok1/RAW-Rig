from __future__ import annotations

from dataclasses import replace
from datetime import datetime
import json
import os
from pathlib import Path
import signal
from threading import Event

from PySide6.QtCore import Qt, QTimer, QUrl, QSignalBlocker
from PySide6.QtGui import QAction, QDesktopServices, QFont, QKeySequence, QIcon
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QFrame, QVBoxLayout, QHBoxLayout,
    QBoxLayout, QGridLayout, QLabel, QPushButton, QComboBox, QCheckBox,
    QGroupBox, QFormLayout, QLineEdit, QFileDialog, QPlainTextEdit, QProgressBar,
    QListWidget, QListWidgetItem, QStackedWidget, QTableWidget,
    QTableWidgetItem, QHeaderView, QAbstractItemView, QSizePolicy, QSplitter)

from .models import CaptureConfig, CaptureError, Preset, bracket_presets
from .planning import step_presets
from .specs import comparison_text
from .widgets import (CameraCard, CameraGrid, PreviewCanvas, AdaptiveColumns, StableScrollArea, ScrollSafeComboBox,
                      ScrollSafeDoubleSpinBox, ScrollSafeSpinBox)
from .worker import CameraWorker


APP_ICON = Path(__file__).parent / "assets" / "raw-rig.svg"


def double(value, low, high, decimals=2, suffix=""):
    widget = ScrollSafeDoubleSpinBox()
    widget.setRange(low, high)
    widget.setDecimals(decimals)
    widget.setValue(value)
    widget.setSuffix(suffix)
    return widget


def integer(value, low, high, suffix=""):
    widget = ScrollSafeSpinBox()
    widget.setRange(low, high)
    widget.setValue(value)
    widget.setSuffix(suffix)
    return widget


def scroll(widget):
    area = StableScrollArea()
    area.setWidgetResizable(True)
    area.setFrameShape(QFrame.Shape.NoFrame)
    area.setWidget(widget)
    return area


def caption(text, name="mutedText"):
    label = QLabel(text)
    label.setObjectName(name)
    label.setWordWrap(True)
    return label


class MainWindow(QMainWindow):
    def __init__(self, demo=False, config=None, expected_cameras=3):
        super().__init__()
        self.demo = demo
        self.initial_config = config or CaptureConfig(expected_cameras=expected_cameras)
        self.cameras, self.busy, self.last_result = [], False, ""
        self.active_command = None
        self.zoomed_index = None
        self.inspector_overlay_open = False
        self.last_reference = None
        self.resolved_presets = None
        self.last_network_report = None
        self.last_white_balance = {}
        self.config_widgets = {}
        self.profile_rows = {}
        self.saved_profiles = {}
        self.loading = True
        self.expected_cameras = self.initial_config.expected_cameras
        self.setWindowTitle("RAW Rig · 멀티 카메라 RAW 데이터셋" + (" · DEMO" if demo else ""))
        self.setWindowIcon(QIcon(str(APP_ICON)))
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.WindowMaximizeButtonHint | Qt.WindowType.WindowMinimizeButtonHint)
        self.setMinimumSize(640, 480)
        self.resize(1540, 960)
        root = QWidget()
        self.setCentralWidget(root)
        shell = QHBoxLayout(root)
        shell.setContentsMargins(0, 0, 0, 0)
        shell.setSpacing(0)
        self._build_nav(shell)
        self.content = QWidget()
        self.content.setObjectName("pageBody")
        content_layout = QVBoxLayout(self.content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)
        shell.addWidget(self.content, 1)
        self._build_header(content_layout)
        self.pages = QStackedWidget()
        content_layout.addWidget(self.pages, 1)
        self._build_workspace()
        self._build_settings()
        self._build_history()
        from .noise_gui import NoisePanel
        self.noise_panel = NoisePanel()
        self.pages.addWidget(self.noise_panel)
        self._build_footer(content_layout)
        self.setStyleSheet(STYLE)
        self.worker = CameraWorker(demo)
        self.worker.discovered.connect(self._discovered)
        self.worker.connected.connect(self._connected)
        self.worker.images.connect(self._images)
        self.worker.progress.connect(self._progress)
        self.worker.resolved.connect(self._resolved)
        self.worker.message.connect(self._log)
        self.worker.error.connect(self._error)
        self.worker.result.connect(self._result)
        self.worker.verified.connect(self._verified)
        self.worker.network_result.connect(self._network_result)
        self.worker.nikon_result.connect(self._nikon_result)
        self.worker.nikon_image.connect(self._nikon_image)
        self.worker.white_balance_result.connect(self._white_balance_result)
        self.worker.noise_result.connect(self.noise_panel.handle_result)
        self.worker.aborted.connect(self._aborted)
        self.worker.idle.connect(self._idle)
        self.worker.start()
        self._bind()
        self._load_config(self.initial_config)
        self._controls()
        self._select_page(0)
        QTimer.singleShot(100, lambda: self._command("scan"))

    def _build_nav(self, shell):
        self.sidebar = QFrame()
        self.sidebar.setObjectName("sidebar")
        self.sidebar.setFixedWidth(172)
        nav = QVBoxLayout(self.sidebar)
        nav.setContentsMargins(14, 28, 14, 20)
        nav.setSpacing(10)
        self.logo = QLabel("RAW\nRIG")
        self.logo.setObjectName("logo")
        nav.addWidget(self.logo)
        self.sidebar_subtitle = caption("CAPTURE STUDIO", "navMuted")
        nav.addWidget(self.sidebar_subtitle)
        nav.addSpacing(32)
        self.nav_buttons = []
        self.nav_labels = [("◉", "촬영"), ("⚙", "촬영 설정"), ("≡", "검사·기록"), ("∿", "노이즈 분석")]
        for index, (icon, text) in enumerate(self.nav_labels):
            button = QPushButton(f"{icon}   {text}")
            button.setObjectName("navButton")
            button.setCheckable(True)
            button.clicked.connect(lambda checked=False, page=index: self._select_page(page))
            nav.addWidget(button)
            self.nav_buttons.append(button)
        nav.addStretch()
        self.sidebar_note = caption("BAYER RAW\n원본 픽셀 저장", "navMuted")
        nav.addWidget(self.sidebar_note)
        shell.addWidget(self.sidebar)

    def _build_header(self, layout):
        header = QFrame()
        header.setObjectName("header")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(28, 17, 24, 17)
        titles = QVBoxLayout()
        titles.setSpacing(4)
        self.page_title = caption("동기 촬영", "pageTitle")
        self.subtitle = caption("실시간 확인부터 RAW 저장까지", "mutedText")
        titles.addWidget(self.page_title)
        titles.addWidget(self.subtitle)
        header_layout.addLayout(titles, 1)
        self.badge = caption("DEMO · 가상 데이터" if self.demo else "Spinnaker · GigE", "modeBadge")
        header_layout.addWidget(self.badge)
        self.load_button = QPushButton("설정 불러오기")
        self.save_button = QPushButton("설정 저장")
        self.fullscreen_button = QPushButton("□  전체화면")
        self.fullscreen_button.setToolTip("전체화면 / 복귀 · F11")
        for button in (self.load_button, self.save_button, self.fullscreen_button):
            header_layout.addWidget(button)
        layout.addWidget(header)

    def _build_workspace(self):
        workspace = QWidget()
        workspace.setObjectName("pageBody")
        self.workspace_layout = QBoxLayout(QBoxLayout.Direction.LeftToRight, workspace)
        self.workspace_layout.setContentsMargins(24, 20, 24, 20)
        self.workspace_layout.setSpacing(20)
        self.stage = QWidget()
        stage_layout = QVBoxLayout(self.stage)
        self.stage_layout = stage_layout
        stage_layout.setContentsMargins(0, 0, 0, 0)
        stage_layout.setSpacing(16)
        stage_header = QHBoxLayout()
        stage_header.addWidget(caption("실시간 미리보기", "sectionTitle"))
        stage_header.addStretch()
        self.preview_check = QCheckBox("LIVE")
        self.preview_check.setChecked(True)
        stage_header.addWidget(self.preview_check)
        self.wb_button = QPushButton("자동 WB · 1회")
        stage_header.addWidget(self.wb_button)
        self.inspector_toggle = QPushButton("설정 패널  ☰")
        self.inspector_toggle.setToolTip("중앙 카메라 화면을 움직이지 않고 옆 설정 패널을 열거나 닫습니다.")
        stage_header.addWidget(self.inspector_toggle)
        stage_layout.addLayout(stage_header)
        self.zoom_hint = caption("이미지를 클릭하면 크게 볼 수 있습니다. 다시 클릭하면 전체 보기로 돌아옵니다.")
        stage_layout.addWidget(self.zoom_hint)
        self.camera_deck = QStackedWidget()
        self.camera_deck.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.cards = [CameraCard(index) for index in range(3)]
        self.overview = CameraGrid(self.cards)
        self.overview.set_count(0)
        self.camera_deck.addWidget(self.overview)
        self.focus_view = QFrame()
        self.focus_view.setObjectName("focusCard")
        focus_layout = QVBoxLayout(self.focus_view)
        focus_layout.setContentsMargins(18, 16, 18, 16)
        focus_top = QHBoxLayout()
        self.focus_title = caption("CAMERA", "sectionTitle")
        focus_top.addWidget(self.focus_title, 1)
        self.back_button = QPushButton("전체 카메라로 돌아가기  ↙")
        focus_top.addWidget(self.back_button)
        focus_layout.addLayout(focus_top)
        self.focus_image = PreviewCanvas()
        self.focus_image.setMinimumSize(40, 48)
        focus_layout.addWidget(self.focus_image, 1)
        self.focus_detail = caption("다시 클릭하여 전체 보기로 돌아가기")
        focus_layout.addWidget(self.focus_detail)
        self.camera_deck.addWidget(self.focus_view)
        stage_layout.addWidget(self.camera_deck, 1)
        self.sequence_panel = QFrame()
        self.sequence_panel.setObjectName("whitePanel")
        sequence_layout = QVBoxLayout(self.sequence_panel)
        sequence_layout.setContentsMargins(18, 17, 18, 17)
        sequence_top = QHBoxLayout()
        sequence_top.addWidget(caption("노출 컨트롤 · 촬영 순서", "sectionTitle"))
        sequence_top.addStretch()
        self.baseline_status = caption("설정 1 = 기준값", "accentText")
        sequence_top.addWidget(self.baseline_status)
        sequence_layout.addLayout(sequence_top)
        self.sequence_grid = QGridLayout()
        self.sequence_grid.setHorizontalSpacing(12)
        self.sequence_tiles = []
        for index, name in enumerate(("기준값", "노출시간 증가", "게인 증가", "노출시간·게인 증가")):
            tile = QFrame()
            tile.setObjectName("baselineTile" if index == 0 else "sequenceTile")
            tile_layout = QVBoxLayout(tile)
            tile_layout.setContentsMargins(12, 12, 12, 12)
            tile_layout.addWidget(caption(f"0{index+1}  {name}", "tileTitle"))
            values = caption("T₀ µs\nG₀ dB", "tileValues")
            tile_layout.addWidget(values)
            self.sequence_tiles.append((tile, values))
            self.sequence_grid.addWidget(tile, 0, index)
        sequence_layout.addLayout(self.sequence_grid)
        self.plan_table = QTableWidget(0, 4)
        self.plan_table.setHorizontalHeaderLabels(["촬영 기준", "대상 카메라", "노출시간 µs", "게인 dB"])
        self.plan_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.plan_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.plan_table.setMinimumHeight(170)
        self.plan_table.setMaximumHeight(290)
        self.plan_table.hide()
        sequence_layout.addWidget(self.plan_table)
        self.sequence_note = caption("기준값은 낮은 노출시간·낮은 게인입니다. 나머지 설정에서 값을 높입니다.")
        sequence_layout.addWidget(self.sequence_note)
        self.summary = caption("4개 설정 × 3대 × 1회 = RAW 12장", "summaryText")
        stage_layout.addWidget(self.summary)
        self.inspector = self._build_inspector()
        self.inspector_scroll = scroll(self.inspector)
        self.inspector_scroll.setObjectName("inspectorPanel")
        self.inspector_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.inspector_scroll.setMinimumWidth(300)
        self.workspace_layout.addWidget(self.stage, 1)
        self.workspace_layout.addWidget(self.inspector_scroll)
        self.workspace = workspace
        # The camera stage has no scroll viewport. Wheel events over it cannot
        # move the page; only the separate inspector owns a scroll position.
        self.pages.addWidget(workspace)

    def _build_inspector(self):
        inspector = QWidget()
        inspector.setObjectName("cameraInspector")
        inspector.setMinimumWidth(300)
        inspector.setMaximumWidth(350)
        layout = QVBoxLayout(inspector)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(16)
        inspector_header = QHBoxLayout()
        inspector_header.addWidget(caption("카메라 · 촬영 설정", "sectionTitle"), 1)
        self.inspector_close = QPushButton("닫기  ×")
        self.inspector_close.setToolTip("옆 설정 패널을 닫고 고정된 카메라 화면으로 돌아갑니다.")
        inspector_header.addWidget(self.inspector_close)
        layout.addLayout(inspector_header)
        self.devices_group = QGroupBox("카메라 연결")
        dev = QVBoxLayout(self.devices_group)
        self.camera_list = QListWidget()
        self.camera_list.setFixedHeight(108)
        dev.addWidget(self.camera_list)
        row = QHBoxLayout()
        self.scan_button = QPushButton("검색")
        self.connect_button = QPushButton("선택 연결")
        self.disconnect_button = QPushButton("해제")
        for button in (self.scan_button, self.connect_button, self.disconnect_button):
            row.addWidget(button)
        dev.addLayout(row)
        self.camera_status = caption("0 / 3대 연결", "accentText")
        dev.addWidget(self.camera_status)
        self.network_button = QPushButton("새 카메라 찾기 · 주소 설정")
        self.network_button.setObjectName("networkButton")
        self.network_button.setMinimumHeight(38)
        dev.addWidget(self.network_button)
        self.network_status = caption("IP·서브넷·주소 충돌을 검사하고 RAW 수신까지 검증합니다.")
        self.network_status.setFixedHeight(54)
        dev.addWidget(self.network_status)
        form = QFormLayout()
        self.reference = ScrollSafeComboBox()
        form.addRow("기준 카메라", self.reference)
        self.sync = ScrollSafeComboBox()
        for text, value in (("PTP · 예약 촬영", "ptp"), ("소프트웨어 트리거", "software"), ("GPIO 동기 촬영", "hardware")):
            self.sync.addItem(text, value)
        form.addRow("동기 촬영", self.sync)
        dev.addLayout(form)
        self.sync_hint = caption("PTP 시계와 프레임 시차를 검사합니다.")
        dev.addWidget(self.sync_hint)
        layout.addWidget(self.devices_group)
        wb_group = QGroupBox("화이트밸런스 · 전체 카메라 공통")
        wb_layout = QFormLayout(wb_group)
        self.wb_reference = ScrollSafeComboBox()
        self.config_widgets["white_balance_reference_serial"] = self.wb_reference
        wb_layout.addRow("WB 기준 카메라", self.wb_reference)
        self.wb_status = caption("미측정 · R 1.0000 / B 1.0000", "accentText")
        wb_layout.addRow(self.wb_status)
        wb_layout.addRow(caption("상단 ‘자동 WB · 1회’로 기준을 측정합니다. 전체 미리보기에 같은 값을 고정하며 RAW 원본은 보존합니다."))
        layout.addWidget(wb_group)
        self.exposure_group = QGroupBox("노출 컨트롤")
        exposure = QVBoxLayout(self.exposure_group)
        self.auto = QCheckBox("촬영할 때 자동 기준 측광")
        self.auto.setChecked(True)
        exposure.addWidget(self.auto)
        self.strategy = ScrollSafeComboBox()
        for label, value in (("기준 1대 · 네 설정", "bracket"), ("카메라별 기준 · 교차 촬영", "cross_reference"), ("각자 자동 측광값 · 동시 촬영", "per_camera")):
            self.strategy.addItem(label, value)
        exposure.addWidget(self.strategy)
        self.cross_bracket = QCheckBox("각 기준에서 네 설정으로 확장")
        exposure.addWidget(self.cross_bracket)
        self.strategy_hint = caption("기준 1대의 자동 측정값을 전체 카메라에 적용합니다.")
        exposure.addWidget(self.strategy_hint)
        self.base_caption = caption("설정 1 · 낮은 노출시간 / 낮은 게인", "accentText")
        exposure.addWidget(self.base_caption)
        form = QFormLayout()
        self.base_form = form
        self.base_exp = double(1000., 1, 30000000, 1, " µs")
        self.base_gain = double(0., 0, 48, 2, " dB")
        form.addRow("기준 노출시간 T₀", self.base_exp)
        form.addRow("기준 게인 G₀", self.base_gain)
        self.high_factor = double(4., 1.001, 100, 3, " ×")
        self.gain_increase = double(12., .001, 48, 3, " dB")
        form.addRow("노출시간 증가 배수", self.high_factor)
        form.addRow("게인 증가량", self.gain_increase)
        exposure.addLayout(form)
        self.meter_button = QPushButton("현재 장면 기준 측광")
        exposure.addWidget(self.meter_button)
        self.baseline_hint = caption("자동 측광값이 설정 1의 기준값이 됩니다.")
        exposure.addWidget(self.baseline_hint)
        edit_button = QPushButton("모든 촬영 설정 편집  →")
        edit_button.clicked.connect(lambda: self._select_page(1))
        exposure.addWidget(edit_button)
        layout.addWidget(self.exposure_group)
        layout.addWidget(self.sequence_panel)
        self.quick_preview_group = QGroupBox("미리보기 설정")
        preview_layout = QFormLayout(self.quick_preview_group)
        self.preview_preset = ScrollSafeComboBox()
        self.preview_preset.addItems(["설정 1 · 기준값", "설정 2", "설정 3", "설정 4"])
        preview_layout.addRow("표시할 설정", self.preview_preset)
        self.apply_preview = QPushButton("미리보기에 적용")
        preview_layout.addRow(self.apply_preview)
        preview_layout.addRow(caption("화이트밸런스·표시 밝기는 RAW 원본에 적용하지 않습니다."))
        layout.addWidget(self.quick_preview_group)
        layout.addStretch()
        return inspector

    def _build_footer(self, layout):
        footer = QFrame()
        footer.setObjectName("footer")
        outer = QVBoxLayout(footer)
        outer.setContentsMargins(24, 12, 24, 16)
        outer.setSpacing(9)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 4)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(5)
        outer.addWidget(self.progress_bar)
        self.footer_row = QHBoxLayout()
        self.state = caption("카메라를 검색하고 연결하세요.")
        self.state.setFixedHeight(36)
        self.state.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.footer_row.addWidget(self.state, 1)
        self.stop_button = QPushButton("촬영 중지")
        self.stop_button.setEnabled(False)
        self.capture_button = QPushButton("측광 후 4개 설정 촬영  →")
        self.capture_button.setObjectName("captureButton")
        self.capture_button.setMinimumHeight(46)
        self.capture_button.setMinimumWidth(220)
        self.footer_row.addWidget(self.stop_button)
        self.footer_row.addWidget(self.capture_button)
        outer.addLayout(self.footer_row)
        layout.addWidget(footer)

    def _build_settings(self):
        page = QWidget()
        page.setObjectName("pageBody")
        layout = QVBoxLayout(page)
        layout.setContentsMargins(24, 20, 24, 24)
        layout.setSpacing(18)
        layout.addWidget(caption("설정 1의 기존값을 기준으로 노출시간과 게인을 높입니다. 자동 기준 측광을 끄면 네 설정을 각각 직접 입력할 수 있습니다."))
        self.preset_group = QGroupBox("네 가지 노출 컨트롤 설정")
        preset_layout = QVBoxLayout(self.preset_group)
        self.manual_hint = caption("자동 모드에서는 촬영 직전 측광값으로 아래 값이 결정됩니다.")
        preset_layout.addWidget(self.manual_hint)
        self.preset_grid = QGridLayout()
        self.preset_tiles, self.preset_editors = [], []
        for index in range(4):
            tile = QFrame()
            tile.setObjectName("sequenceTile")
            form = QFormLayout(tile)
            form.addRow(caption(f"설정 {index + 1}" + (" · 기준값" if index == 0 else ""), "tileTitle"))
            name = QLineEdit()
            exposure = double(1000., 1, 30000000, 1, " µs")
            gain = double(0., 0, 48, 3, " dB")
            form.addRow("저장 이름", name)
            form.addRow("노출시간", exposure)
            form.addRow("게인", gain)
            self.preset_tiles.append(tile)
            self.preset_editors.append((name, exposure, gain))
            self.preset_grid.addWidget(tile, index // 2, index % 2)
        preset_layout.addLayout(self.preset_grid)
        layout.addWidget(self.preset_group)
        self.setting_groups = []

        def group(title):
            box = QGroupBox(title)
            form = QFormLayout(box)
            form.setSpacing(12)
            self.setting_groups.append(box)
            return form

        def field(form, key, label, widget):
            self.config_widgets[key] = widget
            form.addRow(label, widget)
            return widget

        capture = group("촬영과 저장")
        self.camera_count = field(capture, "expected_cameras", "필요한 카메라 수", integer(3, 1, 3, " 대"))
        self.repeats = field(capture, "repeats", "설정별 촬영 수", integer(1, 1, 100, " 회"))
        self.output = QLineEdit("demo_output" if self.demo else "datasets")
        self.config_widgets["output_dir"] = self.output
        output_row = QHBoxLayout()
        output_row.addWidget(self.output, 1)
        self.output_browse = QPushButton("선택")
        output_row.addWidget(self.output_browse)
        capture.addRow("저장 폴더", output_row)
        field(capture, "settle_ms", "설정 안정화 시간", integer(100, 0, 10000, " ms"))
        field(capture, "warmup_frames", "워밍업 촬영 수", integer(1, 0, 10, " 회"))
        field(capture, "timeout_ms", "프레임 수신 제한시간", integer(3000, 100, 120000, " ms"))
        field(capture, "auto_retry_count", "일시 오류 자동 복구", integer(2, 0, 5, " 회"))
        field(capture, "retry_delay_ms", "재시도 간격", integer(400, 0, 5000, " ms"))
        capture.addRow(caption("전송 손실·CRC·오래된 프레임·PTP 일시 오류는 동일 조건으로 묶음 전체를 재촬영합니다. 통과한 묶음은 다시 촬영하지 않습니다. 복구 과정도 기록합니다."))
        capture.addRow(caption("워밍업 촬영은 검사 후 폐기합니다. 완성 데이터는 무결성 검사를 통과한 뒤 저장됩니다."))

        auto = group("자동 기준 측광")
        field(auto, "meter_seconds", "자동 컨트롤 수렴 시간", double(4., .5, 30, 1, " 초"))
        self.low_factor = field(auto, "low_exposure_factor", "기준 노출시간 보정 배수", double(1., .001, 99., 3, " ×"))
        self.low_gain_offset = field(auto, "low_gain_offset_db", "기준 게인 보정량", double(0., -48, 48, 3, " dB"))
        auto.addRow(caption("기본 보정은 1× / 0 dB입니다. 자동 컨트롤의 현재 노출시간·게인을 그대로 설정 1로 사용합니다. 보정하면 보정된 값이 새 기준값이 됩니다."))
        auto.addRow(caption("기준 카메라와 노출시간 증가 배수·게인 증가량은 촬영 화면의 노출 컨트롤에서 변경합니다."))
        field(auto, "meter_target_percent", "목표 밝기", double(20., 1, 90, 1, " %"))
        field(auto, "meter_max_exposure_us", "자동 최대 노출시간", double(100000., 15, 30000000, 1, " µs"))
        field(auto, "meter_max_gain_db", "자동 최대 게인", double(24., 0, 48, 2, " dB"))
        policy = ScrollSafeComboBox()
        policy.addItem("지원 범위 밖이면 중단", "strict")
        policy.addItem("공통 범위로 제한하고 기록", "clip")
        field(auto, "range_policy", "교차 촬영 범위 처리", policy)
        auto.addRow(caption("각 카메라에 같은 목표 밝기·상한을 요청합니다. 지원 여부와 실제 적용값을 기록합니다. 교차 촬영은 모든 기준을 먼저 측정한 뒤 고정합니다."))

        ptp = group("PTP · 공통 시각 예약 촬영")
        field(ptp, "ptp_wait_seconds", "시계 동기 대기", double(45., 1, 180, 1, " 초"))
        field(ptp, "ptp_max_offset_us", "PTP offset 허용값", double(10., .001, 1000000, 3, " µs"))
        field(ptp, "ptp_max_timestamp_spread_us", "프레임 시차 허용값", double(100., .001, 1000000, 3, " µs"))
        field(ptp, "action_lead_ms", "예약 선행 시간", integer(200, 50, 5000, " ms"))
        ptp.addRow(caption("PTP 잠금·예약 ACK·프레임 timestamp를 검사합니다. 잠금 실패 시 촬영을 중단합니다."))

        gpio = group("GPIO 트리거")
        self.master = ScrollSafeComboBox()
        field(gpio, "master_serial", "GPIO 마스터 카메라", self.master)
        field(gpio, "output_line", "마스터 출력 라인", QLineEdit("Line2"))
        field(gpio, "input_line", "슬레이브 입력 라인", QLineEdit("Line0"))
        edge = ScrollSafeComboBox()
        edge.addItem("상승 에지", "RisingEdge")
        edge.addItem("하강 에지", "FallingEdge")
        field(gpio, "trigger_edge", "트리거 에지", edge)
        gpio.addRow(caption("GPIO 모드를 선택할 때 사용합니다. 모델에 맞는 트리거 케이블을 연결해야 합니다."))

        preview = group("미리보기 · 표시 전용")
        field(preview, "preview_red", "화이트밸런스 R", double(1., .1, 8, 4, " ×"))
        field(preview, "preview_blue", "화이트밸런스 B", double(1., .1, 8, 4, " ×"))
        profile = ScrollSafeComboBox()
        profile.addItem("실내 · 인공 조명 / 일반", "Indoor")
        profile.addItem("실외 · 자연광", "Outdoor")
        field(preview, "white_balance_profile", "자동 WB 조명 프로필", profile)
        field(preview, "white_balance_timeout_seconds", "WB 측정 제한시간", double(10., 1, 60, 1, " 초"))
        self.wb_settings_button = QPushButton("WB 1회 측정 · 전체 적용")
        preview.addRow(self.wb_settings_button)
        field(preview, "preview_brightness_ev", "표시 밝기", double(0., -4, 8, 1, " EV"))
        field(preview, "preview_fps", "갱신 속도 상한", double(4., .5, 20, 1, " FPS"))
        preview.addRow(caption("화이트밸런스와 밝기는 화면 표시에만 적용합니다. 저장하는 Bayer RAW 픽셀은 변경하지 않습니다. 실제 속도는 노출시간·연결 속도에 따라 달라집니다."))
        network = group("새 카메라 · 주소 검사와 할당")
        interface = ScrollSafeComboBox()
        interface.addItem("자동 · 카메라가 연결된 유선 NIC", "")
        self.network_interface = field(network, "network_interface", "카메라용 인터페이스", interface)
        field(network, "network_probe_seconds", "ARP 충돌 검사 제한시간", double(4., 1, 10, 1, " 초"))
        field(network, "network_rediscover_seconds", "할당 후 재검색 제한시간", double(12., 2, 60, 1, " 초"))
        network.addRow(caption("촬영 화면의 ‘새 카메라 찾기 · 주소 설정’을 누르면 검사·할당·재검색·RAW 수신 검증·자동 연결을 진행합니다. PC 유선 인터페이스에 IPv4 주소가 있어야 합니다."))
        network.addRow(caption("카메라 주소는 임시 ForceIP로 설정합니다. PC 네트워크 설정과 카메라의 영구 주소는 변경하지 않습니다. 전원을 다시 켠 뒤 필요한 경우 버튼을 다시 누르세요."))
        nikon = group("Nikon Z5 · USB 보조 촬영")
        field(nikon, "dslr_enabled", "촬영 묶음에 추가", QCheckBox("Z5 NEF도 함께 저장"))
        field(nikon, "dslr_port", "USB 포트 · 비우면 자동", QLineEdit())
        field(nikon, "dslr_follow", "설정 연동", QCheckBox("노출시간·게인 환산값 따라가기"))
        self.nikon_reference = field(nikon, "dslr_reference_serial", "따라갈 카메라", ScrollSafeComboBox())
        field(nikon, "dslr_exposure_us", "고정 노출시간", double(20000., 1, 30000000, 1, " µs"))
        field(nikon, "dslr_iso", "고정 / 환산 기준 ISO", integer(100, 25, 204800))
        field(nikon, "dslr_aperture", "조리개 · 지원 선택지 입력", QLineEdit())
        field(nikon, "dslr_white_balance", "WB · 지원 선택지 입력", QLineEdit())
        field(nikon, "dslr_max_quantization_ev", "설정 반올림 허용 오차", double(.2, 0, 1, 3, " EV"))
        field(nikon, "dslr_dispatch_offset_ms", "USB 명령 시각 보정", integer(0, -5000, 5000, " ms"))
        field(nikon, "dslr_timeout_seconds", "촬영·전송 제한시간", double(90., 10, 180, 1, " 초"))
        self.nikon_probe = QPushButton("Z5 검색 · 지원 설정 확인")
        nikon.addRow(self.nikon_probe)
        self.nikon_preview_button = QPushButton("Z5 USB 미리보기 갱신")
        nikon.addRow(self.nikon_preview_button)
        self.nikon_preview = PreviewCanvas()
        self.nikon_preview.setMinimumHeight(180)
        self.nikon_preview.hide()
        nikon.addRow(self.nikon_preview)
        self.nikon_status = caption("Z5를 사진 M 모드로 USB 연결하세요. NEF 단독·무손실 저장을 사용합니다.")
        nikon.addRow(self.nikon_status)
        self.nikon_info = QPlainTextEdit()
        self.nikon_info.setReadOnly(True)
        self.nikon_info.setMaximumHeight(180)
        self.nikon_info.hide()
        nikon.addRow(self.nikon_info)
        nikon.addRow(caption("NEF 원본과 EXIF 실제 설정·RAW 디코딩·SHA-256 검사를 함께 저장합니다. USB는 시계 동기를 지원하지 않으며 명령 시각 보정은 실제 노출 시작 시차를 보장하지 않습니다. ISO = 기준 ISO × 10^(게인 dB/20)은 명목 환산입니다. 조리개/WB를 비우면 현재 값을 유지·기록합니다."))
        self.settings_columns = AdaptiveColumns(self.setting_groups)
        layout.addWidget(self.settings_columns)
        profiles = QGroupBox("카메라 특성 · 실험 조건 기록")
        profile_layout = QVBoxLayout(profiles)
        profile_layout.addWidget(caption("검색된 시리얼별로 설정합니다. RAW 모드 표기는 사용자 기록이며 HDR 동작을 변경하지 않습니다. IMX490은 정확한 카메라 모델과 RAW API를 먼저 확인해야 합니다."))
        self.profile_table = QTableWidget(0, 6)
        self.profile_table.setHorizontalHeaderLabels(["시리얼", "센서", "RAW 모드 표기", "변환 게인", "렌즈 조리개", "실험 메모"])
        self.profile_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.profile_table.horizontalHeader().setStretchLastSection(True)
        self.profile_table.setMinimumHeight(165)
        profile_layout.addWidget(self.profile_table)
        layout.addWidget(profiles)
        self.specs_view = QPlainTextEdit(comparison_text())
        self.specs_view.setReadOnly(True)
        self.specs_view.setMinimumHeight(250)
        layout.addWidget(self.specs_view)
        layout.addStretch()
        self.settings_page = page
        self.pages.addWidget(scroll(page))

    def _build_history(self):
        page = QWidget()
        page.setObjectName("pageBody")
        layout = QVBoxLayout(page)
        layout.setContentsMargins(24, 20, 24, 24)
        self.history_page = page
        self.history_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.history_splitter.setChildrenCollapsible(False)
        self.history_splitter.setHandleWidth(8)
        self.history_split_sizes = {Qt.Orientation.Horizontal: [640, 360], Qt.Orientation.Vertical: [620, 380]}
        self.history_splitter.splitterMoved.connect(self._remember_history_split)
        layout.addWidget(self.history_splitter)
        results = QWidget()
        results.setObjectName("pageBody")
        result_layout = QVBoxLayout(results)
        result_layout.setContentsMargins(0, 0, 0, 0)
        result_layout.setSpacing(16)
        completed = QFrame()
        completed.setObjectName("whitePanel")
        info = QVBoxLayout(completed)
        info.setContentsMargins(20, 20, 20, 20)
        info.addWidget(caption("마지막 완료 데이터셋", "sectionTitle"))
        self.completed_label = caption("아직 촬영한 데이터셋이 없습니다.")
        self.completed_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        info.addWidget(self.completed_label)
        self.open_output = QPushButton("저장 폴더 열기  ↗")
        info.addWidget(self.open_output, 0, Qt.AlignmentFlag.AlignLeft)
        verify = QGroupBox("저장된 데이터셋 무결성 재검사")
        verify_layout = QVBoxLayout(verify)
        row = QHBoxLayout()
        self.verify_path = QLineEdit()
        self.verify_path.setPlaceholderText("manifest.json과 COMPLETE.json이 있는 데이터셋 폴더")
        self.verify_browse = QPushButton("선택")
        self.verify_button = QPushButton("전체 검사")
        row.addWidget(self.verify_path, 1)
        row.addWidget(self.verify_browse)
        row.addWidget(self.verify_button)
        verify_layout.addLayout(row)
        self.verify_status = caption("파일·메타데이터 SHA-256, RAW 크기, 카메라 수와 촬영 수를 검사합니다.")
        verify_layout.addWidget(self.verify_status)
        self.history_top = AdaptiveColumns([completed, verify], minimum_column_width=380, maximum_columns=2)
        result_layout.addWidget(self.history_top)
        network = QGroupBox("카메라 주소 검사 결과")
        self.history_network = network
        network_layout = QVBoxLayout(network)
        self.network_summary = caption("새 카메라 찾기 · 주소 설정을 실행하면 결과가 표시됩니다.")
        network_layout.addWidget(self.network_summary)
        self.network_table = QTableWidget(0, 6)
        self.network_table.setHorizontalHeaderLabels(["카메라", "MAC 주소", "인터페이스", "이전 IP", "현재 IP", "검증 결과"])
        self.network_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.network_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.network_table.verticalHeader().hide()
        self.network_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.network_table.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeMode.Stretch)
        self.network_table.setAlternatingRowColors(True)
        network_layout.addWidget(self.network_table)
        self.network_table.hide()
        self.network_camera_details = caption("카메라 행을 선택하면 RAW·동기 촬영 지원 정보를 확인할 수 있습니다.", "cameraDetail")
        network_layout.addWidget(self.network_camera_details)
        self.network_camera_details.hide()
        self.network_table.itemSelectionChanged.connect(self._network_selection_changed)
        self.network_report_view = QPlainTextEdit()
        self.network_report_view.setReadOnly(True)
        self.network_report_view.setPlaceholderText("‘새 카메라 찾기 · 주소 설정’을 실행하면 인터페이스·IP·MAC·할당·RAW 검증 결과가 표시됩니다.")
        self.network_report_view.setMinimumHeight(160)
        self.network_report_view.setMaximumHeight(260)
        network_layout.addWidget(self.network_report_view)
        self.network_report_view.hide()
        self.network_details = QPushButton("세부 검사 정보 보기")
        self.network_details.setCheckable(True)
        self.network_details.toggled.connect(self.network_report_view.setVisible)
        self.network_save = QPushButton("주소 검사 보고서 저장")
        actions = QHBoxLayout()
        actions.addWidget(self.network_details)
        actions.addWidget(self.network_save)
        actions.addStretch()
        network_layout.addLayout(actions)
        result_layout.addWidget(network)
        result_layout.addStretch(1)
        self.history_results = scroll(results)
        self.history_results.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.history_results.setMinimumSize(0, 80)
        self.history_results.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.history_splitter.addWidget(self.history_results)
        self.history_log_panel = QFrame()
        self.history_log_panel.setObjectName("whitePanel")
        log_layout = QVBoxLayout(self.history_log_panel)
        log_layout.setContentsMargins(16, 16, 16, 16)
        log_layout.addWidget(caption("진행 기록", "sectionTitle"))
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(2000)
        self.log.setMinimumSize(0, 72)
        self.log.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        log_layout.addWidget(self.log, 1)
        self.history_splitter.addWidget(self.history_log_panel)
        self.pages.addWidget(page)

    def _bind(self):
        self.noise_panel.capture_requested.connect(self._noise_capture)
        self.noise_panel.analysis_requested.connect(lambda path: self._command("noise_analyze", (path, Event())))
        self.noise_panel.comparison_requested.connect(lambda paths: self._command("noise_compare", (paths, self.output.text(), Event())))
        self.noise_panel.stop_requested.connect(self._cancel)
        self.noise_panel.notify.connect(self._error)
        self.scan_button.clicked.connect(lambda: self._command("scan"))
        self.connect_button.clicked.connect(self._connect)
        self.disconnect_button.clicked.connect(lambda: self._command("disconnect"))
        self.network_button.clicked.connect(self._network_setup)
        self.nikon_probe.clicked.connect(self._probe_nikon)
        self.nikon_preview_button.clicked.connect(self._preview_nikon)
        self.nikon_preview.clicked.connect(lambda: self.nikon_preview.setMinimumHeight(180 if self.nikon_preview.minimumHeight() > 180 else 420))
        self.network_save.clicked.connect(self._save_network_report)
        self.camera_list.itemChanged.connect(lambda *_: self._controls())
        self.capture_button.clicked.connect(self._capture)
        self.meter_button.clicked.connect(self._meter)
        self.wb_button.clicked.connect(self._white_balance_once)
        self.wb_settings_button.clicked.connect(self._white_balance_once)
        self.stop_button.clicked.connect(self._cancel)
        self.apply_preview.clicked.connect(self._apply_preview)
        self.preview_check.toggled.connect(self._preview_toggled)
        self.fullscreen_button.clicked.connect(self.toggle_fullscreen)
        self.inspector_toggle.clicked.connect(self._toggle_inspector)
        self.inspector_close.clicked.connect(self._toggle_inspector)
        self.load_button.clicked.connect(self._load_dialog)
        self.save_button.clicked.connect(self._save_dialog)
        self.output_browse.clicked.connect(self._choose_output)
        self.verify_browse.clicked.connect(self._choose_verify)
        self.verify_button.clicked.connect(lambda: self._command("verify", self.verify_path.text().strip()))
        self.open_output.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(self.last_result or self.output.text()).expanduser().resolve()))))
        for card in self.cards:
            card.activated.connect(self._zoom)
        self.focus_image.clicked.connect(self._unzoom)
        self.back_button.clicked.connect(self._unzoom)
        self.auto.toggled.connect(self._auto_changed)
        self.strategy.currentIndexChanged.connect(self._strategy_changed)
        self.cross_bracket.toggled.connect(self._strategy_changed)
        self.config_widgets["range_policy"].currentIndexChanged.connect(self._auto_parameters_changed)
        self.config_widgets["dslr_enabled"].toggled.connect(self._controls)
        for widget in (self.base_exp, self.base_gain, self.high_factor, self.gain_increase):
            widget.valueChanged.connect(self._bracket_changed)
        for widget in (self.low_factor, self.low_gain_offset, self.reference, self.config_widgets["meter_seconds"],
                       self.config_widgets["meter_target_percent"], self.config_widgets["meter_max_exposure_us"], self.config_widgets["meter_max_gain_db"]):
            signal_ = widget.currentIndexChanged if isinstance(widget, QComboBox) else widget.valueChanged
            signal_.connect(self._auto_parameters_changed)
        for _, exposure, gain in self.preset_editors:
            exposure.valueChanged.connect(self._manual_changed)
            gain.valueChanged.connect(self._manual_changed)
        for key, widget in self.config_widgets.items():
            if key.startswith("preview_"):
                widget.valueChanged.connect(self._display_changed)
        self.camera_count.valueChanged.connect(self._controls)
        self.repeats.valueChanged.connect(self._controls)
        self.sync.currentIndexChanged.connect(self._sync_changed)
        self.verify_path.textChanged.connect(lambda *_: self._controls())
        self.wb_reference.currentIndexChanged.connect(self._controls)
        for sequence, callback in (("F11", self.toggle_fullscreen), ("Escape", self._escape)):
            action = QAction(self)
            action.setShortcut(QKeySequence(sequence))
            action.triggered.connect(callback)
            self.addAction(action)
        descriptions = {
            self.scan_button: "연결된 카메라를 다시 검색합니다. 현재 연결은 해제됩니다.",
            self.connect_button: "체크한 카메라를 연결하고 실시간 미리보기를 시작합니다.",
            self.disconnect_button: "카메라 획득을 중지하고 장치 연결을 해제합니다.",
            self.network_button: "현재 카메라 연결을 잠시 해제하고 유선 NIC·IP·서브넷·충돌을 검사합니다. 필요한 임시 주소를 할당하고 MAC·IP·RAW 수신을 검증한 뒤 자동 연결합니다.",
            self.nikon_probe: "USB Z5를 검색하고 현재 설정과 지원 선택지를 읽습니다. 사진은 촬영하지 않습니다.",
            self.nikon_preview_button: "USB 미리보기 한 프레임을 가져옵니다. 이미지를 클릭하면 확대·복귀합니다. 미리보기 JPEG는 데이터셋에 저장하지 않습니다.",
            self.network_save: "인터페이스·카메라 주소·MAC·할당·RAW 검증 결과를 JSON 파일로 저장합니다.",
            self.network_details: "카메라별 RAW 검증과 지원 기능을 포함한 상세 검사 기록을 펼치거나 접습니다.",
            self.load_button: "JSON 파일에서 모든 촬영·동기·저장·미리보기 설정을 불러옵니다.",
            self.save_button: "현재 모든 설정을 JSON 파일로 저장합니다.",
            self.meter_button: "현재 장면의 노출시간·게인을 측광하여 설정 1의 기준값으로 사용합니다.",
            self.wb_button: "WB 기준 카메라의 Once 자동 조절이 끝나면 R·B 값을 모든 카메라의 미리보기에 고정 적용합니다. RAW 원본은 보존합니다.",
            self.wb_settings_button: "촬영 화면에서 선택한 WB 기준 카메라로 한 번 측정하고 전체 미리보기에 적용합니다. Continuous 자동 조절을 사용하지 않습니다.",
            self.apply_preview: "선택한 설정의 노출시간·게인을 모든 연결 카메라의 미리보기에 적용합니다.",
            self.stop_button: "측광 또는 촬영을 중지합니다. 미완성 데이터는 .partial 폴더에 남습니다.",
            self.back_button: "확대를 끝내고 이전 카메라 전체 보기로 돌아갑니다.",
            self.output_browse: "RAW 데이터셋을 저장할 폴더를 선택합니다.",
            self.verify_browse: "완료된 데이터셋 폴더를 선택합니다.",
            self.verify_button: "저장된 RAW와 메타데이터의 파일 수·크기·체크섬을 다시 검사합니다.",
            self.open_output: "마지막 완료 데이터셋 또는 현재 저장 폴더를 파일 관리자로 엽니다.",
        }
        for button in self.findChildren(QPushButton):
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.setAttribute(Qt.WidgetAttribute.WA_Hover, True)
            button.setToolTip(descriptions.get(button, button.toolTip() or button.text()))

    @staticmethod
    def _widget_value(widget):
        if isinstance(widget, QCheckBox):
            return widget.isChecked()
        if isinstance(widget, QComboBox):
            return widget.currentData()
        if isinstance(widget, QLineEdit):
            return widget.text().strip()
        return widget.value()

    @staticmethod
    def _set_widget(widget, value):
        blocker = QSignalBlocker(widget)
        if isinstance(widget, QCheckBox):
            widget.setChecked(value)
        elif isinstance(widget, QComboBox):
            index = widget.findData(value)
            if index < 0:
                widget.addItem(str(value), value)
                index = widget.count() - 1
            widget.setCurrentIndex(index)
        elif isinstance(widget, QLineEdit):
            widget.setText(str(value))
        else:
            widget.setValue(value)
        del blocker

    def _editor_presets(self):
        return [Preset(name.text().strip(), exposure.value(), gain.value()) for name, exposure, gain in self.preset_editors]

    def _set_presets(self, presets):
        for (name, exposure, gain), preset in zip(self.preset_editors, presets):
            for widget, value in ((name, preset.name), (exposure, preset.exposure_us), (gain, preset.gain_db)):
                self._set_widget(widget, value)

    def _config(self):
        from dataclasses import asdict
        from .noise import NoiseConfig
        noise = self.noise_panel.settings()
        values = {key: self._widget_value(widget) for key, widget in self.config_widgets.items()}
        values.update(auto_reference=self.auto.isChecked(), reference_serial=self.reference.currentData() or "",
                      capture_strategy=self.strategy.currentData(), cross_bracket=self.cross_bracket.isChecked(),
                      camera_profiles=self._profiles(),
                      white_balance_calibration=self.last_white_balance.copy(),
                      noise_config=asdict(noise) if self.noise_config_explicit or noise != NoiseConfig.fixed_curve() else {},
                      sync_mode=self.sync.currentData(), low_exposure_factor=self.low_factor.value(),
                      high_exposure_factor=self.low_factor.value() * self.high_factor.value(),
                      low_gain_offset_db=self.low_gain_offset.value(),
                      high_gain_offset_db=self.low_gain_offset.value() + self.gain_increase.value(),
                      presets=self._editor_presets())
        config = CaptureConfig(**values)
        config.validate()
        return config

    def _load_config(self, config):
        config.validate()
        self.loading = True
        self.noise_config_explicit = bool(config.noise_config)
        self.noise_panel.load_config(config.noise_config)
        for key, widget in self.config_widgets.items():
            self._set_widget(widget, getattr(config, key))
        self._set_widget(self.sync, config.sync_mode)
        self._set_widget(self.reference, config.reference_serial)
        self._set_widget(self.strategy, config.capture_strategy)
        self._set_widget(self.cross_bracket, config.cross_bracket)
        self.saved_profiles = config.camera_profiles.copy()
        self.last_white_balance = config.white_balance_calibration.copy()
        self._fill_profiles()
        self.auto.setChecked(config.auto_reference)
        self._set_widget(self.high_factor, config.high_exposure_factor / config.low_exposure_factor)
        self._set_widget(self.gain_increase, config.high_gain_offset_db - config.low_gain_offset_db)
        self._set_presets(config.presets)
        self._set_widget(self.base_exp, config.presets[0].exposure_us)
        self._set_widget(self.base_gain, config.presets[0].gain_db)
        self.resolved_presets, self.last_reference = None, None
        self.loading = False
        self._display_changed()
        self._auto_changed()
        self._strategy_changed()
        self._sync_changed()
        self._controls()

    def save_settings(self, path):
        config = self._config()
        target = Path(path).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".tmp")
        temporary.write_text(json.dumps(config.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(target)
        self._log(f"촬영 설정 저장: {target.resolve()}")
        return config

    def load_settings(self, path):
        config = CaptureConfig.from_dict(json.loads(Path(path).expanduser().read_text(encoding="utf-8")))
        self._load_config(config)
        self._log(f"촬영 설정 불러오기: {Path(path).resolve()}")

    def _save_dialog(self):
        path, _ = QFileDialog.getSaveFileName(self, "촬영 설정 저장", "local_config.json", "JSON (*.json)")
        if path:
            try:
                self.save_settings(path)
            except (OSError, ValueError, CaptureError) as error:
                self._error(str(error))

    def _load_dialog(self):
        path, _ = QFileDialog.getOpenFileName(self, "촬영 설정 불러오기", "", "JSON (*.json)")
        if path:
            try:
                self.load_settings(path)
            except (OSError, ValueError, TypeError, CaptureError) as error:
                self._error(str(error))

    def _choose_output(self):
        path = QFileDialog.getExistingDirectory(self, "RAW 저장 폴더", self.output.text())
        if path:
            self.output.setText(path)

    def _choose_verify(self):
        path = QFileDialog.getExistingDirectory(self, "완료 데이터셋 선택", self.last_result or self.output.text())
        if path:
            self.verify_path.setText(path)

    def _auto_changed(self, *_):
        if self.loading:
            return
        auto = self.auto.isChecked()
        self.manual_hint.setText("자동 모드: 촬영 직전 기준 측광 후 네 설정의 실제 값을 갱신합니다." if auto else "수동 모드: 네 설정의 노출시간·게인과 저장 이름을 각각 변경할 수 있습니다.")
        self.baseline_hint.setText("자동 측광값 = 설정 1의 기준값. 측광 후 실제 값을 표시합니다." if auto else "입력한 기존값 = 설정 1. 배수·증가량을 바꾸면 네 설정에 함께 적용합니다.")
        self.resolved_presets = None
        self.last_reference = None
        if not auto:
            self._set_widget(self.base_exp, self.preset_editors[0][1].value())
            self._set_widget(self.base_gain, self.preset_editors[0][2].value())
        self._refresh_sequence()
        self._controls()

    def _strategy_changed(self, *_):
        if self.loading:
            return
        mode = self.strategy.currentData()
        self.strategy_hint.setText({"bracket": "기준 1대의 측정값으로 전체 카메라를 네 설정에서 촬영합니다.",
            "cross_reference": "각 카메라의 자동 측정값을 먼저 고정하고, 기준 카메라를 바꿔가며 전체 카메라에 같은 값을 적용합니다.",
            "per_camera": "각 카메라가 자신의 자동 측정값으로 촬영합니다. 같은 장면에서 각 센서의 실용 성능을 비교합니다."}[mode])
        self.cross_bracket.setVisible(mode == "cross_reference")
        self.auto.setVisible(mode == "bracket")
        self.base_caption.setVisible(mode == "bracket")
        for widget in (self.base_exp, self.base_gain):
            widget.setVisible(mode == "bracket")
            self.base_form.labelForField(widget).setVisible(mode == "bracket")
        expanded = mode == "bracket" or (mode == "cross_reference" and self.cross_bracket.isChecked())
        for widget in (self.high_factor, self.gain_increase):
            widget.setVisible(expanded)
            self.base_form.labelForField(widget).setVisible(expanded)
        if mode != "bracket":
            self.baseline_hint.setText("모든 카메라의 기준을 측광하면 실제 적용 계획을 표시합니다.")
        if mode == "bracket":
            blocker = QSignalBlocker(self.preview_preset)
            self.preview_preset.clear()
            self.preview_preset.addItems(["설정 1 · 기준값", "설정 2", "설정 3", "설정 4"])
            del blocker
        self.resolved_presets = None
        self.last_reference = None
        self._refresh_sequence()
        self._controls()

    def _bracket_changed(self, *_):
        if self.loading:
            return
        if not self.auto.isChecked():
            names = [row[0].text() for row in self.preset_editors]
            presets = bracket_presets(self.base_exp.value(), self.base_exp.value() * self.high_factor.value(),
                                      self.base_gain.value(), self.base_gain.value() + self.gain_increase.value())
            self._set_presets([replace(preset, name=name) for preset, name in zip(presets, names)])
        else:
            self.resolved_presets = None
            self.last_reference = None
        self._refresh_sequence()
        self._controls()

    def _auto_parameters_changed(self, *_):
        if self.loading:
            return
        if self.auto.isChecked() or self.strategy.currentData() != "bracket":
            self.resolved_presets = None
            self.last_reference = None
            self._refresh_sequence()
            self._controls()

    def _manual_changed(self, *_):
        if self.loading or self.auto.isChecked():
            return
        self._set_widget(self.base_exp, self.preset_editors[0][1].value())
        self._set_widget(self.base_gain, self.preset_editors[0][2].value())
        self._refresh_sequence()

    def _display_changed(self, *_):
        if self.loading:
            return
        self.worker.display_options = tuple(self.config_widgets[key].value() for key in ("preview_red", "preview_blue", "preview_brightness_ev"))
        self.worker.preview_fps = self.config_widgets["preview_fps"].value()
        red, blue, _ = self.worker.display_options
        if self.last_white_balance and (self.last_white_balance.get("red") != red or self.last_white_balance.get("blue") != blue):
            self.last_white_balance = {}
        if self.last_white_balance:
            self.wb_status.setText(f"기준 {self.last_white_balance['reference_serial']} · 1회 측정 완료\nR {red:.4f} / B {blue:.4f} · 전체 미리보기 고정")
        else:
            self.wb_status.setText(f"수동 WB · R {red:.4f} / B {blue:.4f}\n전체 미리보기 공통 적용")

    def _sync_changed(self, *_):
        hints = {"ptp": "PTP 시계와 공통 시각 예약 촬영을 사용합니다.",
                 "software": "병렬 소프트웨어 명령으로 촬영합니다. 시차를 기록합니다.",
                 "hardware": "GPIO 마스터의 ExposureActive 출력으로 동기 촬영합니다."}
        self.sync_hint.setText(hints[self.sync.currentData()])

    def _refresh_sequence(self):
        mode = self.strategy.currentData()
        alternative = mode != "bracket"
        for tile, _ in self.sequence_tiles:
            tile.setVisible(not alternative)
        self.plan_table.setVisible(alternative)
        self.preset_group.setVisible(not alternative or (mode == "cross_reference" and self.cross_bracket.isChecked()))
        self.sequence_note.setText("모든 기준을 먼저 측정해 고정합니다. 설정 → 전체 카메라 동시 촬영 → 다음 기준으로 자동 진행합니다." if alternative else "기준값은 낮은 노출시간·낮은 게인입니다. 나머지 설정에서 값을 높입니다.")
        if alternative:
            self.baseline_status.setText("각자 자동 측광값" if mode == "per_camera" else "모든 카메라의 기준값 교차 적용")
            plan = (self.last_reference or {}).get("capture_plan", [])
            self.plan_table.setRowCount(sum(len(step["applied"]) for step in plan) if plan else 1)
            if not plan:
                for col, text in enumerate(("측광 대기", "선택한 전체 카메라", "—", "—")):
                    self.plan_table.setItem(0, col, QTableWidgetItem(text))
            row = 0
            for step in plan:
                for serial, preset in step["applied"].items():
                    values = [step["reference_serial"], serial, f"{preset['exposure_us']:,.1f}", f"{preset['gain_db']:.2f}"]
                    for col, text in enumerate(values):
                        item = QTableWidgetItem(text)
                        item.setToolTip(step["name"] + (" · 지원 범위 제한됨" if step["range_clipped"] else ""))
                        self.plan_table.setItem(row, col, item)
                    row += 1
            return
        auto = self.auto.isChecked()
        pending = auto and self.resolved_presets is None
        for widget in (self.base_exp, self.base_gain):
            widget.setSpecialValueText("측광 대기" if pending else "")
        if pending:
            self._set_widget(self.base_exp, self.base_exp.minimum())
            self._set_widget(self.base_gain, self.base_gain.minimum())
        if auto and self.resolved_presets is None:
            factor, increase = self.high_factor.value(), self.gain_increase.value()
            text = [("T₀ µs", "G₀ dB"), (f"T₀ × {factor:g} µs", "G₀ dB"),
                    ("T₀ µs", f"G₀ + {increase:g} dB"), (f"T₀ × {factor:g} µs", f"G₀ + {increase:g} dB")]
            self.baseline_status.setText("설정 1 = 측광 기준값")
        else:
            presets = self.resolved_presets if auto else self._editor_presets()
            text = [(f"{preset.exposure_us:,.1f} µs", f"{preset.gain_db:.2f} dB") for preset in presets]
            self.baseline_status.setText("설정 1 = 확정 기준값" if auto else "설정 1 = 입력한 기존값")
        for (_, label), (exposure, gain) in zip(self.sequence_tiles, text):
            label.setText(f"노출시간 {exposure}\n게인 {gain}")

    def _resolved(self, presets, reference):
        self.resolved_presets, self.last_reference = presets, reference
        if self.strategy.currentData() != "bracket":
            plan = reference["capture_plan"]
            self.baseline_hint.setText(f"기준 {len(reference['references'])}대 측광 완료 · {len(plan)}개 촬영 그룹" + (" · 범위 제한 포함" if any(s["range_clipped"] for s in plan) else ""))
            blocker = QSignalBlocker(self.preview_preset)
            self.preview_preset.clear()
            self.preview_preset.addItems([step["name"] for step in plan])
            del blocker
            self._refresh_sequence()
            self._controls()
            if self.active_command == "meter":
                self.state.setText(f"전체 기준 측광 완료 · {len(plan)}개 촬영 그룹 계획 확정")
            return
        self._set_presets(presets)
        self._set_widget(self.base_exp, presets[0].exposure_us)
        self._set_widget(self.base_gain, presets[0].gain_db)
        if reference:
            clipped = " · 공통 지원 범위로 제한됨" if reference.get("range_clipped") else ""
            self.baseline_hint.setText(f"측광 완료 · 설정 1: {presets[0].exposure_us:,.1f} µs / {presets[0].gain_db:.2f} dB{clipped}")
        if self.active_command == "meter":
            self.state.setText(f"기준 측광 완료 · 노출시간 {presets[0].exposure_us:,.1f} µs · 게인 {presets[0].gain_db:.2f} dB")
        self._refresh_sequence()

    def _command(self, command, payload=None):
        if self.busy:
            return
        self.inspector_scroll.lock_scroll()
        self.noise_panel.inspector.lock_scroll()
        if command.startswith("noise_"):
            self.worker.cancel = payload[-1]
        self.busy, self.active_command = True, command
        self.command_failed = False
        self._controls()
        self.worker.submit(command, payload)

    def _connect(self):
        serials = [self.camera_list.item(index).data(Qt.ItemDataRole.UserRole) for index in range(self.camera_list.count())
                   if self.camera_list.item(index).checkState() == Qt.CheckState.Checked]
        if serials:
            self._command("connect", serials)

    def _fill_serials(self, cameras):
        for combo in (self.reference, self.master, self.nikon_reference, self.wb_reference):
            current = combo.currentData() or ""
            blocker = QSignalBlocker(combo)
            combo.clear()
            combo.addItem("첫 번째 카메라", "")
            for camera in cameras:
                combo.addItem(camera.serial, camera.serial)
            if current and combo.findData(current) < 0:
                combo.addItem(f"{current} (미연결)", current)
            combo.setCurrentIndex(max(0, combo.findData(current)))
            del blocker

    def _profiles(self):
        result = self.saved_profiles.copy()
        for serial, fields in self.profile_rows.items():
            sensor, mode, conversion, aperture, notes = fields
            value = {**result.get(serial, {}), "sensor_hint": sensor.text().strip(), "raw_mode": mode.currentData(),
                     "gain_conversion_mode": conversion.currentData(), "lens_aperture": aperture.text().strip(), "notes": notes.text().strip()}
            defaults = {"sensor_hint": "", "raw_mode": "unknown", "gain_conversion_mode": "preserve", "lens_aperture": "", "notes": ""}
            value = {key: item for key, item in value.items() if key in result.get(serial, {}) or item != defaults.get(key)}
            if serial in result or value:
                result[serial] = value
        return result

    def _fill_profiles(self, cameras=None):
        if cameras is not None:
            self.saved_profiles = self._profiles()
            self.profile_cameras = cameras
        serials = list(dict.fromkeys([c.serial for c in getattr(self, "profile_cameras", [])] + list(self.saved_profiles)))
        self.profile_table.setRowCount(len(serials))
        self.profile_rows = {}
        for row, serial in enumerate(serials):
            profile = self.saved_profiles.get(serial, {})
            item = QTableWidgetItem(serial)
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.profile_table.setItem(row, 0, item)
            sensor = QLineEdit(profile.get("sensor_hint", ""))
            mode = ScrollSafeComboBox()
            for label, value in (("미확인", "unknown"), ("선형 RAW", "linear"), ("HDR RAW", "hdr")):
                mode.addItem(label, value)
            conversion = ScrollSafeComboBox()
            for label, value in (("현재 값 유지", "preserve"), ("Low · 포화 용량", "Low"), ("High · 낮은 읽기 잡음", "High")):
                conversion.addItem(label, value)
            self._set_widget(mode, profile.get("raw_mode", "unknown"))
            self._set_widget(conversion, profile.get("gain_conversion_mode", "preserve"))
            fields = (sensor, mode, conversion, QLineEdit(profile.get("lens_aperture", "")), QLineEdit(profile.get("notes", "")))
            self.profile_rows[serial] = fields
            for col, widget in enumerate(fields, 1):
                self.profile_table.setCellWidget(row, col, widget)
                signal_ = widget.currentIndexChanged if isinstance(widget, QComboBox) else widget.textChanged
                signal_.connect(self._auto_parameters_changed)

    def _white_balance_once(self):
        try:
            config = CaptureConfig(white_balance_reference_serial=self.wb_reference.currentData() or "",
                white_balance_profile=self.config_widgets["white_balance_profile"].currentData(),
                white_balance_timeout_seconds=self.config_widgets["white_balance_timeout_seconds"].value())
            config.validate()
        except CaptureError as error:
            self._error(str(error))
            return
        if self.busy:
            return
        self.worker.cancel = Event()
        self.wb_status.setText("기준 카메라 WB 1회 측정 중…")
        self._command("white_balance", (config, self.worker.cancel))

    def _white_balance_result(self, report):
        self._set_widget(self.config_widgets["preview_red"], report["red"])
        self._set_widget(self.config_widgets["preview_blue"], report["blue"])
        self.last_white_balance = report.copy()
        self._display_changed()
        self.state.setText(f"WB 1회 조절 완료 · 기준 {report['reference_serial']} · 전체 {len(report['applied_serials'])}대 고정 적용")

    def _probe_nikon(self):
        config = CaptureConfig(dslr_port=self.config_widgets["dslr_port"].text().strip())
        self.worker.cancel = Event()
        self.nikon_status.setText("USB Z5 검색·지원 설정 확인 중…")
        self._command("nikon_probe", (config, self.worker.cancel))

    def _nikon_result(self, report):
        self.nikon_status.setText(f"{report['camera']['model']} · {report['camera']['port']} · 지원 설정 읽기 완료")
        self._set_widget(self.config_widgets["dslr_port"], report["camera"]["port"])
        lines = []
        for name, widget in report["widgets"].items():
            if name in ("shutterspeed", "iso", "autoiso", "isoauto", "f-number", "aperture", "whitebalance", "imagequality", "rawcompression", "expprogram"):
                lines.append(f"{name}: {widget.get('current', '—')}\n지원값: {', '.join(widget['choices'])}")
        self.nikon_info.setPlainText("\n\n".join(lines))
        self.nikon_info.show()

    def _preview_nikon(self):
        config = CaptureConfig(dslr_port=self.config_widgets["dslr_port"].text().strip())
        self.worker.cancel = Event()
        self._command("nikon_preview", (config, self.worker.cancel))

    def _nikon_image(self, image):
        self.nikon_preview.set_image(image)
        self.nikon_preview.show()
        self.nikon_status.setText("Z5 USB 미리보기 갱신 완료 · 클릭하여 확대·복귀 · 데이터셋은 NEF로 저장")

    def _discovered(self, cameras):
        blocker = QSignalBlocker(self.camera_list)
        self.camera_list.clear()
        for index, camera in enumerate(cameras):
            item = QListWidgetItem(f"{camera.serial}  ·  {camera.model}")
            item.setData(Qt.ItemDataRole.UserRole, camera.serial)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked if index < self.camera_count.value() else Qt.CheckState.Unchecked)
            self.camera_list.addItem(item)
        del blocker
        self._fill_serials(cameras)
        self._fill_profiles(cameras)
        self.state.setText(f"카메라 {len(cameras)}대 발견 · 촬영할 카메라를 선택하고 연결하세요.")
        self._log(f"카메라 {len(cameras)}대 검색됨")

    def _connected(self, cameras):
        self.cameras = cameras
        self.overview.set_count(len(cameras))
        self.resolved_presets = None
        if self.zoomed_index is not None:
            self._unzoom()
        for card in self.cards:
            card.clear()
        if cameras:
            self._fill_serials(cameras)
            for card, camera in zip(self.cards, cameras):
                card.model.setText(f"{camera.serial} · {camera.model}")
                card.status.setText("연결됨")
            self.state.setText(f"카메라 {len(cameras)}대 연결 · 실시간 미리보기")
            self.worker.preview_enabled = self.preview_check.isChecked()
        self._refresh_sequence()
        self._controls()

    def _images(self, images):
        by_serial = {row[0].serial: row for row in images}
        images = [by_serial[c.serial] for c in self.cameras if c.serial in by_serial]
        self.noise_panel.update_images(images)
        for index, (camera, image, settings, frame_id) in enumerate(images[:3]):
            self.cards[index].update_frame(camera, image, settings, frame_id)
            if self.zoomed_index == index:
                self.focus_image.set_image(image)
                self.focus_title.setText(f"CAMERA {index+1:02d}  ·  {camera.serial}")
                self.focus_detail.setText(f"{camera.model}  ·  노출시간 {settings['exposure_us']:,.1f} µs  ·  게인 {settings['gain_db']:.2f} dB  ·  다시 클릭하면 전체 보기")

    def _noise_capture(self, phase, noise, session_path):
        try:
            settings = {key: self._widget_value(self.config_widgets[key]) for key in (
                "timeout_ms", "settle_ms", "warmup_frames", "auto_retry_count", "retry_delay_ms",
                "ptp_wait_seconds", "ptp_max_offset_us", "ptp_max_timestamp_spread_us", "action_lead_ms")}
            profiles=self._profiles()
            if noise.protocol == 'exposure_sweep':
                for camera in self.cameras:
                    profiles[camera.serial]={**profiles.get(camera.serial, {}), 'black_level_percent':1.0, 'black_level_clamping':False}
                    if len(self.cameras) == 1:
                        profiles[camera.serial]['link_throughput_bytes_per_second']=90000000
            config = CaptureConfig(output_dir=self.output.text(), expected_cameras=max(1, len(self.cameras)),
                sync_mode=noise.sync_mode, auto_reference=False, camera_profiles=profiles, **settings)
            event = Event()
            self._command("noise_" + phase, (config, noise, session_path, event))
        except CaptureError as error:
            self._error(str(error))

    def _meter(self):
        try:
            config = self._config()
        except CaptureError as error:
            self._error(str(error))
            return
        self.worker.cancel = Event()
        self.state.setText("현재 장면의 기준 노출시간·게인을 측광하고 있습니다…")
        self._command("meter", (config, self.worker.cancel))

    def _capture(self):
        try:
            config = self._config()
        except CaptureError as error:
            self._error(str(error))
            return
        self.progress_bar.setRange(0, config.planned_groups * config.repeats)
        self.progress_bar.setValue(0)
        self.worker.cancel = Event()
        self.state.setText("기준 측광 → 촬영 계획 적용 → RAW 무결성 검사")
        self._command("capture", (config, self.worker.cancel))

    def _network_setup(self):
        try:
            config = CaptureConfig(expected_cameras=self.camera_count.value(),
                network_interface=self.network_interface.currentData() or "",
                network_probe_seconds=self.config_widgets["network_probe_seconds"].value(),
                network_rediscover_seconds=self.config_widgets["network_rediscover_seconds"].value(),
                auto_retry_count=self.config_widgets["auto_retry_count"].value(),
                retry_delay_ms=self.config_widgets["retry_delay_ms"].value())
            config.validate()
        except CaptureError as error:
            self._error(str(error))
            return
        self.worker.cancel = Event()
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.network_status.setText("네트워크와 카메라 주소를 검사하고 있습니다…")
        self.state.setText("새 카메라 검색 → 주소 검사·할당 → 재검색 → RAW 검증")
        self._command("network", (config, self.worker.cancel))

    def _network_result(self, report):
        self.last_network_report = report
        self.network_report_view.setPlainText(json.dumps(report, ensure_ascii=False, indent=2))
        verified = sum(row["verified"] for row in report["cameras"])
        assigned = sum(row["assigned"] for row in report["cameras"])
        prefix = "DEMO · " if report.get("simulated") else ""
        if report["passed"]:
            text = f"{prefix}주소·RAW 검증 통과 {verified}대 · 주소 할당 {assigned}대"
            recovered = sum(bool(row.get("recovered")) for row in report["cameras"])
            if recovered:
                text += f" · 자동 복구 {recovered}대"
            self.progress_bar.setRange(0, 1)
            self.progress_bar.setValue(1)
        else:
            text = f"{prefix}검증 통과 {verified}대 · " + (report["errors"][0] if report["errors"] else "주소 검사 미완료")
            actions = list(dict.fromkeys(row.get("diagnosis", {}).get("action", "") for row in report["cameras"] if not row["verified"]))
            if any(actions):
                text += "\n다음 조치: " + " / ".join(action for action in actions if action)
            self.progress_bar.setRange(0, 1)
            self.progress_bar.setValue(0)
        self.network_status.setText(text)
        self.network_status.setToolTip(text)
        hosts = [f"{item['name']}  {item['address']}/{item['prefix']}" for item in report["interfaces"] if item["wired"]]
        summary = text + ("\nPC 유선 연결: " + " · ".join(hosts) if hosts else "")
        connected_count = len(report.get("connected_serials", []))
        if report["passed"] and connected_count < self.camera_count.value():
            summary += f"\n촬영 준비 {connected_count}/{self.camera_count.value()}대 · 카메라를 추가 연결하거나 ‘필요한 카메라 수’를 변경하세요."
        if report.get("diagnostic_path"):
            summary += "\n진단 자동 저장: " + report["diagnostic_path"]
        self.network_summary.setText(summary)
        self.network_table.setRowCount(len(report["cameras"]))
        for index, row in enumerate(report["cameras"]):
            diagnosis = row.get("diagnosis", {})
            status = "RAW 수신 검증 통과" + (" · 자동 복구 완료" if row.get("recovered") else "") if row["verified"] else diagnosis.get("category", "확인 필요") + "\n" + row["error"]
            if diagnosis.get("action"):
                status += "\n" + diagnosis["action"]
            values = [row["serial"] or row["model"], row["mac"], row["interface"], row["original_ip"], row["ip"] or "미확인", status]
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setToolTip(f"{row['model']}\n현재 IP: {row['ip'] or '미확인'} / {row['mask'] or '미확인'}\n요청 IP: {row.get('requested_ip', '유지')}\n" + values[-1])
                self.network_table.setItem(index, column, item)
            self.network_table.setRowHeight(index, 34 if row["verified"] else 96)
        self._fit_network_table()
        if self.network_table.rowCount():
            self.network_table.selectRow(0)
            self._network_selection_changed()
        else:
            self.network_camera_details.hide()
        self.state.setText(text)
        self._log(text)
        current = self.network_interface.currentData() or ""
        blocker = QSignalBlocker(self.network_interface)
        self.network_interface.clear()
        self.network_interface.addItem("자동 · 카메라가 연결된 유선 NIC", "")
        for interface in report["interfaces"]:
            if interface["wired"] and self.network_interface.findData(interface["name"]) < 0:
                self.network_interface.addItem(f"{interface['name']} · {interface['address']}/{interface['prefix']}", interface["name"])
        if current and self.network_interface.findData(current) < 0:
            self.network_interface.addItem(f"{current} (미연결)", current)
        self.network_interface.setCurrentIndex(max(0, self.network_interface.findData(current)))
        del blocker
        rows = {row["serial"]: row for row in report["cameras"]}
        for index in range(self.camera_list.count()):
            item = self.camera_list.item(index)
            row = rows.get(item.data(Qt.ItemDataRole.UserRole))
            if row:
                item.setToolTip(f"{row['ip']} · {row['mask']}\nMAC {row['mac']}\n" + ("주소·RAW 검증 통과" if row["verified"] else row["error"]))
                item.setText(f"{row['serial']} · {row['ip'] or 'IP 미확인'}" + (" ✓" if row["verified"] else " · " + row.get("diagnosis", {}).get("category", "확인 필요")))
                if not row["verified"]:
                    item.setCheckState(Qt.CheckState.Unchecked)

    def _fit_network_table(self):
        table = self.network_table
        table.setVisible(table.rowCount() > 0)
        height = table.horizontalHeader().height() + table.frameWidth() * 2
        height += sum(table.rowHeight(row) for row in range(table.rowCount()))
        if table.horizontalScrollBar().maximum() > 0:
            height += table.horizontalScrollBar().sizeHint().height()
        table.setFixedHeight(min(260, max(60, height)))

    def _network_selection_changed(self):
        if self.last_network_report is None:
            return
        index = self.network_table.currentRow()
        rows = self.last_network_report["cameras"]
        if not 0 <= index < len(rows):
            self.network_camera_details.hide()
            return
        camera = rows[index]
        validation = camera.get("validation", {})
        lines = [f"{camera['model']} · {camera['serial']}"]
        shape = validation.get("shape", [])
        if len(shape) == 2:
            lines.append(f"{validation.get('pixel_format', 'RAW')} · {shape[1]:,} × {shape[0]:,} 픽셀")
        if validation.get("raw_verified"):
            samples = validation.get("consecutive_samples", len(validation.get("verified_samples", [])))
            raw_status = "RAW 수신 검증 통과"
            if samples:
                raw_status += f" · 연속 {samples}장"
            if "crc_passed" in validation:
                raw_status += " · CRC " + ("통과" if validation["crc_passed"] else "미확인")
            lines.append(raw_status)
        features = []
        for key, label in (("ptp_supported", "PTP"), ("scheduled_action_supported", "예약 촬영")):
            if key in validation:
                features.append(f"{label} 기능 " + ("지원" if validation[key] else "미지원"))
        if features:
            lines.append(" · ".join(features))
        if camera.get("error"):
            lines.append(camera["error"])
        action = camera.get("diagnosis", {}).get("action")
        if action:
            lines.append("다음 조치: " + action)
        self.network_camera_details.setText("\n".join(lines))
        self.network_camera_details.show()

    def _save_network_report(self):
        if self.last_network_report is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "주소 검사 보고서 저장", "camera_network_report.json", "JSON (*.json)")
        if path:
            try:
                Path(path).write_text(json.dumps(self.last_network_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                self._log(f"주소 검사 보고서 저장: {path}")
            except OSError as error:
                self._error(str(error))

    def _aborted(self, message):
        self.command_failed = True
        if self.active_command and self.active_command.startswith("noise_"):
            self.noise_panel.status.setText("측정 중지 · 완료된 차광 자료는 유지됩니다. 미완성 자료는 분석에 사용하지 않습니다.")
        if self.active_command == "white_balance":
            self._display_changed()
            self.wb_status.setText(self.wb_status.text() + "\n측정 중지 · 기존 값 유지")
        self.state.setText(message)
        self.state.setToolTip(message)

    def _cancel(self):
        self.worker.cancel.set()
        self.stop_button.setEnabled(False)
        self.state.setText("현재 장치 작업을 마무리하고 중지합니다…")

    def _apply_preview(self):
        try:
            config = self._config()
            if config.capture_strategy != "bracket":
                plan = (self.last_reference or {}).get("capture_plan", [])
                if not plan:
                    raise CaptureError("현재 장면의 모든 기준을 먼저 측광하세요.")
                self._command("preview_setting", (step_presets(plan[self.preview_preset.currentIndex()]), config))
                return
            if config.auto_reference and self.resolved_presets is None:
                raise CaptureError("현재 장면 기준 측광을 먼저 실행하세요.")
            presets = self.resolved_presets if config.auto_reference else config.presets
            self._command("preview_setting", (presets[self.preview_preset.currentIndex()], config))
        except CaptureError as error:
            self._error(str(error))

    def _preview_toggled(self, enabled):
        if not self.loading:
            self._command("preview", enabled)

    def _progress(self, message, done, total):
        if self.active_command and self.active_command.startswith("noise_"):
            self.noise_panel.status.setText(message)
        self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(done)
        self.state.setText(message)
        self.state.setToolTip(message)
        self._log(message)

    def _log(self, message):
        self.log.appendPlainText(f"{datetime.now():%H:%M:%S}  {message}")

    def _error(self, message):
        if self.busy:
            self.command_failed = True
        self._log(f"오류 · {message}")
        self.state.setText(f"오류 · {message}")
        self.state.setToolTip(f"오류 · {message}")
        if self.active_command and self.active_command.startswith("noise_"):
            self.noise_panel.status.setText(f"확인 필요 · {message} · 완료된 차광 자료는 유지됩니다.")
        if self.active_command in ("nikon_probe", "nikon_preview"):
            self.nikon_status.setText(message)
        elif self.active_command == "verify":
            self.verify_status.setText(f"무결성 확인 필요 · {message}")
        elif self.active_command == "network":
            self.network_status.setText(f"연결 확인 필요 · {message}")
        elif self.active_command == "white_balance":
            self._display_changed()
            self.wb_status.setText(self.wb_status.text() + "\n측정 미완료 · 기존 값 유지")
        if message.startswith("미리보기 중단"):
            blocker = QSignalBlocker(self.preview_check)
            self.preview_check.setChecked(False)
            del blocker

    def _result(self, path):
        self.last_result = path
        self.verify_path.setText(path)
        manifest = json.loads((Path(path) / "manifest.json").read_text())
        frames = len(manifest["frames"])
        nef = len(manifest.get("companions", []))
        self.completed_label.setText(f"RAW {frames}장" + (f" + NEF {nef}장" if nef else "") + f" · 무결성 검사 통과\n{path}")
        fractions = []
        for record in manifest["frames"]:
            meta = json.loads((Path(path) / record["metadata"]).read_text())
            fractions.append((record["serial"], record["step"] + 1, meta["statistics"]["full_scale_fraction"]))
        self._log("RAW 상한 도달 비율 · " + " / ".join(f"{serial} 설정{step}: {fraction:.2%}" for serial, step, fraction in fractions))
        self.state.setText(f"촬영 완료 · RAW {frames}장 저장 · 무결성 검사 통과")
        self._log(f"완료 데이터셋: {path}")

    def _verified(self, report):
        self.verify_status.setText(f"검사 통과 · RAW {report['frames']}장\n{report['path']}")
        self.state.setText(f"무결성 재검사 통과 · RAW {report['frames']}장")

    def _idle(self, command=None):
        if self.active_command is not None and command is not None and command != self.active_command:
            return
        if not getattr(self, "command_failed", False):
            if self.active_command == "disconnect":
                self.state.setText("카메라 연결을 해제했습니다.")
            elif self.active_command == "preview_setting":
                self.state.setText("선택한 노출 컨트롤 설정으로 미리보기를 표시합니다.")
        self.busy, self.active_command = False, None
        self._controls()
        QTimer.singleShot(0, self.inspector_scroll.unlock_scroll)
        QTimer.singleShot(0, self.noise_panel.inspector.unlock_scroll)

    def _controls(self, *_):
        if not hasattr(self, "camera_count"):
            return
        ready = not self.busy
        self.noise_panel.set_state(ready, self.cameras, self.active_command)
        count = self.camera_count.value()
        connected = len(self.cameras)
        alternative = self.strategy.currentData() != "bracket"
        auto = self.auto.isChecked() or alternative
        self.camera_status.setText(f"{connected} / {count}대 연결")
        self.expected_cameras = count
        groups = count * (4 if self.cross_bracket.isChecked() else 1) if self.strategy.currentData() == "cross_reference" else (1 if alternative else 4)
        nef = f" + NEF {groups * self.repeats.value()}장" if self.config_widgets["dslr_enabled"].isChecked() else ""
        self.summary.setText(f"{groups}개 그룹 × {count}대 × {self.repeats.value()}회 = RAW {groups * count * self.repeats.value()}장{nef}")
        checked = sum(self.camera_list.item(index).checkState() == Qt.CheckState.Checked for index in range(self.camera_list.count()))
        self.scan_button.setEnabled(ready)
        self.connect_button.setEnabled(ready and 0 < checked <= 3)
        self.disconnect_button.setEnabled(ready and connected > 0)
        self.camera_list.setEnabled(ready)
        self.capture_button.setEnabled(ready and connected == count)
        self.capture_button.setText(f"측광 후 {groups}개 그룹 촬영  →" if auto else "4개 설정 RAW 촬영  →")
        self.capture_button.setToolTip(f"{count}대가 모두 연결되어야 촬영할 수 있습니다. 현재 {connected}대 연결됨." if connected != count
            else f"선택한 {groups}개 촬영 그룹을 순서대로 적용하고 Bayer RAW 저장·무결성 검사를 수행합니다.")
        self.capture_button.setVisible(self.pages.currentIndex() != 3)
        self.stop_button.setEnabled(self.busy and self.active_command in ("capture", "meter", "network", "nikon_probe", "nikon_preview", "white_balance", "noise_dark", "noise_bright", "noise_analyze", "noise_compare"))
        self.stop_button.setText("작업 중지" if self.active_command in ("network", "white_balance") else "촬영 중지")
        self.network_button.setEnabled(ready)
        self.nikon_probe.setEnabled(ready)
        self.nikon_preview_button.setEnabled(ready)
        self.profile_table.setEnabled(ready)
        self.network_save.setEnabled(ready and self.last_network_report is not None)
        self.network_details.setEnabled(ready and self.last_network_report is not None)
        for widget in (self.auto, self.reference, self.sync, self.high_factor, self.gain_increase,
                       self.preview_check, self.preview_preset, self.load_button, self.save_button, self.output_browse):
            widget.setEnabled(ready)
        for widget in self.config_widgets.values():
            widget.setEnabled(ready)
        self.strategy.setEnabled(ready)
        self.cross_bracket.setEnabled(ready)
        self.auto.setEnabled(ready and not alternative)
        self.reference.setEnabled(ready and not alternative)
        self.high_factor.setEnabled(ready and (not alternative or (self.strategy.currentData() == "cross_reference" and self.cross_bracket.isChecked())))
        self.gain_increase.setEnabled(self.high_factor.isEnabled())
        self.base_exp.setEnabled(ready and not auto)
        self.base_gain.setEnabled(ready and not auto)
        self.meter_button.setEnabled(ready and auto and connected > 0)
        wb_available = ready and connected > 0 and (not self.wb_reference.currentData()
            or self.wb_reference.currentData() in {camera.serial for camera in self.cameras})
        self.wb_button.setEnabled(wb_available)
        self.wb_settings_button.setEnabled(wb_available)
        self.apply_preview.setEnabled(ready and connected > 0 and (not auto or self.resolved_presets is not None))
        for name, exposure, gain in self.preset_editors:
            name.setEnabled(ready)
            exposure.setEnabled(ready and not auto)
            gain.setEnabled(ready and not auto)
        self.verify_button.setEnabled(ready and bool(self.verify_path.text().strip()))
        self.verify_path.setEnabled(ready)
        self.verify_browse.setEnabled(ready)
        for button in self.findChildren(QPushButton):
            button.setCursor(Qt.CursorShape.PointingHandCursor if button.isEnabled() else Qt.CursorShape.ArrowCursor)

    def _select_page(self, index):
        self.pages.setCurrentIndex(index)
        titles = [("동기 촬영", "실시간 확인부터 RAW 저장까지"),
                  ("촬영 설정", "노출 컨트롤 · 동기 촬영 · 저장 · 미리보기"),
                  ("검사·기록", "완료 데이터셋과 무결성 검사 결과"),
                  ("노이즈 분석", "차광·밝은 RAW 반복 촬영 · 픽셀 맵 · PTC·SNR")]
        self.page_title.setText(titles[index][0])
        self.subtitle.setText(titles[index][1])
        for button_index, button in enumerate(self.nav_buttons):
            button.setChecked(button_index == index)
        self._controls()
        QTimer.singleShot(0, self._reflow)

    def _zoom(self, index):
        if self.cards[index].saved_image is None:
            return
        self.zoomed_index = index
        self.saved_inspector_scroll = self.inspector_scroll.verticalScrollBar().value()
        self.focus_image.set_image(self.cards[index].saved_image)
        self.focus_title.setText(f"CAMERA {index+1:02d}  ·  {self.cameras[index].serial}")
        self.focus_detail.setText(self.cards[index].detail.text() + " · 다시 클릭하면 전체 보기")
        self.camera_deck.setCurrentIndex(1)
        self.inspector_scroll.hide()
        self.summary.hide()
        self.zoom_hint.setText("선택한 카메라를 확대했습니다. 이미지를 다시 클릭하거나 Esc를 눌러 돌아갑니다.")
        self._reflow()

    def _unzoom(self):
        if self.zoomed_index is None:
            return
        self.zoomed_index = None
        self.camera_deck.setCurrentIndex(0)
        self.summary.show()
        self.zoom_hint.setText("이미지를 클릭하면 크게 볼 수 있습니다. 다시 클릭하면 전체 보기로 돌아옵니다.")
        self._reflow()
        QTimer.singleShot(0, lambda: self.inspector_scroll.verticalScrollBar().setValue(self.saved_inspector_scroll))

    def _toggle_inspector(self):
        self.inspector_overlay_open = not self.inspector_overlay_open
        self._reflow()

    def toggle_fullscreen(self):
        if self.isFullScreen():
            self.showMaximized() if self.was_maximized else self.showNormal()
            self.fullscreen_button.setText("□  전체화면")
        else:
            self.was_maximized = self.isMaximized()
            self.showFullScreen()
            self.fullscreen_button.setText("↙  전체화면 종료")

    def _escape(self):
        if self.pages.currentIndex() == 3 and self.noise_panel.graph_zoomed:
            self.noise_panel._zoom()
        elif self.pages.currentIndex() == 3 and self.noise_panel.camera_zoom_index is not None:
            self.noise_panel._camera_unzoom()
        elif self.pages.currentIndex() == 3 and self.noise_panel.overlay_open:
            self.noise_panel._toggle()
        elif self.zoomed_index is not None:
            self._unzoom()
        elif self.inspector_overlay_open and self.inspector_scroll.isVisible():
            self._toggle_inspector()
        elif self.isFullScreen():
            self.toggle_fullscreen()

    @staticmethod
    def _regrid(grid, widgets, columns):
        while grid.count():
            grid.takeAt(0)
        for column in range(4):
            grid.setColumnStretch(column, 0)
        for index, widget in enumerate(widgets):
            grid.addWidget(widget, index // columns, index % columns)
        for column in range(columns):
            grid.setColumnStretch(column, 1)

    def _remember_history_split(self, *_):
        splitter = self.history_splitter
        self.history_split_sizes[splitter.orientation()] = splitter.sizes()

    def _restore_history_split(self):
        splitter = self.history_splitter
        splitter.setSizes(self.history_split_sizes[splitter.orientation()])

    def _reflow(self):
        if not hasattr(self, "settings_columns"):
            return
        compact = self.width() < 1050
        self.sidebar.setFixedWidth(72 if compact else 172)
        self.logo.setText("R\nR" if compact else "RAW\nRIG")
        self.sidebar_subtitle.setVisible(not compact)
        self.sidebar_note.setVisible(not compact)
        for button, (icon, text) in zip(self.nav_buttons, self.nav_labels):
            button.setText(icon if compact else f"{icon}   {text}")
            button.setToolTip(text)
        self.badge.setVisible(self.width() >= 1250)
        self.subtitle.setVisible(self.width() >= 900)
        self.load_button.setText("불러오기" if compact else "설정 불러오기")
        self.save_button.setText("저장" if compact else "설정 저장")
        overlay = self.width() < 1320
        margins = 16 if compact else 24
        self.workspace_layout.setContentsMargins(margins, 16, margins, 16)
        self.stage_layout.setSpacing(10 if compact else 16)
        self.workspace_layout.setDirection(QBoxLayout.Direction.LeftToRight)
        self.inspector.setMaximumWidth(350)
        self.inspector_scroll.setMinimumWidth(300)
        self.inspector_scroll.setMaximumWidth(350)
        self.inspector_scroll.setMinimumHeight(0)
        self.inspector_close.setVisible(overlay)
        self.inspector_toggle.setVisible(overlay and self.zoomed_index is None)
        self.zoom_hint.setVisible(not compact or self.zoomed_index is not None)
        if overlay:
            if self.workspace_layout.indexOf(self.inspector_scroll) >= 0:
                self.workspace_layout.removeWidget(self.inspector_scroll)
            width = min(350, self.workspace.width() - margins * 2)
            self.inspector_scroll.setGeometry(self.workspace.width() - margins - width, 16,
                                              width, max(0, self.workspace.height() - 32))
            self.inspector_scroll.setVisible(self.inspector_overlay_open and self.zoomed_index is None)
            self.inspector_scroll.raise_()
        else:
            if self.workspace_layout.indexOf(self.inspector_scroll) < 0:
                self.workspace_layout.addWidget(self.inspector_scroll)
            self.inspector_scroll.setVisible(self.zoomed_index is None)
        self.inspector_toggle.setText("설정 닫기  ×" if self.inspector_overlay_open else "설정 패널  ☰")
        self._regrid(self.sequence_grid, [tile for tile, _ in self.sequence_tiles], 2)
        available = self.content.width() - 48
        preset_columns = 4 if available >= 1240 else 2 if available >= 680 else 1
        self._regrid(self.preset_grid, self.preset_tiles, preset_columns)
        orientation = Qt.Orientation.Horizontal if available >= 1120 else Qt.Orientation.Vertical
        splitter = self.history_splitter
        horizontal = orientation == Qt.Orientation.Horizontal
        self.history_results.setMinimumWidth(520 if horizontal else 0)
        self.history_log_panel.setMinimumWidth(280 if horizontal else 0)
        if orientation != splitter.orientation():
            splitter.setOrientation(orientation)
        self._restore_history_split()
        # Side-bar and orientation changes settle their geometry at the next event turn.
        QTimer.singleShot(0, self._restore_history_split)
        splitter.handle(1).setToolTip("드래그하여 검사 결과와 진행 기록의 공간을 조절합니다.")
        self.history_page.layout().setContentsMargins(margins, 16, margins, 16)
        self._fit_network_table()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        QTimer.singleShot(0, self._reflow)

    def closeEvent(self, event):
        if self.worker.isRunning():
            self.worker.request_shutdown()
            self.state.setText("카메라 연결을 정리하고 종료합니다…")
            event.ignore()
            QTimer.singleShot(100, self.close)
        else:
            event.accept()


STYLE = """
QWidget { font-family: 'Noto Sans CJK KR', 'Noto Sans', sans-serif; font-size: 12px; color: #26334b; }
QMainWindow, QScrollArea, QStackedWidget, QWidget#pageBody { background: #f4f5f9; }
QWidget#cameraInspector { background: #f4f5f9; }
QScrollArea#inspectorPanel { background: #f4f5f9; border: 1px solid #e4e7ef; border-radius: 10px; }
QFrame#sidebar { background: #393e70; }
QLabel#logo { color: white; font-size: 30px; font-weight: 800; letter-spacing: 2px; }
QLabel#navMuted { color: #b5bad8; font-size: 10px; }
QPushButton#navButton { background: transparent; color: #d8dcf2; text-align: left; border: none; padding: 13px 12px; font-weight: 600; }
QPushButton#navButton:hover { background: #454c83; }
QPushButton#navButton:checked { background: #626baf; color: white; }
QPushButton#navButton:checked:hover { background: #7880bd; color: white; }
QPushButton#navButton:pressed { background: #303664; }
QFrame#header, QFrame#footer { background: white; border: 0; }
QFrame#header { border-bottom: 1px solid #e5e7ef; }
QSplitter::handle { background: #f4f5f9; }
QSplitter::handle:hover { background: #ddd6ef; }
QFrame#footer { border-top: 1px solid #e5e7ef; }
QLabel#pageTitle { font-size: 23px; font-weight: 700; color: #1f2b41; }
QLabel#sectionTitle { font-size: 15px; font-weight: 700; }
QLabel#mutedText { color: #7b8598; font-size: 11px; }
QLabel#accentText { color: #6762ae; font-weight: 600; font-size: 11px; }
QLabel#modeBadge { background: #efedfb; color: #7565b1; border-radius: 7px; padding: 8px 12px; }
QFrame#cameraCard, QFrame#focusCard, QFrame#whitePanel, QGroupBox { background: white; border: 1px solid #e4e7ef; border-radius: 12px; }
QFrame#cameraCard:hover { border: 1px solid #a9a5d0; }
QGroupBox { font-weight: 700; margin-top: 12px; padding: 24px 14px 14px 14px; }
QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; left: 18px; top: 0; background: white; padding: 0 5px; }
QLabel#cardTitle { color: #384259; font-size: 11px; font-weight: 700; letter-spacing: 1px; }
QLabel#cameraDetail { font-size: 11px; color: #546077; }
QLabel#liveBadge { background: #e9f7f0; color: #368166; border-radius: 5px; padding: 3px 7px; font-size: 10px; }
QLabel#mutedBadge { background: #f1f3f7; color: #8b93a4; border-radius: 5px; padding: 3px 7px; font-size: 10px; }
QFrame#sequenceTile { background: #f7f8fc; border: 1px solid #e9ebf3; border-radius: 8px; }
QFrame#baselineTile { background: #f0edfb; border: 1px solid #cec8ef; border-radius: 8px; }
QLabel#tileTitle { font-weight: 700; font-size: 11px; }
QLabel#tileValues { font-size: 11px; color: #5c6880; }
QLabel#summaryText { color: #7a8195; font-weight: 500; }
QPushButton { background: white; border: 1px solid #dce0eb; border-radius: 7px; padding: 8px 12px; font-weight: 500; }
QPushButton:hover { color: #574985; background: #efebfb; border-color: #9d8acc; }
QPushButton:pressed { background: #eae6f8; }
QPushButton:disabled { color: #a5adbb; background: #f5f6f9; border-color: #e7e9ef; }
QPushButton#captureButton { color: white; background: #7265b5; border: none; padding: 12px 20px; font-weight: 700; }
QPushButton#captureButton:hover { color: white; background: #6255a5; }
QPushButton#captureButton:pressed { background: #51458e; }
QPushButton#captureButton:disabled { color: #f3f1fa; background: #bbb5d6; }
QPushButton#networkButton { color: #6855a4; background: #f4f0fe; border-color: #c7bee5; }
QPushButton#networkButton:hover { color: #574985; background: #e8e0fb; border-color: #9d8acc; }
QPushButton#networkButton:pressed { background: #dcd2f5; }
QPushButton#networkButton:disabled { color: #a5adbb; background: #f5f6f9; border-color: #e7e9ef; }
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox { background: #fafbfe; border: 1px solid #dfe3ed; border-radius: 6px; padding: 6px 8px; min-height: 22px; selection-background-color: #b6a9e0; }
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus { border-color: #9887c9; }
QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled, QComboBox:disabled { color: #8f98aa; background: #f4f5f9; }
QSpinBox, QDoubleSpinBox { padding-right: 21px; }
QSpinBox::up-button, QDoubleSpinBox::up-button { subcontrol-origin: border; subcontrol-position: top right; width: 18px; border-left: 1px solid #e3e7f1; border-bottom: 1px solid #e3e7f1; border-top-right-radius: 6px; background: #f2f4fa; }
QSpinBox::down-button, QDoubleSpinBox::down-button { subcontrol-origin: border; subcontrol-position: bottom right; width: 18px; border-left: 1px solid #e3e7f1; border-bottom-right-radius: 6px; background: #f2f4fa; }
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow { image: url('__UP_ICON__'); width: 10px; height: 6px; }
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow, QComboBox::down-arrow { image: url('__DOWN_ICON__'); width: 10px; height: 6px; }
QComboBox::drop-down { subcontrol-origin: border; subcontrol-position: top right; width: 23px; border-left: 1px solid #e3e7f1; border-top-right-radius: 6px; border-bottom-right-radius: 6px; }
QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover,
QComboBox::drop-down:hover { background: #e8e2f8; }
QCheckBox { spacing: 8px; }
QCheckBox::indicator { width: 15px; height: 15px; border: 1px solid #c7cbda; border-radius: 4px; background: white; }
QCheckBox::indicator:checked { background: #8274be; border-color: #8274be; image: url('__CHECK_ICON__'); }
QCheckBox::indicator:hover { border-color: #8872bd; background: #f0eafa; }
QCheckBox::indicator:checked:hover { background: #6854a8; border-color: #6854a8; }
QCheckBox::indicator:disabled { background: #e5e7ee; }
QListWidget { border: 1px solid #e5e7ef; border-radius: 7px; background: #fafbfe; padding: 4px; font-size: 11px; }
QListWidget::item { padding: 5px 0; }
QListWidget::item:selected { color: #464574; background: #eae7f7; border-radius: 4px; }
QPlainTextEdit { background: white; border: 1px solid #e4e7ef; border-radius: 9px; padding: 12px; font-size: 11px; }
QTableWidget { background: white; alternate-background-color: #f8f9fd; border: 1px solid #e4e7ef; border-radius: 7px; gridline-color: #edf0f6; font-size: 11px; selection-background-color: #ece8f8; selection-color: #494368; }
QHeaderView::section { background: #f4f5fa; color: #65708a; border: 0; border-bottom: 1px solid #e4e7ef; padding: 9px 12px; font-size: 11px; font-weight: 600; }
QProgressBar { background: #efedf5; border: none; border-radius: 2px; }
QProgressBar::chunk { background: #8c7cc6; border-radius: 2px; }
QScrollBar:vertical { background: transparent; width: 9px; margin: 4px 0; }
QScrollBar::handle:vertical { background: #d1d4e1; border-radius: 4px; min-height: 28px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: transparent; }
QToolTip { background: #333957; color: white; border: none; border-radius: 5px; padding: 7px 10px; }
"""
for token, filename in (("__UP_ICON__", "arrow-up.svg"), ("__DOWN_ICON__", "arrow-down.svg"), ("__CHECK_ICON__", "check.svg")):
    STYLE = STYLE.replace(token, (Path(__file__).parent / "assets" / filename).as_posix())


def launch_gui(demo=False, config=None, expected_cameras=3):
    # OpenCV's bundled Qt path is incompatible with PySide6's platform plugin.
    for key in ("QT_QPA_PLATFORM_PLUGIN_PATH", "QT_QPA_FONTDIR"):
        if "cv2" in os.environ.get(key, ""):
            os.environ.pop(key, None)
    app = QApplication.instance() or QApplication(["raw-rig"])
    app.setApplicationName("raw-rig")
    app.setApplicationDisplayName("RAW Rig · RAW 카메라 촬영")
    app.setDesktopFileName("raw-rig")
    app.setWindowIcon(QIcon(str(APP_ICON)))
    app.setFont(QFont("Noto Sans CJK KR", 10))
    from .instance import GuiInstance
    instance = GuiInstance(demo)
    windows = []
    def activate():
        if windows:
            if windows[0].isMinimized():
                windows[0].showNormal()
            windows[0].raise_()
            windows[0].activateWindow()
    if not instance.acquire(activate):
        return 0
    window = MainWindow(demo, config, expected_cameras)
    windows.append(window)
    window.show()
    timer = QTimer()
    timer.timeout.connect(lambda: None)
    timer.start(200)
    old_handlers = {sig: signal.signal(sig, lambda *_: window.close()) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        return app.exec()
    finally:
        instance.close()
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
