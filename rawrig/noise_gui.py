"""Fixed noise-analysis workspace with an independently scrolling inspector."""

from dataclasses import asdict, replace
import json
from pathlib import Path

from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QImage, QDesktopServices
from PySide6.QtCore import QUrl
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QFrame,
    QLabel,
    QFormLayout,
    QGroupBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QAbstractItemView,
    QFileDialog,
    QStackedWidget,
    QSizePolicy,
)

from .models import CaptureError
from .noise import NoiseConfig, roi_bounds
from .flicker import measured_period
from .widgets import (
    PreviewCanvas,
    ScrollSafeComboBox,
    StableScrollArea,
    CameraCard,
    CameraGrid,
)


class NoisePanel(QWidget):
    capture_requested = Signal(str, object, str)
    analysis_requested = Signal(str)
    comparison_requested = Signal(object)
    stop_requested = Signal()
    notify = Signal(str)

    def __init__(self):
        super().__init__()
        self.session_path = ""
        self.report = None
        self.plot_base = ""
        self.session_config = None
        self.dark_complete = False
        self.bright_complete = False
        self.retaking_bright = False
        self.ready = True
        self.cameras = []
        self.overlay_open = False
        self.graph_zoomed = False
        self.camera_zoom_index = None
        self.showing_graph = False
        self.body = QHBoxLayout(self)
        self.body.setContentsMargins(24, 16, 24, 16)
        self.body.setSpacing(20)
        self.viewer = QWidget()
        view = QVBoxLayout(self.viewer)
        view.setContentsMargins(0, 0, 0, 0)
        top = QHBoxLayout()
        title = QLabel("노이즈 분석 · RAW / PTC")
        title.setObjectName("sectionTitle")
        top.addWidget(title, 1)
        self.view_button = QPushButton("그래프 보기")
        self.view_button.setToolTip(
            "실시간 카메라 화면과 저장된 분석 그래프를 전환합니다."
        )
        self.view_button.hide()
        top.addWidget(self.view_button)
        self.toggle = QPushButton("측정 설정  ☰")
        top.addWidget(self.toggle)
        view.addLayout(top)
        self.status = QLabel("① 차광 촬영 → ② 밝은 촬영 · 반복 수·분석 조건 고정")
        self.status.setWordWrap(True)
        self.status.setObjectName("accentText")
        view.addWidget(self.status)
        self.summary = QTableWidget(0, 7)
        self.summary.setHorizontalHeaderLabels(
            [
                "카메라 / 게인 / 채널",
                "읽기 RN (DN)",
                "RN (e−)",
                "게인 DN/e−",
                "유효 포화 e−",
                "DR dB",
                "검증",
            ]
        )
        self.summary.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.summary.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.summary.setMinimumHeight(0)
        self.summary.setMaximumHeight(164)
        self.summary.hide()
        view.addWidget(self.summary)
        self.plot_choice = ScrollSafeComboBox()
        self.plot_choice.setToolTip(
            "보고 싶은 그래프를 선택합니다. 휠로 선택값이 바뀌지 않습니다."
        )
        view.addWidget(self.plot_choice)
        self.plot_choice.hide()
        self.display_deck = QStackedWidget()
        self.display_deck.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored
        )
        self.display_deck.setMinimumSize(40, 48)
        self.camera_cards = [CameraCard(i) for i in range(3)]
        self.live_grid = CameraGrid(self.camera_cards)
        self.live_grid.set_count(0)
        self.display_deck.addWidget(self.live_grid)
        self.live_focus = PreviewCanvas()
        self.live_focus.setMinimumSize(40, 48)
        self.live_focus.setToolTip("다시 클릭하면 전체 카메라 보기로 돌아갑니다.")
        self.display_deck.addWidget(self.live_focus)
        self.canvas = PreviewCanvas()
        self.canvas.setMinimumSize(40, 48)
        self.canvas.placeholder = "차광 후 노이즈 분포 · 밝은 촬영 후 밝기별 곡선"
        self.canvas.setToolTip(
            "그래프 클릭: 크게 보기 · 다시 클릭: 이전 화면 · 원본 PNG/SVG는 결과 폴더에 저장됩니다"
        )
        self.display_deck.addWidget(self.canvas)
        view.addWidget(self.display_deck, 1)
        self.quality = QLabel(
            "균일하고 안정된 확산광을 사용하세요. 움직이는 장면·모니터 PWM·포화는 측정 정확도에 영향을 줍니다."
        )
        self.quality.setObjectName("mutedText")
        self.quality.setWordWrap(True)
        view.addWidget(self.quality)
        self.body.addWidget(self.viewer, 1)
        self.controls = QWidget()
        self.controls.setObjectName("cameraInspector")
        controls = QVBoxLayout(self.controls)
        controls.setContentsMargins(0, 0, 0, 0)
        close_row = QHBoxLayout()
        close_row.addWidget(QLabel("측정 · 촬영 설정"), 1)
        self.close_button = QPushButton("닫기  ×")
        close_row.addWidget(self.close_button)
        controls.addLayout(close_row)
        workflow = QGroupBox("두 번 누르면 촬영부터 분석까지")
        wf = QVBoxLayout(workflow)
        self.instructions = QLabel(
            "1. 렌즈 캡으로 빛을 차단하고 ①.\n2. 캡을 제거하고 균일한 밝은 면을 비춘 뒤 ②.\n조명 이상은 자동 제외·보충합니다. 화면이 새하얗게 포화되면 조명을 약하게 하세요."
        )
        self.instructions.setWordWrap(True)
        wf.addWidget(self.instructions)
        self.protocol_choice = ScrollSafeComboBox()
        self.protocol_choice.addItem(
            "밝기별 곡선 · 자동 노출시간 순회", "automatic_ptc"
        )
        self.protocol_choice.addItem("픽셀 분포 · 노출시간·게인 고정", "two_condition")
        self.protocol_choice.addItem(
            "정밀 곡선 · 노출시간 32단계 · 단계별 1000장", "exposure_sweep"
        )
        self.protocol_choice.setToolTip(
            "측정 목적만 선택합니다. 반복 수·ROI·분석 조건은 고정이며 휠로 선택이 바뀌지 않습니다."
        )
        wf.addWidget(self.protocol_choice)
        self.dark_button = QPushButton("① 차광 RAW 연속 촬영")
        self.bright_button = QPushButton("② 밝은 RAW 촬영 · 자동 분석")
        self.stop_button = QPushButton("작업 중지")
        action_bar = QHBoxLayout()
        for button in (self.dark_button, self.bright_button, self.stop_button):
            action_bar.addWidget(button)
        view.insertLayout(2, action_bar)
        self.estimate = QLabel()
        self.estimate.setWordWrap(True)
        self.estimate.setObjectName("mutedText")
        wf.addWidget(self.estimate)
        controls.addWidget(workflow)
        group = QGroupBox("고정 실험 조건")
        form = QFormLayout(group)
        self.condition_labels = {}
        for label, value in (
            ("차광 촬영", "1000장"),
            ("밝은 촬영", "1000장"),
            ("분석 영역", "256 × 256 · 원래 Bayer 값"),
            ("노출시간·게인", "시작값을 읽어 두 조건에 고정"),
            ("자동 조절·ISP", "OFF · RAW 원본 보존"),
            ("밝기 단계", "차광 / 밝은 장면 두 조건"),
        ):
            text = QLabel(value)
            text.setWordWrap(True)
            self.condition_labels[label] = text
            form.addRow(label, text)
        self.operating_label = QLabel(
            "차광 시작 시 카메라의 실제 노출시간·게인을 기록합니다."
        )
        self.operating_label.setWordWrap(True)
        self.operating_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        form.addRow(self.operating_label)
        controls.addWidget(group)
        self.inputs = [self.protocol_choice]
        self.limitation = QLabel(
            "두 조건으로 잔차·읽기 노이즈·픽셀 분포·명암 노이즈를 측정합니다. 여러 밝기에서 측정하는 PTC 곡선·정밀 게인 맵·전자 단위는 미측정으로 표시합니다."
        )
        self.limitation.setWordWrap(True)
        self.limitation.setObjectName("mutedText")
        controls.addWidget(self.limitation)
        results = QGroupBox("세션 · 결과")
        rl = QVBoxLayout(results)
        self.path_label = QLabel("선택된 노이즈 세션 없음")
        self.path_label.setWordWrap(True)
        self.path_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        rl.addWidget(self.path_label)
        self.load_button = QPushButton("이전 측정 열기 · 이어서 촬영")
        self.analyze_button = QPushButton("저장된 RAW 재검사 · 분석")
        self.compare_button = QPushButton("다른 측정과 비교")
        self.open_button = QPushButton("결과 폴더 열기 ↗")
        for button in (
            self.load_button,
            self.analyze_button,
            self.compare_button,
            self.open_button,
        ):
            rl.addWidget(button)
        source = QLabel(
            '<a href="https://doi.org/10.1142/S1793545816300081">Li et al. (2016) · PTC 논문</a>'
        )
        source.setOpenExternalLinks(True)
        rl.addWidget(source)
        controls.addWidget(results)
        controls.addStretch()
        self.inspector = StableScrollArea()
        self.inspector.setWidgetResizable(True)
        self.inspector.setFrameShape(QFrame.Shape.NoFrame)
        self.inspector.setObjectName("inspectorPanel")
        self.inspector.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.inspector.setWidget(self.controls)
        self.inspector.setFixedWidth(350)
        self.body.addWidget(self.inspector)
        self.dark_button.clicked.connect(lambda: self._capture("dark"))
        self.bright_button.clicked.connect(lambda: self._capture("bright"))
        self.stop_button.clicked.connect(self.stop_requested)
        self.load_button.clicked.connect(self._load_dialog)
        self.analyze_button.clicked.connect(
            lambda: self.analysis_requested.emit(self.session_path)
        )
        self.compare_button.clicked.connect(self._compare_dialog)
        self.open_button.clicked.connect(
            lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(self.session_path))
        )
        self.plot_choice.currentIndexChanged.connect(self._plot_changed)
        self.canvas.clicked.connect(self._zoom)
        self.live_focus.clicked.connect(self._camera_unzoom)
        for card in self.camera_cards:
            card.activated.connect(self._camera_zoom)
        self.view_button.clicked.connect(self._switch_view)
        self.toggle.clicked.connect(self._toggle)
        self.close_button.clicked.connect(self._toggle)
        self.status.setFixedHeight(38)
        self.protocol_choice.currentIndexChanged.connect(self._protocol_changed)
        self._update()

    def _protocol_changed(self):
        self._update()
        if self.session_config and self.ready:
            if self.session_config != asdict(self.settings()):
                self.status.setText(
                    "이전 결과 보존 · 선택한 모드의 ① 차광 촬영을 먼저 시작하세요."
                )
            elif self.dark_complete and not self.bright_complete:
                self.status.setText(
                    "이 모드의 차광 검증 완료 · 밝은 면을 준비하고 ②를 누르세요."
                )

    def settings(self):
        if self.protocol_choice.currentData() == "exposure_sweep":
            loaded = getattr(self, "_loaded_noise_config", None)
            if loaded is not None and loaded.protocol == "exposure_sweep":
                return loaded
            return NoiseConfig.precision()
        return (
            NoiseConfig.fixed()
            if self.protocol_choice.currentData() == "two_condition"
            else replace(
                NoiseConfig.fixed_curve(),
                anti_flicker_period_us=measured_period(self.cameras),
            )
        )

    def load_config(self, data):
        if data:
            cfg = NoiseConfig.from_dict(data)
            self._loaded_noise_config = cfg
            if cfg.protocol == "exposure_sweep":
                index = self.protocol_choice.findData(cfg.protocol)
                self.protocol_choice.setItemText(index, f"노출시간 {cfg.levels}단계 · 밝은 단계별 기본 {cfg.frames}장")
            self.protocol_choice.setCurrentIndex(
                max(0, self.protocol_choice.findData(cfg.protocol))
            )
        self._update()

    def set_state(self, ready, cameras, active=None):
        if [c.serial for c in cameras] != [c.serial for c in self.cameras]:
            self.camera_zoom_index = None
            for card in self.camera_cards:
                card.clear()
            self.live_grid.set_count(len(cameras))
            self._show_view()
        self.ready, self.cameras = ready, cameras
        self.stop_button.setEnabled(
            not ready
            and active
            in ("noise_dark", "noise_bright", "noise_analyze", "noise_compare")
        )
        for widget in self.inputs:
            widget.setEnabled(ready)
        self.load_button.setEnabled(ready)
        self.analyze_button.setEnabled(ready and self.dark_complete)
        self.compare_button.setEnabled(
            ready and self.report is not None and self.bright_complete
        )
        self.open_button.setEnabled(bool(self.session_path))
        self._update()

    def _update(self, *_):
        cfg = self.settings()
        fixed = cfg.protocol == "two_condition"
        levels = len(cfg.plan())
        timing = (
            f"플리커 회피 · {cfg.anti_flicker_period_us / 1000:.3f} ms 정수배 · {levels}단계"
            if cfg.anti_flicker_period_us
            else f"{cfg.exposure_min_us:g}~{cfg.exposure_max_us:g} µs 로그 {levels}단계"
        )
        saved_plan = getattr(self, "_loaded_acquisition_plan", [])
        loaded = getattr(self, "_loaded_noise_config", None)
        if saved_plan and loaded is not None and loaded.protocol == cfg.protocol:
            actual_times = [
                preset["exposure_us"]
                for point in saved_plan
                for preset in point.get("camera_presets", {}).values()
            ]
            if actual_times:
                timing = f"저장된 실제 계획 · {len(saved_plan)}단계 · {min(actual_times):g}~{max(actual_times):g} µs"
        self.instructions.setText(
            "1. 렌즈 캡 등으로 빛을 완전히 차단하고 ①을 누르세요.\n2. 캡을 제거하고 균일한 밝은 광원을 비춘 뒤 ②를 누르세요.\n"
            + (
                "카메라 시작값을 읽어 노출시간·게인을 그대로 고정합니다."
                if fixed
                else f"광원 밝기는 그대로 두세요. 두 촬영에서 같은 {levels}개 노출시간을 자동 순회합니다."
            )
        )
        from collections import Counter
        def count_summary(phase):
            counts=Counter(cfg.stack_count(phase,p) for p in cfg.plan(phase))
            return ' + '.join(f'{n}단계×{count:,}장' for count,n in sorted(counts.items(),reverse=True))
        for label, value in {
            "차광 촬영": "1000장"
            if fixed
            else f"{cfg.count('dark'):,}장 · {count_summary('dark')}",
            "밝은 촬영": "1000장"
            if fixed
            else f"{cfg.count('bright'):,}장 · {count_summary('bright')}",
            "노출시간·게인": "카메라 시작값을 두 조건에 고정"
            if fixed
            else timing + " · 최저 공통 게인 고정 · 지원 범위 자동 반영",
            "밝기 단계": "차광 / 밝은 장면 두 조건"
            if fixed
            else "광원 고정 · 노출시간만 자동 순회",
        }.items():
            self.condition_labels[label].setText(value)
        self.limitation.setText(
            "두 조건으로 분포를 측정합니다. 밝기별 PTC·게인·전자 단위는 미측정입니다."
            if fixed
            else "PTC는 논문 §2.3의 단계별 100장, 최단 차광 맵은 §3.1의 1000장입니다. 광량 대신 노출시간을 바꾸고 단계별 차광으로 보정합니다. 픽셀 게인 맵은 그림 6의 단계별 1000장보다 정밀도가 낮습니다."
        )
        if cfg.protocol == "exposure_sweep":
            self.limitation.setText(
                f"광원 고정 · 단계별 차광 {cfg.dark_frames or cfg.frames} + 밝은 RAW {cfg.frames}장. 짧은 노출시간은 조명 위상을 검사해 안정된 프레임만 모읍니다. 광량 조절 논문 실험과 방식은 다릅니다. 블랙 레벨 1% · 자동 클램핑 OFF · 보정·평균화 없이 RAW 저장."
            )
        self.estimate.setText(
            f"차광 {cfg.count('dark')}회 + 밝은 장면 {cfg.count('bright')}회 × 연결 {len(self.cameras)}대\n카메라당 ROI RAW 약 {(cfg.count('dark') + cfg.count('bright')) * 256 * 256 * 2 / 1024**3:.2f} GiB · 실험 조건 고정"
        )
        self.dark_button.setEnabled(self.ready and bool(self.cameras))
        self.dark_button.setToolTip(
            "미완료 차광 세션이면 완료된 지점의 RAW·설정·통계를 검사해 재사용하고 남은 단계만 촬영합니다."
        )
        same = self.session_config == asdict(cfg)
        self.bright_button.setEnabled(
            self.ready and bool(self.cameras) and same and self.dark_complete
        )
        self.bright_button.setToolTip(
            (
                "차광에서 기록한 시작값 그대로 밝은 RAW 1000장을 촬영합니다."
                if fixed
                else f"차광과 같은 {levels}개 노출시간을 자동 순회하고 밝기별 곡선을 계산합니다."
            )
            + " 조명 사전 검사 후 어둡거나 밝기가 바뀐 프레임을 자동 제외하고 정상 장수를 보충합니다. 완료 세션이면 원본을 보존하고 차광만 검증·복사해 밝은 촬영을 다시 합니다."
            if same
            else "선택한 측정 방식의 차광을 먼저 촬영하세요. 다른 조건의 차광과 결합할 수 없습니다."
        )
        self.reflow()
        QTimer.singleShot(0, self._fit_inspector_text)

    def _fit_inspector_text(self):
        # Recalculate wrapped text after protocol changes at the actual sidebar width.
        # The sidebar scrolls; labels must never clip to a previous mode's height.
        for label in (
            self.instructions,
            self.estimate,
            self.operating_label,
            self.limitation,
            self.path_label,
            *self.condition_labels.values(),
        ):
            height = label.heightForWidth(label.width())
            if height > 0:
                label.setMinimumHeight(height)

    def _capture(self, phase):
        try:
            self.showing_graph = False
            self.graph_zoomed = False
            self.camera_zoom_index = None
            self._show_view()
            cfg = self.settings()
            self.capture_requested.emit(phase, cfg, self.session_path)
        except CaptureError as error:
            self.notify.emit(str(error))

    def handle_result(self, result):
        if result["phase"] != "comparison":
            self.session_path = result["path"]
        self.path_label.setText(self.session_path)
        if result["phase"] == "dark":
            self.dark_complete, self.bright_complete = True, False
            self.report = None
            self.showing_graph = False
            self.view_button.hide()
            self._show_view()
            self.plot_choice.clear()
            self.canvas.set_image(None)
            self.summary.hide()
            session = json.loads((Path(self.session_path) / "session.json").read_text())
            self.retaking_bright = bool(session.get("bright_retake_from"))
            self.session_config = asdict(NoiseConfig.from_dict(session["noise_config"]))
            self._show_operating_settings(session)
            self.status.setText(
                f"차광 RAW {result['frames']:,}장 검사 완료. 캡을 제거하고 균일 확산광을 비춘 뒤 ②를 누르세요."
            )
        elif result["phase"] == "bright":
            self.bright_complete = True
            self.status.setText("밝은 RAW 저장 완료 · 노이즈 계산 및 그래프 생성 중…")
        elif result["phase"] == "incomplete":
            session = json.loads((Path(self.session_path) / "session.json").read_text())
            self.retaking_bright = bool(session.get("bright_retake_from"))
            self.session_config = asdict(NoiseConfig.from_dict(session["noise_config"]))
            self.dark_complete = (
                Path(self.session_path) / "DARK_COMPLETE.json"
            ).is_file()
            self.bright_complete = (
                Path(self.session_path) / "BRIGHT_COMPLETE.json"
            ).is_file()
            self.status.setText(
                "미완료 자료 보존 · 완료된 차광은 다시 사용할 수 있습니다."
            )
        elif "report" in result:
            if result["phase"] == "analysis":
                self.dark_complete = self.bright_complete = True
            self.show_report(
                result["report"],
                comparison=result["phase"] == "comparison",
                base=result["path"],
                dark_only=result["phase"] == "dark_analysis",
            )
        self._update()

    def _show_operating_settings(self, session):
        self._loaded_acquisition_plan = session.get("acquisition_plan", [])
        plan = session.get("acquisition_plan")
        text = "\n".join(
            f"{serial}: {v['exposure_us']:g} µs · {v['gain_db']:g} dB"
            for serial, v in session.get("operating_settings", {}).items()
        )
        if plan:
            text = f"적용 계획: {plan[0]['exposure_us']:g}~{plan[-1]['exposure_us']:g} µs · {len(plan)}단계 · {plan[0]['gain_db']:g} dB\n센서 반올림 후 실제 값과 chunk는 각 RAW 메타데이터에 기록됩니다."
            if "camera_presets" in plan[0]:
                text = "\n".join(
                    f"실제 적용 {serial}: {plan[0]['camera_presets'][serial]['exposure_us']:g}~{plan[-1]['camera_presets'][serial]['exposure_us']:g} µs · {plan[0]['camera_presets'][serial]['gain_db']:g} dB"
                    for serial in plan[0]["camera_presets"]
                )
        self.operating_label.setText(
            text or "카메라별 실제 적용값은 촬영 메타데이터에 기록됩니다."
        )

    def show_report(self, report, comparison=False, base=None, dark_only=False):
        self.report = report
        self.showing_graph = True
        self.camera_zoom_index = None
        self.view_button.show()
        self.plot_base = str(
            Path(base)
            if comparison
            else Path(self.session_path)
            / ("dark_analysis" if dark_only else "analysis")
        )
        rows = [
            (c, ch, data)
            for c in report["cameras"]
            for ch, data in c["channels"].items()
        ]
        self.summary.setRowCount(len(rows))

        def value(v):
            return "—" if v is None else f"{v:.4g}"

        for row, (camera, channel, data) in enumerate(rows):
            s = data["summary"]
            values = [
                f"{camera['serial']} / {camera['gain_db']:g} / {channel}",
                value(s["read_noise_rms_dn"]),
                value(s["read_noise_rms_e"]),
                value(s["gain_dn_per_e"]),
                value(s["effective_saturation_capacity_e"]),
                value(s["dynamic_range_db"]),
                f"R² {data['fit']['r_squared']:.3f}"
                if data["fit"]["valid"]
                else ("차광 검증 통과" if dark_only else "조건 확인"),
            ]
            for column, text in enumerate(values):
                self.summary.setItem(row, column, QTableWidgetItem(text))
        self.summary.setVisible(not self.graph_zoomed)
        self.summary.setFixedHeight(
            min(164, self.summary.horizontalHeader().height() + len(rows) * 30 + 3)
        )
        self.plot_choice.clear()
        for plot in report["plots"]:
            self.plot_choice.addItem(plot["label"], plot["path"])
        self._plot_changed()
        self._show_view()
        warnings = [
            w for camera in report["cameras"] for w in camera.get("warnings", [])
        ]
        self.quality.setText(
            " / ".join(dict.fromkeys(warnings))
            if warnings
            else "RAW·메타데이터·통계·결과 파일 무결성 검사 통과"
        )
        self.quality.setToolTip(self.quality.text())
        self.quality.setMaximumHeight(58)
        self.status.setText(
            ("가상 데이터 · 실제 센서 측정 아님 · " if report["simulated"] else "")
            + (
                "측정 간 비교 완료"
                if comparison
                else f"{'차광 분석' if dark_only else '분석'} 완료 · 그래프 {len(report['plots'])}개 · RAW 원본·결과 저장"
            )
        )
        if (
            not comparison
            and not dark_only
            and report["noise_config"]["protocol"] == "automatic_ptc"
        ):
            usable = min(
                sum(r["fit_used"] for r in data["curve"]) for _, _, data in rows
            )
            rejected = sum(
                r["rejected_groups"]
                for c in report["cameras"]
                for r in c.get("illumination_filter", [])
            )
            self.quality.setText(
                f"PTC 사용 가능 단계 최소 {usable}개 · 조명 이상 제외 {rejected}회 · 자세한 검사 결과는 이 문장에 마우스를 올리세요."
            )
            if not any(data["fit"]["valid"] for _, _, data in rows):
                self.status.setText(
                    "PTC 계산 불가 · 조명 변동/포화 등 조건 확인 · 차광 유지, ②로 밝은 촬영만 다시"
                )

    def _plot_changed(self, *_):
        relative = self.plot_choice.currentData()
        if relative:
            image = QImage(str(Path(self.plot_base) / relative))
            self.canvas.set_image(image if not image.isNull() else None)

    def update_images(self, images):
        for index, (camera, image, settings, frame_id) in enumerate(images[:3]):
            card = self.camera_cards[index]
            card.update_frame(camera, image, settings, frame_id)
            x, y, w, h = roi_bounds((settings["height"], settings["width"]), 256)
            card.image.roi = (
                x / settings["width"],
                y / settings["height"],
                w / settings["width"],
                h / settings["height"],
            )
            if self.camera_zoom_index == index:
                self.live_focus.roi = card.image.roi
                self.live_focus.set_image(image)

    def _switch_view(self):
        self.showing_graph = not self.showing_graph
        self.graph_zoomed = False
        self.camera_zoom_index = None
        self._show_view()

    def _show_view(self):
        self.display_deck.setCurrentWidget(
            self.canvas
            if self.showing_graph
            else self.live_focus
            if self.camera_zoom_index is not None
            else self.live_grid
        )
        self.plot_choice.setVisible(self.showing_graph and self.report is not None)
        self.view_button.setText("라이브 화면" if self.showing_graph else "그래프 보기")
        self.reflow()

    def _camera_zoom(self, index):
        if self.camera_cards[index].saved_image is None:
            return
        self.camera_zoom_index = index
        self.live_focus.roi = getattr(self.camera_cards[index].image, "roi", None)
        self.live_focus.set_image(self.camera_cards[index].saved_image)
        self._show_view()

    def _camera_unzoom(self):
        self.camera_zoom_index = None
        self._show_view()

    def load_session(self, path, auto_analysis=True):
        session = json.loads((Path(path) / "session.json").read_text())
        self.retaking_bright = bool(session.get("bright_retake_from"))
        if session.get("kind") != "rawrig_noise":
            raise CaptureError("RAW Rig 노이즈 측정 폴더가 아닙니다.")
        self.load_config(session["noise_config"])
        self.session_config = asdict(NoiseConfig.from_dict(session["noise_config"]))
        self.session_path = str(Path(path).resolve())
        self.dark_complete = (Path(path) / "DARK_COMPLETE.json").is_file()
        self.bright_complete = (Path(path) / "BRIGHT_COMPLETE.json").is_file()
        self.path_label.setText(self.session_path)
        self.report = None
        self.showing_graph = False
        self.view_button.hide()
        self.plot_choice.clear()
        self.canvas.set_image(None)
        self.summary.hide()
        self._show_view()
        self._show_operating_settings(session)
        if auto_analysis and (session["status"] == "complete" or self.dark_complete):
            self.analysis_requested.emit(self.session_path)
        else:
            self.status.setText(
                "이전 측정 선택됨 · 차광 자료는 밝은 촬영 전에 다시 검사합니다."
            )
        self._update()

    def _load_dialog(self):
        path = QFileDialog.getExistingDirectory(
            self, "노이즈 측정 세션 폴더 선택", self.session_path
        )
        if path:
            try:
                self.load_session(path)
            except (OSError, KeyError, ValueError, CaptureError) as error:
                self.notify.emit(str(error))

    def _compare_dialog(self):
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "비교할 측정의 analysis/report.json 선택",
            self.session_path,
            "JSON (*.json)",
        )
        if files:
            paths = [self.session_path] + [str(Path(p).parent.parent) for p in files]
            self.comparison_requested.emit(list(dict.fromkeys(paths)))

    def _zoom(self):
        self.graph_zoomed = not self.graph_zoomed
        self.summary.setVisible(not self.graph_zoomed and self.report is not None)
        self.quality.setVisible(not self.graph_zoomed)
        self.reflow()

    def _toggle(self):
        self.overlay_open = not self.overlay_open
        self.reflow()

    def reflow(self):
        overlay = self.width() < 1000
        zoomed = self.graph_zoomed or self.camera_zoom_index is not None
        self.status.setFixedHeight(24 if self.height() < 450 else 38)
        self.status.setWordWrap(self.height() >= 450)
        self.quality.setVisible(not zoomed and self.height() >= 450)
        resume = bool(
            self.session_path
            and not self.dark_complete
            and self.session_config == asdict(self.settings())
        )
        self.dark_button.setText(
            "① 차광 이어 촬영"
            if resume
            else "① 차광 촬영"
            if self.width() < 800
            else "① 차광 RAW 연속 촬영"
        )
        self.bright_button.setText(
            ("② 밝은 재촬영" if self.width() < 800 else "② 밝은 재촬영 · 차광 재사용")
            if self.bright_complete or self.retaking_bright
            else (
                "② 밝은 촬영·분석"
                if self.width() < 800
                else "② 밝은 RAW 촬영 · 자동 분석"
            )
        )
        self.stop_button.setText("중지" if self.width() < 800 else "작업 중지")
        self.toggle.setVisible(overlay and not zoomed)
        self.close_button.setVisible(overlay)
        if overlay:
            if self.body.indexOf(self.inspector) >= 0:
                self.body.removeWidget(self.inspector)
            self.inspector.setGeometry(
                max(16, self.width() - 366), 16, 350, max(0, self.height() - 32)
            )
            self.inspector.setVisible(self.overlay_open and not zoomed)
            self.inspector.raise_()
        else:
            if self.body.indexOf(self.inspector) < 0:
                self.body.addWidget(self.inspector)
            self.inspector.setVisible(not zoomed)
        self.toggle.setText("설정 닫기  ×" if self.overlay_open else "측정 설정  ☰")
        self.summary.setVisible(
            self.showing_graph
            and self.report is not None
            and not zoomed
            and self.height() >= 450
        )
        self.quality.setVisible(not zoomed and self.height() >= 450)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        QTimer.singleShot(0, self.reflow)
        QTimer.singleShot(0, self._fit_inspector_text)
