"""Responsive camera tiles and paint-based preview without pixmap size constraints."""
from PySide6.QtCore import Qt, Signal, QRectF, QPointF, QEvent, QTimer
from PySide6.QtGui import QPainter, QColor, QPainterPath, QFont, QWheelEvent, QPen
from PySide6.QtWidgets import (QWidget, QFrame, QLabel, QVBoxLayout, QHBoxLayout, QGridLayout,
                              QSizePolicy, QComboBox, QDoubleSpinBox, QSpinBox,
                              QApplication, QAbstractScrollArea, QScrollArea)


class StableScrollArea(QScrollArea):
    """Preserve the user's scroll position during updates and focus reassignment."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.scroll_locked = False
        self.scroll_anchor = 0
        self.manual_scroll = False
        self.verticalScrollBar().installEventFilter(self)
        self.verticalScrollBar().valueChanged.connect(self._scroll_changed)
        self.verticalScrollBar().rangeChanged.connect(self._restore_scroll)

    def lock_scroll(self):
        if not self.scroll_locked:
            self.scroll_anchor = self.verticalScrollBar().value()
            self.scroll_locked = True

    def unlock_scroll(self):
        self._restore_scroll()
        self.scroll_locked = False

    def _restore_scroll(self, *_):
        if self.scroll_locked and not self.manual_scroll:
            self.verticalScrollBar().setValue(min(self.scroll_anchor, self.verticalScrollBar().maximum()))

    def _scroll_changed(self, value):
        if self.manual_scroll:
            self.scroll_anchor = value
        else:
            self._restore_scroll()

    def viewportEvent(self, event):
        if event.type() == QEvent.Type.Wheel:
            self.manual_scroll = True
            try:
                return super().viewportEvent(event)
            finally:
                self.scroll_anchor = self.verticalScrollBar().value()
                self.manual_scroll = False
        return super().viewportEvent(event)

    def eventFilter(self, watched, event):
        if watched is self.verticalScrollBar():
            if event.type() in (QEvent.Type.MouseButtonPress, QEvent.Type.KeyPress, QEvent.Type.Wheel):
                self.manual_scroll = True
                if event.type() != QEvent.Type.MouseButtonPress:
                    QTimer.singleShot(0, self._end_manual_scroll)
            elif event.type() == QEvent.Type.MouseButtonRelease:
                QTimer.singleShot(0, self._end_manual_scroll)
        return super().eventFilter(watched, event)

    def _end_manual_scroll(self):
        self.scroll_anchor = self.verticalScrollBar().value()
        self.manual_scroll = False


class AdaptiveColumns(QWidget):
    """Pack panels into independent columns without equal-height grid-row gaps."""
    def __init__(self, widgets, minimum_column_width=460, maximum_columns=4, parent=None):
        super().__init__(parent)
        self.panels = list(widgets)
        self.minimum_column_width = minimum_column_width
        self.maximum_columns = maximum_columns
        self.columns = 0
        self.panel_layout = QHBoxLayout(self)
        self.panel_layout.setContentsMargins(0, 0, 0, 0)
        self.panel_layout.setSpacing(18)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
        self.reflow(1)

    def reflow(self, columns):
        if columns == self.columns:
            return
        while self.panel_layout.count():
            item = self.panel_layout.takeAt(0)
            column = item.layout()
            while column.count():
                column.takeAt(0)
            column.deleteLater()
        stacks = [QVBoxLayout() for _ in range(columns)]
        heights = [0] * columns
        for panel in self.panels:
            # Keep each panel at its natural height, then fill the shortest column.
            index = min(range(columns), key=heights.__getitem__)
            stacks[index].addWidget(panel)
            heights[index] += panel.sizeHint().height() + 18
        for stack in stacks:
            stack.setSpacing(18)
            stack.addStretch()
            self.panel_layout.addLayout(stack, 1)
        self.columns = columns
        self.updateGeometry()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        spacing = self.panel_layout.spacing()
        columns = max(1, min(self.maximum_columns,
                             (self.width() + spacing) // (self.minimum_column_width + spacing)))
        self.reflow(columns)


class _ScrollSafeInput:
    """Route wheel gestures to scrolling containers, even while editing a value."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        if self.lineEdit() is not None:
            self.lineEdit().installEventFilter(self)

    def wheelEvent(self, event):
        parent = self.parentWidget()
        while parent is not None:
            if isinstance(parent, QAbstractScrollArea):
                # Ignoring a child wheel event alone does not reliably scroll its parent.
                # Let Qt's scrollbar handle angle/pixel deltas, modifiers and phases.
                viewport = parent.viewport()
                forwarded = QWheelEvent(QPointF(viewport.mapFromGlobal(event.globalPosition().toPoint())),
                    event.globalPosition(), event.pixelDelta(), event.angleDelta(), event.buttons(),
                    event.modifiers(), event.phase(), event.inverted(), event.source(), event.pointingDevice())
                forwarded.setAccepted(False)
                QApplication.sendEvent(viewport, forwarded)
                if forwarded.isAccepted():
                    event.accept()
                    return
            parent = parent.parentWidget()
        event.ignore()

    def eventFilter(self, watched, event):
        if watched is self.lineEdit() and event.type() == QEvent.Type.Wheel:
            self.wheelEvent(event)
            return True
        return super().eventFilter(watched, event)


class ScrollSafeComboBox(_ScrollSafeInput, QComboBox):
    """The closed control never changes selection from a page-scroll gesture."""


class ScrollSafeDoubleSpinBox(_ScrollSafeInput, QDoubleSpinBox):
    """Values change only through explicit text, keys or step-button input."""


class ScrollSafeSpinBox(_ScrollSafeInput, QSpinBox):
    """Values change only through explicit text, keys or step-button input."""


class PreviewCanvas(QWidget):
    clicked = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.image = None
        self.placeholder = "카메라 연결을 기다리고 있습니다"
        self.setMinimumSize(140, 135)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip("클릭하여 크게 보기 · 다시 클릭하여 전체 카메라로 돌아가기")

    def set_image(self, image):
        self.image = image
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        bounds = QRectF(self.rect())
        path = QPainterPath()
        path.addRoundedRect(bounds, 10, 10)
        painter.setClipPath(path)
        painter.fillRect(bounds, QColor("#182032"))
        if self.image is not None:
            size = self.image.size()
            size.scale(self.size(), Qt.AspectRatioMode.KeepAspectRatio)
            target = QRectF((self.width() - size.width()) / 2, (self.height() - size.height()) / 2, size.width(), size.height())
            painter.drawImage(target, self.image)
            roi = getattr(self, "roi", None)
            if roi:
                x, y, w, h = roi
                painter.setPen(QPen(QColor("#bca5ff"), 2))
                painter.drawRect(QRectF(target.x() + x * target.width(), target.y() + y * target.height(), w * target.width(), h * target.height()))
        else:
            painter.setPen(QColor("#8190a8"))
            painter.setFont(QFont("Noto Sans CJK KR", 10))
            painter.drawText(bounds, Qt.AlignmentFlag.AlignCenter, self.placeholder)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mouseReleaseEvent(event)


class CameraCard(QFrame):
    activated = Signal(int)

    def __init__(self, index):
        super().__init__()
        self.index = index
        self.compact = False
        self.minimal = False
        self.current_camera = None
        self.setObjectName("cameraCard")
        self.setMinimumHeight(248)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout = QVBoxLayout(self)
        self.card_layout = layout
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(10)
        top = QHBoxLayout()
        self.title = QLabel(f"CAMERA {index + 1:02d}")
        self.title.setObjectName("cardTitle")
        self.status = QLabel("미연결")
        self.status.setObjectName("mutedBadge")
        top.addWidget(self.title)
        top.addStretch()
        top.addWidget(self.status)
        layout.addLayout(top)
        self.image = PreviewCanvas()
        layout.addWidget(self.image, 1)
        self.detail = QLabel("노출시간 — µs     게인 — dB")
        self.detail.setWordWrap(True)
        self.detail.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.detail.setObjectName("cameraDetail")
        self.model = QLabel("카메라를 선택하고 연결하세요")
        self.model.setObjectName("mutedText")
        self.model.setWordWrap(True)
        layout.addWidget(self.detail)
        layout.addWidget(self.model)
        self.saved_image = None
        self.image.clicked.connect(lambda: self.activated.emit(self.index))

    def update_frame(self, camera, image, settings, frame_id):
        self.current_camera = camera
        self.saved_image = image
        self.image.set_image(image)
        self.status.setText("DEMO" if camera.simulated else "LIVE")
        self.status.setObjectName("liveBadge")
        self.status.style().unpolish(self.status)
        self.status.style().polish(self.status)
        self.detail.setText(f"노출시간 {settings['exposure_us']:,.1f} µs   ·   게인 {settings['gain_db']:.2f} dB")
        self.model.setText(f"{camera.serial}  ·  {settings['width']} × {settings['height']}  ·  {settings['pixel_format']}")
        self.title.setToolTip(camera.model)
        self._update_title()
        self.setToolTip(f"{camera.serial} · {camera.model}\n{self.detail.text()}\n{self.model.text()}")

    def _update_title(self):
        self.title.setText(f"{self.index+1:02d} · {self.current_camera.serial}" if self.compact and self.current_camera
                           else f"CAMERA {self.index+1:02d}")

    def set_compact(self, compact, minimal=False):
        if self.compact == compact and self.minimal == minimal:
            return
        self.compact = compact
        self.minimal = minimal
        self.setMinimumHeight(72 if minimal else 132 if compact else 248)
        self.image.setMinimumSize(40 if compact else 140, 32 if minimal else 48 if compact else 135)
        self.card_layout.setContentsMargins(6, 6, 6, 6) if minimal else self.card_layout.setContentsMargins(8, 8, 8, 8) if compact else self.card_layout.setContentsMargins(14, 12, 14, 12)
        self.card_layout.setSpacing(4 if minimal else 6 if compact else 10)
        self.detail.setVisible(not minimal)
        self.model.setVisible(not compact)
        self._update_title()

    def clear(self):
        self.current_camera = None
        self._update_title()
        self.saved_image = None
        self.image.set_image(None)
        self.status.setText("미연결")
        self.status.setObjectName("mutedBadge")
        self.status.style().unpolish(self.status)
        self.status.style().polish(self.status)
        self.detail.setText("노출시간 — µs     게인 — dB")
        self.model.setText("카메라를 선택하고 연결하세요")


class CameraGrid(QWidget):
    """Keep the overview in two columns; adapt information density to its size."""
    def __init__(self, cards):
        super().__init__()
        self.cards = cards
        self.active_count = len(cards)
        self.columns = 0
        self.grid = QGridLayout(self)
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setSpacing(14)
        self.reflow(2)

    def set_count(self, count):
        count = max(0, min(len(self.cards), count))
        if count != self.active_count:
            self.active_count = count
            self.columns = 0
            self.reflow(1 if count <= 1 else 2)
            self._adapt_density()

    def reflow(self, columns):
        if self.columns == columns:
            return
        while self.grid.count():
            self.grid.takeAt(0)
        for row in range(3):
            self.grid.setRowStretch(row, 0)
            self.grid.setColumnStretch(row, 0)
        for index, card in enumerate(self.cards):
            card.setVisible(index < max(1, self.active_count))
            if index < max(1, self.active_count):
                self.grid.addWidget(card, index // columns, index % columns)
                self.grid.setRowStretch(index // columns, 1)
        for column in range(columns):
            self.grid.setColumnStretch(column, 1)
        self.columns = columns

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._adapt_density()

    def _adapt_density(self):
        spacing = self.grid.spacing()
        columns = 1 if self.active_count <= 1 else 2
        rows = (max(1, self.active_count) + columns - 1) // columns
        row_height = (self.height() - (rows - 1) * spacing) // rows
        compact = (row_height < 248
                   or (self.width() - (columns - 1) * spacing) // columns < 240)
        for card in self.cards:
            card.set_compact(compact, minimal=row_height < 132)
