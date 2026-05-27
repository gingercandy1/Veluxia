import asyncio
from typing import AsyncGenerator, Callable, Optional

from PySide6.QtCore import (
    Qt, QTimer, QPropertyAnimation, QEasingCurve, )
from PySide6.QtGui import (
    QColor, QPainter, QPen, QConicalGradient, QTextCursor, QFont,
)
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QTextEdit, QSizePolicy, QFrame,
)


class _InlineSpinner(QWidget):
    SIZE = 16

    def __init__(self, color: str = "#a0a0a0", parent=None):
        super().__init__(parent)
        self._color = QColor(color)
        self._angle = 0
        self._timer = QTimer(self)
        self._timer.setInterval(1000 // 60)
        self._timer.timeout.connect(self._tick)
        self.setFixedSize(self.SIZE, self.SIZE)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

    def start(self):
        self._timer.start()

    def stop(self):
        self._timer.stop()
        self.update()

    def _tick(self):
        self._angle = (self._angle + 6) % 360
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        s = self.SIZE
        w = 2  # ring width
        r = s // 2 - w - 1

        p.translate(s / 2, s / 2)

        # 背景圆环
        track = QColor(self._color)
        track.setAlphaF(0.15)
        p.setPen(QPen(track, w, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        p.drawEllipse(-r, -r, r * 2, r * 2)

        # 渐变弧段
        grad = QConicalGradient(0, 0, -self._angle)
        head = QColor(self._color)
        head.setAlphaF(1.0)
        tail = QColor(self._color)
        tail.setAlphaF(0.0)
        mid = QColor(self._color)
        mid.setAlphaF(0.55)
        grad.setColorAt(0.0, head)
        grad.setColorAt(0.4, mid)
        grad.setColorAt(0.72, tail)
        grad.setColorAt(1.0, tail)

        p.setPen(QPen(grad, w, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        p.drawArc(-r, -r, r * 2, r * 2,
                  (-self._angle + 90) * 16,
                  -260 * 16)
        p.end()

class _ThinkingTextArea(QTextEdit):
    MAX_H = 200  # 折叠态最大高度

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("thinking_text_area")
        self.setReadOnly(True)
        self.setMaximumHeight(self.MAX_H)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setStyleSheet("""
            QTextEdit#thinking_text_area {
                background:    transparent;
                border:        none;
                color:         rgba(180, 180, 180, 0.75);
                font-size:     13px;
                padding:       0px 4px;
                line-height:   1.5;
                selection-background-color: rgba(255,255,255,0.1);
            }
            QScrollBar:vertical {
                width: 4px;
                background: transparent;
            }
            QScrollBar::handle:vertical {
                background: rgba(255,255,255,0.15);
                border-radius: 2px;
            }
        """)

        self._target_height = 0
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def set_target_height(self, height: int):
        self._target_height = max(0, height)
        self.setFixedHeight(self._target_height)

    def append_chunk(self, text: str):
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertText(text)
        self.setTextCursor(cursor)
        self.ensureCursorVisible()


class ThinkingBlock(QFrame):
    """
    完整的思考过程展示组件。
    布局：
    ┌─────────────────────────────────────────────┐
    │ [spinner]  思考中...              [展开 ▾]  │  ← _header_row
    ├─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┤
    │  思考文本流式追加…                           │  ← _text_area（可折叠）
    └─────────────────────────────────────────────┘
    """
    THINKING = "thinking"
    GENERATING = "generating"
    DONE = "done"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("thinking_block")
        self._state = self.THINKING
        self._expanded = True
        self._elapsed_ms = 0
        self._build_ui()

        # 折叠动画
        self._anim = QPropertyAnimation(self._text_area, b"height")
        self._anim.setDuration(200)
        self._anim.setEasingCurve(QEasingCurve.Type.InOutCubic)
        self._anim.finished.connect(self._on_anim_finished)

        # 启动 spinner
        self._spinner.start()

    def _on_anim_finished(self):
        if self._expanded:
            self._text_area.setMaximumHeight(self._get_content_height())
        else:
            self._text_area.setMaximumHeight(0)

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 8, 12, 8)
        root.setSpacing(8)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(6)

        self._spinner = _InlineSpinner(color="#8888aa")

        self._status_label = QLabel("思考中...")
        self._status_label.setObjectName("thinking_status_label")
        font = self._status_label.font()
        font.setPointSize(12)
        font.setStyleStrategy(QFont.StyleStrategy.PreferAntialias)
        self._status_label.setFont(font)

        self._toggle_btn = QLabel("收起 ▴")
        self._toggle_btn.setObjectName("thinking_toggle_btn")
        self._toggle_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._toggle_btn.mousePressEvent = lambda _: self._toggle_expand()

        header.addWidget(self._spinner)
        header.addWidget(self._status_label)
        header.addStretch()
        header.addWidget(self._toggle_btn)

        # 分隔线
        self._sep = QFrame()
        self._sep.setFrameShape(QFrame.Shape.HLine)
        self._sep.setObjectName("thinking_sep")

        # 思考文本区
        self._text_area = _ThinkingTextArea()

        root.addLayout(header)
        root.addWidget(self._sep)
        root.addWidget(self._text_area)

    def _toggle_expand(self):
        self._expanded = not self._expanded
        self._run_expand_anim(self._expanded)
        self._toggle_btn.setText("收起 ▴" if self._expanded else "展开 ▾")

    def _get_content_height(self) -> int:
        doc = self._text_area.document()
        doc.setTextWidth(self._text_area.viewport().width() or 400)
        content_h = int(doc.size().height()) + 8  # +8 留点内边距
        return min(content_h, _ThinkingTextArea.MAX_H)

    def _run_expand_anim(self, expand: bool):
        if self._anim.state() == QPropertyAnimation.State.Running:
            self._anim.stop()

        start = self._text_area.height()
        end = self._get_content_height() if expand else 0

        self._anim.setStartValue(start)
        self._anim.setEndValue(end)
        self._anim.start()

    def _collapse_text_area(self):
        """生成阶段自动折叠思考区"""
        self._expanded = False
        self._run_expand_anim(False)
        self._toggle_btn.setText("展开 ▾")

    def append_thinking(self, chunk: str):
        """THINKING 阶段：流式追加思考文本"""
        if self._state != self.THINKING:
            return
        self._text_area.append_chunk(chunk)

        if self._expanded:
            new_h = self._get_content_height()
            self._text_area.setMaximumHeight(new_h)

    def switch_to_generating(self):
        if self._state != self.THINKING:
            return
        self._state = self.GENERATING

        self._spinner.stop()
        self._spinner.setVisible(False)
        self._status_label.setText("已深度思考")
        self._collapse_text_area()

    def finish(self, elapsed_ms: int = 0):
        """
        stream_done 时调用。
        显示思考耗时，状态置为 DONE。
        """
        if self._state == self.DONE:
            return
        self._elapsed_ms = elapsed_ms
        self._state = self.DONE

        self._spinner.stop()
        self._spinner.setVisible(False)

        secs = elapsed_ms / 1000
        if secs >= 1:
            time_str = f"{secs:.1f}s" if secs < 60 else f"{int(secs // 60)}m{int(secs % 60)}s"
            self._status_label.setText(f"已深度思考  ·  {time_str}")
        else:
            self._status_label.setText("已深度思考")

    def hide_think_area(self):
        self.setVisible(False)



