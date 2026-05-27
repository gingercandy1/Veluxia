from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QGuiApplication, QFont, QTextCursor
from PySide6.QtWidgets import (
    QFrame, QWidget, QVBoxLayout, QHBoxLayout,
    QTextEdit, QLabel, )
from pygments import highlight
from pygments.formatters.html import HtmlFormatter
from pygments.lexers import get_lexer_by_name, guess_lexer
from pygments.util import ClassNotFound

from ui.base.action_button import ActionButton
from ui.base.widget import BaseWidget


class CodeBlockWidget(BaseWidget):
    """带语言标签、语法高亮、悬停复制按钮的代码块。"""

    def __init__(self, code: str="", lang: str = "", parent=None):
        super().__init__(parent)
        self.setObjectName("code_block_widget")
        self.code = code
        self.lang = lang.strip() or "text"
        self._is_streaming = True

        self._build_ui()
        if code:
            self._apply_highlight()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, -10, -10)
        layout.setSpacing(0)
        layout.addLayout(self._build_top_bar())
        layout.addWidget(self._build_editor())
        self._fit_editor_height()

        self._copy_btn = ActionButton(
            svg_str=":svg/copy.svg",
            tooltip="复制消息",
            width=26, height=26,
            icon_size_width=21,
            icon_size_height=21,
            parent=self
        )
        self._copy_btn.setVisible(False)
        self._copy_btn.clicked.connect(self._on_copy)

        x = self._editor.geometry().right() + 80
        y = self._editor.y() + self._lang_label.height() + 35
        self._copy_btn.move(x, y)

    def _build_top_bar(self) -> QHBoxLayout:
        bar = QHBoxLayout()
        bar.setContentsMargins(10, 0, 10, 0)

        self._lang_label = QLabel(self.lang)
        self._lang_label.setObjectName("lang_label")
        self._lang_label.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self._lang_label.setFixedSize(60, 24)

        bar.addWidget(self._lang_label)
        bar.addStretch()
        return bar

    def _build_editor(self) -> QTextEdit:
        self._editor = QTextEdit()
        self._editor.setReadOnly(True)
        self._editor.setFrameShape(QFrame.NoFrame)
        self._editor.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._editor.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        font = QFont("Cascadia Code", 12)
        font.setStyleHint(QFont.Monospace)
        self._editor.setFont(font)
        return self._editor

    # ── 语法高亮 ─────────────────────────────────────────────────────────────
    _CODE_FONT = "JetBrains Mono, Cascadia Code, Consolas, monospace"
    _CODE_SIZE = "11.5pt"
    _LINE_HEIGHT = "1.3"

    def _apply_highlight(self):
        lexer = self._resolve_lexer()
        formatter = HtmlFormatter(style="monokai", noclasses=True, nowrap=True)
        highlighted = highlight(self.code, lexer, formatter)
        self._editor.setHtml(f"""
            <pre style="
                margin: 0;
                padding: 14px 16px;
                background: #0d1117;
                border-radius: 0 0 6px 6px;
                font-family: {self._CODE_FONT};
                font-size: {self._CODE_SIZE};
                line-height: {self._LINE_HEIGHT};
                letter-spacing: 0.2px;
            ">{highlighted}</pre>
        """)
        self._fit_editor_height()

    def _fit_editor_height(self):
        doc = self._editor.document()
        doc.setTextWidth(self._editor.viewport().width() or 600)

        height = self.get_textedit_content_height()
        self._editor.setFixedHeight(height)

        widget_height = self.layout().contentsMargins().top() + \
                        self.layout().contentsMargins().bottom() + \
                        self._editor.height()

        self.setFixedHeight(self._editor.height() + 60)

    def get_textedit_content_height(self) -> int:
        """
        计算 QTextEdit 内容实际高度
        """
        doc = self._editor.document()
        fm = self._editor.fontMetrics()
        line_height = fm.lineSpacing()
        doc_height = doc.documentLayout().documentSize().height()

        content_height = max(int(doc_height), line_height) + 30
        # 加上上下边距
        margins = self._editor.contentsMargins()
        frame_margin = int(doc.documentMargin()) * 2
        total_height = content_height + margins.top() + margins.bottom() + frame_margin
        return total_height

    def _resolve_lexer(self):
        try:
            if self.lang and self.lang != "text":
                return get_lexer_by_name(self.lang, stripnl=False)
            return guess_lexer(self.code)
        except ClassNotFound:
            return get_lexer_by_name("text")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._fit_editor_height()

    def enterEvent(self, event):
        self._copy_btn.setVisible(True)

    def leaveEvent(self, event):
        self._copy_btn.setVisible(False)

    def _on_copy(self):
        QGuiApplication.clipboard().setText(self.code)

    def append_chunk(self, text: str) -> None:
        """
        流式追加代码字符。
        策略：
          - 每个字符：insertText 追加（无高亮，O(1)）
          - 遇到换行：QTimer.singleShot(0) 触发全量高亮（低频，下一帧执行）

        用 singleShot(0) 的原因：同一帧内可能连续来多个换行，
        合并成一次 setHtml 避免闪烁。
        """
        self.code += text

        # 先用 insertText 追加，保证打字效果流畅
        self._editor.setUpdatesEnabled(False)
        cursor = QTextCursor(self._editor.document())  # 注：可复用，此处简化
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertText(text)
        self._editor.setUpdatesEnabled(True)
        self._fit_editor_height()

        # 换行时触发一次全量高亮（防抖：同帧多次只执行一次）
        if "\n" in text:
            if not getattr(self, "_highlight_pending", False):
                self._highlight_pending = True
                QTimer.singleShot(0, self._do_highlight)

    def _do_highlight(self) -> None:
        """在事件循环下一帧执行全量高亮。"""
        self._highlight_pending = False
        # 记录滚动位置，setHtml 后恢复
        sb = self._editor.verticalScrollBar()
        prev = sb.value() if sb else 0
        self._apply_highlight()
        if sb:
            sb.setValue(prev)
        # 重置游标到末尾，保证后续 insertText 继续追加
        cursor = QTextCursor(self._editor.document())
        cursor.movePosition(QTextCursor.MoveOperation.End)

    def finish(self) -> None:
        """流结束，最终高亮一次并固定高度。"""
        self._is_streaming = False
        self._apply_highlight()
        self._fit_editor_height()