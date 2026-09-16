import os.path
from pathlib import Path
from typing import Optional, Union

from PySide6.QtCore import (
    Qt, QPropertyAnimation, QEasingCurve, Signal, QUrl, QTimer, )
from PySide6.QtGui import QColor, QPainter, QPainterPath, QLinearGradient, QClipboard, QFont, QDesktopServices, QPen, \
    QConicalGradient
from PySide6.QtWidgets import (
    QFrame, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QSizePolicy, QGraphicsOpacityEffect, QMessageBox, QApplication, QPushButton, QTextEdit,
)

from src.app.ui.base.action_button import ActionButton
from src.app.ui.base.widget import BaseWidget
from src.app.ui.message.content_factory import ContentLoader, ContentBuilder
from src.app.ui.message.text.streaming_renderer import StreamingRenderer
from src.app.ui.message.think_bubble import ThinkingBlock
from src.resources import *


class FadeMask(QWidget):
    MASK_HEIGHT = 150

    def __init__(self, bg_color: QColor, parent=None):
        super().__init__(parent)
        self._bg = QColor(bg_color)
        self.is_expanded = False
        self.is_reverse = True

        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setVisible(False)

    def set_bg_color(self, color: QColor):
        self._bg = QColor(color)
        self.update()

    def set_expanded(self, expanded: bool):
        self.is_expanded = expanded

    def set_reverse(self, reverse: bool):
        self.is_reverse = reverse
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        grad = QLinearGradient(0, 0, 0, self.height())

        top = QColor(self._bg)
        top.setAlpha(0)
        bot = QColor(self._bg)
        bot.setAlpha(255)

        if self.is_reverse:
            grad.setColorAt(0.0, top)
            grad.setColorAt(1.0, bot)
        else:
            grad.setColorAt(0.0, bot)
            grad.setColorAt(1.0, top)


        path = QPainterPath()
        path.addRoundedRect(self.rect().adjusted(1, 1, -1, -1), 8, 8)
        p.setPen(Qt.PenStyle.NoPen)
        p.fillPath(path, grad)
        p.end()


class MaskContainer(BaseWidget):
    """带渐变遮罩 + 右下角展开/收起按钮的容器"""
    expanded = Signal(bool)  # True = 展开，False = 收起

    def __init__(self,
                 bg_color: QColor = QColor("#000000"),
                 button_width: int = 120,
                 button_height: int = 30,
                 parent=None):
        super().__init__(parent)
        self.setObjectName("mask_container")

        self._bg = QColor(bg_color)
        self.is_expanded = False

        # ==================== FadeMask ====================
        self.fade_mask = FadeMask(bg_color, self)
        self.fade_mask.setVisible(True)

        # ==================== 按钮 ====================
        self.btn = QPushButton("Show more ↓", self)
        font = self.btn.font() # 获取当前字体
        font.setStyleStrategy(QFont.StyleStrategy.PreferAntialias)
        font.setHintingPreference(QFont.HintingPreference.PreferNoHinting)
        self.btn.setFont(font)
        self.btn.setFixedSize(button_width, button_height)
        self.btn.setCursor(Qt.CursorShape.PointingHandCursor)

        self._setup_button_style()
        self.btn.clicked.connect(self._on_button_clicked)

        self.setVisible(False)

    def _setup_button_style(self):
        self.btn.setStyleSheet("""
            QPushButton {
                background: transparent;
                color: rgba(255, 255, 255, 210);
                border: none;
                border-radius: 5px;
                font-size: 13px;
            }
            QPushButton:hover {
                background: transparent;
                color: white;
            }
            QPushButton:pressed {
                background: transparent;
                color: white;
            }
        """)

    def _on_button_clicked(self):
        self.is_expanded = not self.is_expanded
        self._update_ui()
        self.expanded.emit(self.is_expanded)

    def _update_ui(self):
        label = "Show less ↑" if self.is_expanded else "Show more ↓"
        self.btn.setText(label)
        self.fade_mask.setVisible(not self.is_expanded)

    # ==================== 公开接口 ====================
    def set_bg_color(self, color: QColor):
        self._bg = QColor(color)
        self.fade_mask.set_bg_color(color)

    def set_expanded(self, expanded: bool):
        self.is_expanded = expanded
        self._update_ui()

    def resizeEvent(self, event):
        super().resizeEvent(event)

        # FadeMask 铺满整个容器
        self.fade_mask.setGeometry(-1, 0, self.width()+2, self.height())

        # 按钮定位到右下角
        margin_right = 12
        margin_bottom = 10
        x = self.width() - self.btn.width() - margin_right
        y = self.height() - self.btn.height() - margin_bottom
        self.btn.move(x, y)


class CollapseContainer(QFrame):
    def __init__(self, bubble_box: QFrame, parent=None):
        super().__init__(parent)
        self.setObjectName("collapse_container")
        self.setContentsMargins(0, 0, 0, 0)
        self._bubble_box = bubble_box

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(bubble_box)

        self._mask = MaskContainer(parent=self)
        self._anim = QPropertyAnimation(self, b"maximumHeight")
        self._anim.setDuration(220)
        self._anim.setEasingCurve(QEasingCurve.Type.InOutCubic)

    def show_mask(self, visible: bool):
        self._mask.setVisible(visible)
        if visible:
            self._reposition_mask()

    def set_mask_color(self, color: QColor):
        self._mask.set_bg_color(color)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._reposition_mask()

    def _reposition_mask(self):
        if not self._mask.is_expanded:
            h = FadeMask.MASK_HEIGHT
        else:
            h = 30
        self._mask.setGeometry(
            0,
            self.height() - h + 3,
            self.width(),
            h,
        )
        self._mask.raise_()

    def animate_height(self, target_h: int):
        """平滑动画到目标高度。"""
        if self._anim.state() == QPropertyAnimation.State.Running:
            self._anim.stop()
        self._anim.setStartValue(self.height())
        self._anim.setEndValue(target_h)
        self._anim.start()


class BottomActionBar(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.opacity_effect = QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(self.opacity_effect)
        self.opacity_effect.setOpacity(0)

        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(12, 6, 12, 8)
        self.layout.setSpacing(8)

        self.setFixedHeight(35)

    def add_button(self, svg_path, tooltip):
        btn = ActionButton(
            svg_str=svg_path,
            tooltip=tooltip
        )
        self.layout.addWidget(btn)
        return btn

    def show_bar(self):
        self.opacity_effect.setOpacity(1)

    def hide_bar(self):
        self.opacity_effect.setOpacity(0)

    def _find_parent_bubble(self) -> Optional['MessageBubble']:
        """向上查找父级的 MessageBubble"""
        parent = self.parent()
        while parent:
            if isinstance(parent, MessageBubble):
                return parent
            parent = parent.parent()
        return None


class UserBottomActionBar(BottomActionBar):
    copy_clicked = Signal()
    edit_clicked = Signal()
    retry_clicked = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName('user_bottom_action_bar')

        # 使用新的 ActionButton 类
        self._copy_btn = self.add_button(":/svg/copy.svg", "复制消息")
        self._edit_btn = self.add_button(":/svg/edit.svg", "编辑消息")
        self._retry_btn = self.add_button(":/svg/refresh.svg", "重新发送")
        self._translate_btn = self.add_button(":/svg/translate.svg", "翻译")

        self.layout.addStretch()

        self._copy_btn.clicked.connect(self._on_copy)
        self._edit_btn.clicked.connect(self._on_edit)
        self._retry_btn.clicked.connect(self._on_retry)

    def _on_copy(self):
        parent_bubble = self._find_parent_bubble()
        if not parent_bubble:
            return

        text_to_copy = ""
        if hasattr(parent_bubble, '_content') and isinstance(parent_bubble._content, str):
            text_to_copy = parent_bubble._content
        elif hasattr(parent_bubble, '_content_widget'):
            widget = parent_bubble._content_widget
            if hasattr(widget, 'toPlainText'):
                text_to_copy = widget.toPlainText()
            elif hasattr(widget, 'text'):
                text_to_copy = widget.text()

        if text_to_copy.strip():
            clipboard: QClipboard = QApplication.clipboard()
            clipboard.setText(text_to_copy.strip(), QClipboard.Mode.Clipboard)
            QMessageBox.information(None, "已复制", "消息内容已复制到剪贴板", QMessageBox.StandardButton.Ok)
            print("✅ 消息已复制到剪贴板")
        else:
            print("⚠️ 没有可复制的内容")
        self.copy_clicked.emit()

    def _on_edit(self):
        parent_bubble = self._find_parent_bubble()
        if parent_bubble and hasattr(parent_bubble, 'is_user') and parent_bubble.is_user:
            self.edit_clicked.emit()
        else:
            QMessageBox.warning(None, "提示", "仅支持编辑用户自己的消息")

    def _on_retry(self):
        self.retry_clicked.emit()


class AssistantBottomActionBar(BottomActionBar):
    copy_clicked = Signal()
    export_clicked = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName('assistant_bottom_action_bar')
        # 使用新的 ActionButton 类
        self._copy_btn = self.add_button(":/svg/copy.svg","复制消息")
        self._export_btn = self.add_button(":/svg/export.svg", "导出")
        self._translate_btn = self.add_button(":/svg/translate.svg", "翻译")
        self.layout.addStretch()

        self._copy_btn.clicked.connect(self._on_copy)
        self._export_btn.clicked.connect(self._on_export)
        self._translate_btn.clicked.connect(self._on_translate)

    def _on_copy(self):
        parent_bubble = self._find_parent_bubble()
        if not parent_bubble:
            return

        text_to_copy = ""
        if hasattr(parent_bubble, '_content') and isinstance(parent_bubble._content, str):
            text_to_copy = parent_bubble._content
        elif hasattr(parent_bubble, '_content_widget'):
            widget = parent_bubble._content_widget
            if hasattr(widget, 'toPlainText'):
                text_to_copy = widget.toPlainText()
            elif hasattr(widget, 'text'):
                text_to_copy = widget.text()

        if text_to_copy.strip():
            clipboard: QClipboard = QApplication.clipboard()
            clipboard.setText(text_to_copy.strip(), QClipboard.Mode.Clipboard)
            QMessageBox.information(None, "已复制", "消息内容已复制到剪贴板", QMessageBox.StandardButton.Ok)
            print("✅ 消息已复制到剪贴板")
        else:
            print("⚠️ 没有可复制的内容")
        self.copy_clicked.emit()

    def _on_export(self):
        self.export_clicked.emit()

    def _on_translate(self):
        pass


class MessageBubble(QFrame):
    """通用消息气泡，包含 meta 行、气泡框与折叠控制。"""
    BUBBLE_MAX_WIDTH = 800
    BUBBLE_MIN_WIDTH = 500
    COLLAPSE_MAX_H   = 400

    retry_requested = Signal(str)
    content_edited = Signal(str, str)

    def __init__(
            self,
            role: str,
            content: Union[str, dict, Path],
            timestamp: str,
            message_id: str = None,
            parent=None,
    ):
        super().__init__(parent)
        self.setObjectName("bubble")
        self.timestamp = timestamp
        self.role      = role
        self.is_user   = role == "user"
        self.message_id = message_id if message_id else f"msg_{id(self)}"
        self.raw_content = content

        self._content = ""
        self._attachments = []
        self._expanded = False
        self._collapse_wrap = None
        self._content_height = None

        # 编辑
        self._original_content = ""
        self._edit_widget = None
        self._is_editing = False

        self.setMouseTracking(True)
        self._build_ui(role, content, timestamp)

    def _build_action_bar(self) -> BottomActionBar:
        raise NotImplementedError(
            f"{type(self).__name__} 必须实现 _build_action_bar()"
        )

    def _build_meta_row(self, timestamp: str) -> QHBoxLayout:
        raise NotImplementedError(
            f"{type(self).__name__} 必须实现 _build_meta_row()"
        )

    def _build_bubble_box(self, content) -> QFrame:
        raise NotImplementedError(
            f"{type(self).__name__} 必须实现 _build_meta_row()"
        )

    def _pre_content_hook(self, layout: QVBoxLayout):
        pass

    def _build_ui(self, role: str, content, timestamp: str):
        self._bubble_wrap = self._create_bubble_wrap(role, content, timestamp)
        outer = QHBoxLayout(self)
        outer.setContentsMargins(20, 0, -20, 0)
        outer.setSpacing(0)
        if self.is_user:
            outer.addStretch()
            outer.addWidget(self._bubble_wrap)
        else:
            outer.addWidget(self._bubble_wrap, Qt.AlignmentFlag.AlignHCenter)

    def _create_bubble_wrap(self, role: str, content, timestamp: str) -> QFrame:
        wrap = QFrame()
        wrap.setObjectName("bubble_wrap")
        wrap.setContentsMargins(0, 0, 0, 0)
        wrap.setMinimumWidth(self.BUBBLE_MIN_WIDTH)
        wrap.setMaximumWidth(self.BUBBLE_MAX_WIDTH)
        wrap.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

        layout = QVBoxLayout(wrap)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        self.bubble_meta = self._build_meta_row(timestamp)
        layout.addLayout(self.bubble_meta)

        self._pre_content_hook(layout)

        self._bubble_box = self._build_bubble_box(content)
        self._collapse_wrap = CollapseContainer(self._bubble_box)
        layout.addWidget(self._collapse_wrap)

        self.bottom_bar = self._build_action_bar()
        layout.addWidget(self.bottom_bar)
        return wrap

    def calc_bubble_box_height(self):
        self._content_height =  self._bubble_box.height() + 30

    def showEvent(self, event):
        super().showEvent(event)
        self.calc_bubble_box_height()
        self._check_collapse()

    def _check_collapse(self):
        if self._bubble_box.sizeHint().height() <= self.COLLAPSE_MAX_H:
            return

        self._collapse_wrap.setMaximumHeight(self.COLLAPSE_MAX_H)
        self._collapse_wrap.show_mask(True)
        self._collapse_wrap._mask.expanded.connect(self._toggle_expand)

    def _toggle_expand(self):
        if not self._expanded:
            full_h = self._bubble_box.sizeHint().height()
            self._collapse_wrap.setMaximumHeight(16_777_215)
            self._collapse_wrap.animate_height(full_h)
            self._collapse_wrap.show_mask(True)
            self._collapse_wrap._mask.set_expanded(True)
            self._expanded = True
        else:
            self._collapse_wrap.animate_height(self.COLLAPSE_MAX_H)
            self._collapse_wrap._anim.finished.connect(
                lambda: self._collapse_wrap.setMaximumHeight(self.COLLAPSE_MAX_H),
                Qt.ConnectionType.SingleShotConnection
            )
            self._collapse_wrap.show_mask(True)
            self._collapse_wrap._mask.set_expanded(False)
            self._expanded = False

    def enterEvent(self, e):
        self.bottom_bar.show_bar()
        super().enterEvent(e)

    def leaveEvent(self, e):
        self.bottom_bar.hide_bar()
        super().leaveEvent(e)


class UserMessageBubble(MessageBubble):
    def __init__(self, content, timestamp: str, message_id: str, parent=None):
        super().__init__("user", content, timestamp, message_id, parent)
        self.setObjectName("user_bubble")
        self._bubble_box.setObjectName("user_bubble_box")

    def _build_action_bar(self):
        bar = UserBottomActionBar(self)
        bar.retry_clicked.connect(self._on_retry_requested)
        bar.edit_clicked.connect(self._on_start_editing)
        return bar

    def _build_meta_row(self, timestamp: str) -> QHBoxLayout:
        meta      = QHBoxLayout()
        meta.setContentsMargins(2, 0, 2, 0)

        role_label = QLabel("User")
        role_label.setObjectName("bubble_role_label")
        time_label = QLabel(timestamp)
        time_label.setObjectName("bubble_time_label")

        meta.addStretch()
        meta.addWidget(time_label)
        meta.addSpacing(6)
        meta.addWidget(role_label)
        return meta

    def _build_bubble_box(self, content) -> QFrame:
        _bubble_box = QFrame()
        _bubble_box.setContentsMargins(0, 0, 0, 0)
        _bubble_box.setObjectName("bubble_box")

        self._box_layout = QVBoxLayout(_bubble_box)
        self._box_layout.setContentsMargins(0, 0, 0, 0)
        self._box_layout.setSpacing(0)

        if isinstance(content, dict):
            self._content        = content.get("content", "")
            self._attachments    = content.get("attachments", None)
        else:
            self._content        = str(content)
            self._attachments    = None

        self.builder = ContentBuilder(self._box_layout)
        self.builder.build(self._content, self._attachments)
        return _bubble_box

    def _on_retry_requested(self):
        self.retry_requested.emit(self.message_id)

    def _on_start_editing(self):
        if self._is_editing:
            return

        self._is_editing = True
        self._original_content = self._content
        content_height = self._bubble_box.height()

        # 1. 移除原来的内容 widget
        layout = self._bubble_box.layout()
        while layout.count() > 0:
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)  # 先断开父子关系
                widget.deleteLater()

        # 2. 创建编辑器
        self._edit_widget = QTextEdit()
        self._edit_widget.setPlainText(self._content)  # 用纯文本编辑 Markdown 更友好
        # self._edit_widget.setMinimumHeight(120)

        # 美化编辑器
        self._edit_widget.setStyleSheet("""
            QTextEdit {
                background: #1e1e1e;
                color: #dcdcdc;
                border: 1px solid #3c3c3c;
                border-radius: 6px;
                padding: 8px;
                font-size: 14px;
                line-height: 1.4;
            }
        """)

        self._bubble_box.layout().insertWidget(0, self._edit_widget)

        self._bubble_box.setFixedHeight(content_height + 20)
        self._switch_to_edit_bar()

    def _switch_to_edit_bar(self):
        self.bottom_bar.hide()

        edit_bar = QWidget()
        edit_bar.setFixedHeight(48)
        edit_bar.setContentsMargins(0, 0, 0, 0)
        layout = QHBoxLayout(edit_bar)
        layout.setContentsMargins(0, 0, 0, 0)

        btn_save = QPushButton("保存")
        btn_cancel = QPushButton("取消")
        btn_save.setFixedHeight(32)
        btn_cancel.setFixedHeight(32)

        btn_save.clicked.connect(self._save_edit)
        btn_cancel.clicked.connect(self._cancel_edit)

        layout.addStretch()
        layout.addWidget(btn_cancel)
        layout.addWidget(btn_save)

        self._temp_edit_bar = edit_bar
        self._bubble_wrap.layout().addWidget(edit_bar)

    def _save_edit(self):
        if not self._edit_widget:
            return

        new_content = self._edit_widget.toPlainText().strip()
        if new_content == self._original_content:
            self._cancel_edit()
            return
        self._content = new_content

        self._exit_editing_mode()
        self._re_render_content()
        self.content_edited.emit(self.message_id, new_content)

    def _cancel_edit(self):
        self._exit_editing_mode()
        self._re_render_content()

    def _exit_editing_mode(self):
        if self._temp_edit_bar:
            self._temp_edit_bar.setParent(None)
            self._temp_edit_bar.deleteLater()
            self._temp_edit_bar = None

        if self._edit_widget:
            self._edit_widget.setParent(None)
            self._edit_widget.deleteLater()
            self._edit_widget = None

        self.bottom_bar.show()
        self._is_editing = False

    def _re_render_content(self):
        layout = self._bubble_box.layout()
        while layout.count() > 0:
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()

        self.builder.build(self._content, self._attachments)



class SpinnerWidget(QWidget):
    """ Claude 风格旋转加载动画组件。"""
    def __init__(
        self,
        parent=None,
        size: int = 48,
        ring_width: int = 3,
        color: str = "#a0a0a0",
        speed: int = 6,
        fps: int = 60,
        fade_duration: int = 250,
    ):
        super().__init__(parent)

        self._size         = size
        self._ring_width   = ring_width
        self._color        = QColor(color)
        self._speed        = speed
        self._angle        = 0          # 当前旋转角度
        self._arc_len      = 260        # 弧段长度（度）
        self._fade_dur     = fade_duration

        self.setFixedSize(size, size)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        # self.setVisible(False)

        # ── 旋转定时器 ────────────────────────────────────────────────────────
        self._timer = QTimer(self)
        self._timer.setInterval(1000 // fps)
        self._timer.timeout.connect(self._tick)

        # ── 透明度特效 + 动画 ─────────────────────────────────────────────────
        self._opacity_effect = QGraphicsOpacityEffect(self)
        self._opacity_effect.setOpacity(0.0)
        self.setGraphicsEffect(self._opacity_effect)

        self._fade_anim = QPropertyAnimation(self._opacity_effect, b"opacity", self)
        self._fade_anim.setEasingCurve(QEasingCurve.Type.InOutSine)
        self._fade_anim.setDuration(self._fade_dur)
        self._fade_anim.finished.connect(self._on_fade_finished)

    def show_spinner(self):
        """渐显并开始旋转。"""
        self._fade_anim.stop()
        self.setVisible(True)
        self._timer.start()
        self._fade_anim.setStartValue(self._opacity_effect.opacity())
        self._fade_anim.setEndValue(1.0)
        self._fade_anim.start()

    def hide_spinner(self):
        """渐隐，动画结束后自动隐藏并停止旋转。"""
        self._fade_anim.stop()
        self._fade_anim.setStartValue(self._opacity_effect.opacity())
        self._fade_anim.setEndValue(0.0)
        self._fade_anim.start()

    def set_color(self, color: str):
        self._color = QColor(color)
        self.update()

    def set_speed(self, speed: int):
        """每帧旋转角度，范围建议 2-15。"""
        self._speed = speed

    def set_ring_width(self, width: int):
        self._ring_width = width
        self.update()

    def _tick(self):
        self._angle = (self._angle + self._speed) % 360
        self.update()

    def _on_fade_finished(self):
        """渐隐完成后隐藏组件并停止定时器。"""
        if self._opacity_effect.opacity() == 0.0:
            self._timer.stop()
            self.setVisible(False)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        side   = min(self.width(), self.height())
        margin = self._ring_width + 2
        rect_size = side - margin * 2

        painter.translate(self.width() / 2, self.height() / 2)

        # ── 轨道（背景圆环）──────────────────────────────────────────────────
        track_color = QColor(self._color)
        track_color.setAlphaF(0.15)
        pen = QPen(track_color, self._ring_width, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        painter.drawEllipse(
            -rect_size // 2, -rect_size // 2,
            rect_size, rect_size
        )

        # ── 渐变弧段 ──────────────────────────────────────────────────────────
        # 用锥形渐变模拟弧段头尾的透明度渐变，使动画更流畅自然
        gradient = QConicalGradient(0, 0, -self._angle)

        head_color = QColor(self._color)
        head_color.setAlphaF(1.0)

        tail_color = QColor(self._color)
        tail_color.setAlphaF(0.0)

        mid_color = QColor(self._color)
        mid_color.setAlphaF(0.6)

        arc_fraction = self._arc_len / 360.0
        gradient.setColorAt(0.0,             head_color)
        gradient.setColorAt(arc_fraction * 0.5, mid_color)
        gradient.setColorAt(arc_fraction,    tail_color)
        gradient.setColorAt(1.0,             tail_color)

        pen = QPen(gradient, self._ring_width, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)

        painter.drawArc(
            -rect_size // 2, -rect_size // 2,
            rect_size, rect_size,
            (-self._angle + 90) * 16,   # Qt 以 1/16 度为单位，12点钟方向为起点
            -self._arc_len * 16,
        )

        painter.end()

class AssistantMessageBubble(MessageBubble):
    def __init__(self, content, timestamp: str, message_id: str, parent=None):
        super().__init__("assistant", content, timestamp, message_id, parent)
        self.setObjectName("assistant_bubble")
        self._bubble_box.setObjectName("assistant_bubble_box")
        self._bubble_box.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

    def _pre_content_hook(self, layout):
        self.thinking_block = ThinkingBlock()
        self.thinking_block.setVisible(False)
        layout.addWidget(self.thinking_block)

        self.spinner_widget = self.build_spinner()
        layout.addWidget(self.spinner_widget)

    def _build_action_bar(self) -> QWidget:
        bar = AssistantBottomActionBar(self)
        bar.export_clicked.connect(self.on_export_clicked)
        return bar

    def _build_bubble_box(self, content) -> QFrame:
        _bubble_box = QFrame()
        _bubble_box.setContentsMargins(0, 0, 0, 0)
        _bubble_box.setObjectName("bubble_box")

        self._box_layout = QVBoxLayout(_bubble_box)
        self._box_layout.setContentsMargins(0, 0, 0, 0)
        self._box_layout.setSpacing(0)

        self._streaming_renderer = StreamingRenderer(self._box_layout)
        self._content_loader     = ContentLoader(self._box_layout)
        return _bubble_box

    def _build_meta_row(self, timestamp: str) -> QHBoxLayout:
        meta      = QHBoxLayout()
        meta.setContentsMargins(2, 0, 2, 0)

        role_label = QLabel("AI Assistant")
        role_label.setObjectName("bubble_role_label")
        time_label = QLabel(timestamp)
        time_label.setObjectName("bubble_time_label")

        meta.addWidget(role_label)
        meta.addSpacing(6)
        meta.addWidget(time_label)
        meta.addStretch()
        return meta

    def build_spinner(self):
        spinner_container = QWidget()
        spinner_container_layout = QHBoxLayout(spinner_container)
        spinner_container_layout.setContentsMargins(20, 0, 20, 0)
        spinner_container_layout.setSpacing(0)

        spinner = SpinnerWidget(
            parent=spinner_container,
            size=35,  # 尺寸
            ring_width=3,  # 圆弧宽度
            color="#a0a0a0",  # 颜色
            speed=6,  # 旋转速度
            fade_duration=250  # 渐变时长 ms
        )
        spinner.show_spinner()
        spinner_container_layout.addWidget(spinner)
        spinner_container_layout.addStretch()
        spinner_container.setFixedHeight(42)
        return spinner_container

    def on_export_clicked(self):
        if self._attachments:
            attachment = self._attachments[0]
            dir_path = os.path.dirname(attachment)
            QDesktopServices.openUrl(QUrl.fromLocalFile(dir_path))

    def append_thinking(self, chunk: str):
        if not self.thinking_block.isVisible():
            self.thinking_block.setVisible(True)
        self.thinking_block.append_thinking(chunk)

    def switch_to_generating(self):
        self.thinking_block.switch_to_generating()

    def append_output(self, chunk: str):
        self._content += chunk
        self._streaming_renderer.append_chunk(chunk)

    def load_attachments(self, attachments: list[str]) -> None:
        """静态媒体通道入口，收到附件数据时调用。"""
        self._attachments = attachments
        self._content_loader.load(attachments)

    def finish(self, elapsed_ms: int = 0) -> None:
        """ 完成文本、媒体渲染 """
        self.thinking_block.finish(elapsed_ms)
        self._streaming_renderer.finish()
        # 再追加媒体
        if self._attachments:
            self._content_loader.load(self._attachments)

    def hide_think_area(self):
        self.thinking_block.hide_think_area()

def create_message_bubble(role: str, content, timestamp: str, message_id: str) -> MessageBubble:
    if role == "user":
        return UserMessageBubble(content, timestamp, message_id)
    else:
        return AssistantMessageBubble(content, timestamp, message_id)