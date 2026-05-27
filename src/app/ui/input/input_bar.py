from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from PySide6.QtCore import Signal, Qt, QRect, QTimer, QPropertyAnimation, QEasingCurve, QSize
from PySide6.QtGui import (
    QDragEnterEvent, QDropEvent, QKeyEvent, QPixmap,
    QPainter, QPainterPath, QIcon, QColor
)
from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout,
    QComboBox, QPushButton, QLabel, QFileDialog,
    QScrollArea, QApplication, QTextEdit
)

from ui.base.action_button import ActionButton
from ui.param.param_factory import WidgetFactory
from ui.base.widget import BaseWidget
from ui.input.slide_stack import SlideStackWidget, DotIndicatorBar
from ui.param.param_drawer import ParamDrawer
from ui.window_data import WindowData
from src.shared.enum_type import FactoryType
from src.resources import *


@dataclass
class Attachment:
    path: Path
    is_image: bool = False
    thumb: Optional[QPixmap] = field(default=None, repr=False)

    IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif'}

    @classmethod
    def from_path(cls, path: Path) -> "Attachment":
        is_img = path.suffix.lower() in cls.IMAGE_EXTS
        thumb = None
        if is_img:
            pix = QPixmap(str(path))
            if not pix.isNull():
                thumb = pix.scaled(56, 56,
                                   Qt.KeepAspectRatioByExpanding,
                                   Qt.SmoothTransformation)
        return cls(path=path, is_image=is_img, thumb=thumb)


@dataclass
class InputPayload:
    mode: str           # "text" | "image" | "anim" | "model"
    prompt: str
    params: dict[str, Any]
    attachments: list[Attachment]


class AttachmentChip(BaseWidget):
    remove_requested = Signal(object)   # 发送自身

    _EXT_COLORS = {
        '.txt':  ('#E6F1FB', '#0C447C'),
        '.csv':  ('#EAF3DE', '#27500A'),
        '.json': ('#FAEEDA', '#633806'),
        '.py':   ('#EEEDFE', '#3C3489'),
    }

    def __init__(self, attachment: Attachment, parent=None):
        super().__init__(parent)
        self.setObjectName("attachment_chip")
        self.attachment = attachment
        self._build()

    def _build(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(5)
        self.setFixedHeight(WindowData.AttachmentHeight)
        self.setMaximumWidth(WindowData.AttachmentMaxWidth)

        # 缩略图 / 文件类型徽章
        thumb_lbl = QLabel()
        thumb_lbl.setFixedSize(WindowData.AttachmentIconSize)
        thumb_lbl.setAlignment(Qt.AlignCenter)

        if self.attachment.is_image and self.attachment.thumb:
            # 圆角裁剪
            thumb_lbl.setPixmap(self._rounded(self.attachment.thumb, 4))
        else:
            ext = self.attachment.path.suffix.lower()
            bg, fg = self._EXT_COLORS.get(ext, ('#F1EFE8', '#5F5E5A'))
            thumb_lbl.setText(ext.lstrip('.').upper()[:3])
            thumb_lbl.setStyleSheet(
                f"background:{bg};color:{fg};border-radius:4px;"
                f"font-size:9px;font-weight:600;"
            )

        # 文件名
        name_lbl = QLabel(self.attachment.path.name)
        name_lbl.setObjectName("chip_name")
        name_lbl.setMaximumWidth(WindowData.AttachmentLabelMaxWidth)
        # 溢出省略
        fm = name_lbl.fontMetrics()
        elided = fm.elidedText(self.attachment.path.name, Qt.ElideMiddle, WindowData.AttachmentLabelMaxWidth)
        name_lbl.setText(elided)
        name_lbl.setToolTip(str(self.attachment.path))

        # 删除按钮
        rm_btn = QPushButton("×")
        rm_btn.setFixedSize(16, 16)
        rm_btn.setObjectName("chip_remove")
        rm_btn.clicked.connect(lambda: self.remove_requested.emit(self))

        layout.addWidget(thumb_lbl)
        layout.addWidget(name_lbl, 1)
        layout.addWidget(rm_btn)

    @staticmethod
    def _rounded(pix: QPixmap, radius: int) -> QPixmap:
        size = pix.size()
        out = QPixmap(size)
        out.fill(Qt.transparent)
        painter = QPainter(out)
        painter.setRenderHint(QPainter.Antialiasing)
        path = QPainterPath()
        path.addRoundedRect(0, 0, size.width(), size.height(), radius, radius)
        painter.setClipPath(path)
        painter.drawPixmap(0, 0, pix)
        painter.end()
        return out


class AttachmentBar(QScrollArea):
    """横向滚动的 Chip 容器，附件为空时隐藏。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(52)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setWidgetResizable(True)
        self.setFrameShape(QScrollArea.NoFrame)
        self.setVisible(False)

        inner = QWidget()
        self._layout = QHBoxLayout(inner)
        self._layout.setContentsMargins(8, 6, 8, 6)
        self._layout.setSpacing(6)
        self._layout.addStretch()
        self.setWidget(inner)

    def add_chip(self, chip: AttachmentChip):
        self._layout.insertWidget(self._layout.count() - 1, chip)
        self.setVisible(True)

    def remove_chip(self, chip: AttachmentChip):
        self._layout.removeWidget(chip)
        chip.deleteLater()
        if self._layout.count() <= 1:   # 只剩 stretch
            self.setVisible(False)


class InputBar(BaseWidget):
    """ 统一输入栏 """
    submitted = Signal(object)   # InputPayload
    update_height = Signal(object)
    update_model = Signal(object)

    _MODE_CONFIGS = {
        "text": {
            "label": "🗒️ 文本",
            "placeholder": "输入内容主题，例如：写一段游戏角色背景故事",
            "accept_files": True,
            "file_filter": "Text (*.txt *.md *.csv *.json)",
        },
        "image": {
            "label": "📸 图片",
            "placeholder": "提示词 {cfg=7.5} {steps=30} {seed=42} {width=1024}",
            "accept_files": True,
            "file_filter": "Images (*.png *.jpg *.jpeg *.webp)",
        },
        "animation": {
            "label": "🎞️ 动画",
            "placeholder": "side view warrior {frames=16} {fps=8} {motion=3.0}",
            "accept_files": True,
            "file_filter": "Images (*.png *.jpg *.jpeg *.webp)",
        },
        "speech": {
            "label": "🎙️ 语音",
            "placeholder": "输入要朗读的文本，支持中文和英文...",
            "accept_files": True,
            "file_filter": "Text (*.txt *.md)",
            "options": {
                "voice_mode": {
                    "label": "语音模式",
                    "type": "select",
                    "default": "design",
                    "choices": ["design", "tts", "clone"]
                },
                "speaker": {
                    "label": "说话人风格",
                    "type": "select",
                    "default": "default",
                    "choices": ["default", "warm_male", "gentle_female", "deep_narrator", "energetic_youth",
                                "calm_elder"]
                },
                "speed": {
                    "label": "语速",
                    "type": "slider",
                    "default": 1.0,
                    "min": 0.7,
                    "max": 1.5,
                    "step": 0.05
                },
                "emotion": {
                    "label": "情感风格",
                    "type": "select",
                    "default": "neutral",
                    "choices": ["neutral", "happy", "sad", "angry", "excited", "calm", "serious"]
                }
            }
        }
    }

    _LABEL_TO_KEY = {str(v["label"]): k for k, v in _MODE_CONFIGS.items()}


    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("input_bar")
        self.setAcceptDrops(True)
        self._attachments: list[Attachment] = []
        self._chips: list[AttachmentChip] = []
        self._build_ui()
        self._connect()
        self._on_mode_changed()

        self._resize_timer = QTimer(self)
        self._resize_timer.setSingleShot(True)
        self._resize_timer.setInterval(16)
        self._resize_timer.timeout.connect(self.on_input_bar_height_changed)

        self._geo_anim = QPropertyAnimation(self, b"height")
        self._geo_anim.setDuration(16)
        self._geo_anim.setEasingCurve(QEasingCurve.Type.InCubic)

        self._initial_geometry_set = False

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self.stack = SlideStackWidget()
        self.dots = DotIndicatorBar()

        self.input_page = self.build_input_page()
        self.param_page = self.build_param_page()

        self.stack.add_page(self.input_page, "input_page")
        self.stack.add_page(self.param_page, "params_page")
        self.dots.set_page_count(self.stack.count())

        # 双向绑定
        self.stack.current_changed.connect(self.dots.set_current_index)
        self.stack.current_changed.connect(lambda: self._schedule_resize())
        self.stack.drag_progress.connect(self.dots.update_drag)
        self.dots.switch_to.connect(self.stack.set_current_index)

        root.addWidget(self.dots)
        root.addWidget(self.stack)

    def build_input_page(self):
        input_widget = QWidget()
        root = QVBoxLayout(input_widget)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── 顶部主行 ──
        top_row = QWidget()
        top_layout = QHBoxLayout(top_row)
        top_layout.setContentsMargins(20, 10, -20, -10)
        top_layout.setSpacing(10)

        self.mode_combo = QComboBox()
        self.mode_combo.setObjectName("mode_combo")
        self.mode_combo.setFixedHeight(44)
        self.mode_combo.setFixedWidth(100)

        for cfg in self._MODE_CONFIGS.values():
            self.mode_combo.addItem(cfg["label"])

        self.prompt_input = QTextEdit()
        self.prompt_input.setAcceptRichText(False)
        self.prompt_input.setFixedHeight(44)
        self.prompt_input.setObjectName("prompt_input")

        # 工具栏（上传 + 粘贴）
        toolbar = QWidget()
        toolbar.setObjectName("tool_bar")
        tbar_layout = QHBoxLayout(toolbar)
        tbar_layout.setContentsMargins(4, 4, 4, 4)
        tbar_layout.setSpacing(2)

        self.upload_btn = self._icon_button(":svg/upload.svg", self.tr("upload files / pic"))
        self.upload_btn.setObjectName("upload_btn")
        self.paste_btn  = self._icon_button(":svg/paste.svg", self.tr("paste"))
        self.paste_btn.setObjectName("paste_btn")

        tbar_layout.addWidget(self.upload_btn)
        tbar_layout.addWidget(self.paste_btn)

        self.send_btn = ActionButton(":svg/up.svg", self.tr("send"), width=40, height=40)
        self.send_btn.set_color(QColor(120, 106, 75, 30), QColor(200, 106, 75, 255))
        self.send_btn.setObjectName("send_btn")
        self.send_btn.setEnabled(False)

        top_layout.addWidget(self.mode_combo)
        top_layout.addWidget(self.prompt_input, 1)
        top_layout.addWidget(toolbar)
        top_layout.addWidget(self.send_btn)

        self.attach_bar = AttachmentBar()
        self.attach_bar.setObjectName("attachment_bar")

        root.addWidget(top_row)
        root.addWidget(self.attach_bar)

        return input_widget

    def build_param_page(self):
        self.param_drawer = ParamDrawer()
        return self.param_drawer

    @staticmethod
    def _icon_button(svg_path: str, tip: str) -> QPushButton:
        btn = QPushButton()
        btn.setFixedSize(30, 30)
        btn.setToolTip(tip)
        btn.setIcon(QIcon(svg_path))
        btn.setIconSize(QSize(20, 20))
        btn.setObjectName("icon_btn")
        return btn

    def get_textedit_content_height(self) -> int:
        """
        计算 QTextEdit 内容实际高度
        """
        doc = self.prompt_input.document()
        layout = doc.documentLayout()
        # 强制更新布局
        doc.setTextWidth(self.prompt_input.viewport().width())
        content_height = int(layout.documentSize().height())
        # 加上上下边距
        margins = self.prompt_input.contentsMargins()
        frame_margin = int(doc.documentMargin()) * 2
        total_height = content_height + margins.top() + margins.bottom() + frame_margin + 20
        return total_height

    def _connect(self):
        self.mode_combo.currentTextChanged.connect(self._on_mode_changed)
        self.prompt_input.document().contentsChanged.connect(self._schedule_resize)
        self.send_btn.clicked.connect(self.submit)
        self.upload_btn.clicked.connect(self._pick_files)
        self.paste_btn.clicked.connect(self._paste_clipboard)

    # ══════════════════ 模式切换 ══════════════════
    def _on_mode_changed(self):
        key = self._LABEL_TO_KEY.get(self.mode_combo.currentText(), "text")
        cfg = self._MODE_CONFIGS[key]
        self.prompt_input.setPlaceholderText(cfg["placeholder"])

    def _schedule_resize(self):
        self._resize_timer.start()

    def calc_input_height(self):
        width = self.window().width()
        height = self.window().height()

        width_margin = 20
        height_margin = 25

        text_content_height = self.get_textedit_content_height()
        attachment_height = self.attach_bar.height()
        input_bar_width = width - width_margin * 2 - WindowData.SettingWidth
        input_bar_height = min(WindowData.InputBarMaxHeight,
                               attachment_height + text_content_height)

        text_context_margin = 30
        text_edit_height = input_bar_height - text_context_margin
        self.prompt_input.setFixedHeight(text_edit_height)

        input_bar_x = width_margin
        input_bar_y = height - input_bar_height - height_margin
        input_geometry = QRect(input_bar_x,
                              input_bar_y,
                              input_bar_width,
                              input_bar_height)
        return input_geometry.height()

    def calc_param_height(self):
        widget = self.param_page.param_widget
        if widget:
            height = widget.height() + 40
            return min(height, WindowData.InputBarMaxHeight)
        return self.calc_input_height()

    def on_input_bar_height_changed(self):
        if self.stack.current_index() == 0:
            input_height = self.calc_input_height()
        else:
            input_height = self.calc_param_height()
        if input_height == self.height():
            return

        # 如果動畫正在跑且目標相同，不重複觸發
        if self._geo_anim.state() == QPropertyAnimation.Running:
            if self._geo_anim.endValue() == input_height:
                return
            self._geo_anim.stop()

        self._geo_anim.setStartValue(self.height())
        self._geo_anim.setEndValue(input_height)
        self._geo_anim.start()
        self.raise_()

        self.param_drawer.setFixedHeight(input_height)
        self.stack.adjust_page_sizes(input_height)
        self.input_page.setFixedHeight(input_height)
        self.update_height.emit(input_height)

    # ══════════════════ prompt 实时解析 ══════════════════
    def set_model(self, name):
        """切换模型时自动加载对应参数"""
        type_str = self.label_to_key.get(self.mode_combo.currentText(), "")
        type_enum = FactoryType.convert_by_text(type_str)
        widget = WidgetFactory.build_widget(type_enum, name)
        self.param_drawer.load_schema(widget)

    # ══════════════════ 文件选择 ══════════════════
    def _pick_files(self):
        key = self._LABEL_TO_KEY.get(self.mode_combo.currentText(), "text")
        flt = self._MODE_CONFIGS[key]["file_filter"]
        paths, _ = QFileDialog.getOpenFileNames(self, "选择文件", "", flt)
        for p in paths:
            self._add_attachment(Path(p))

    # ══════════════════ 粘贴剪贴板 ══════════════════
    def _paste_clipboard(self):
        cb = QApplication.clipboard()
        mime = cb.mimeData()

        if mime.hasImage():
            pix = cb.pixmap()
            if not pix.isNull():
                # 保存到临时目录
                import tempfile, uuid
                tmp = Path(tempfile.gettempdir()) / f"paste_{uuid.uuid4().hex[:8]}.png"
                pix.save(str(tmp))
                self._add_attachment(tmp)

        elif mime.hasUrls():
            for url in mime.urls():
                p = Path(url.toLocalFile())
                if p.exists():
                    self._add_attachment(p)

        elif mime.hasText():
            # 文本追加到输入框
            cursor = self.prompt_input.textCursor()
            position = cursor.position()
            txt = self.prompt_input.toPlainText()
            new_txt = txt[:position] + mime.text() + txt[position:]
            self.prompt_input.setText(new_txt)

    # ══════════════════ 拖拽 ══════════════════
    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls() or event.mimeData().hasImage():
            event.acceptProposedAction()
            self.setStyleSheet("""
                InputBar {
                    border: 1px solid #3b8bd4;
                    border-radius: 10px;
                    background: #1e2128;
                }
            """)

    def dragLeaveEvent(self, event):
        self._reset_drop_style()

    def dropEvent(self, event: QDropEvent):
        self._reset_drop_style()
        mime = event.mimeData()

        if mime.hasUrls():
            for url in mime.urls():
                p = Path(url.toLocalFile())
                if p.exists() and p.is_file():
                    self._add_attachment(p)
        elif mime.hasImage():
            import tempfile, uuid
            pix = QPixmap()
            pix.loadFromData(mime.data("image/png"))
            if not pix.isNull():
                tmp = Path(tempfile.gettempdir()) / f"drop_{uuid.uuid4().hex[:8]}.png"
                pix.save(str(tmp))
                self._add_attachment(tmp)

        event.acceptProposedAction()

    def _reset_drop_style(self):
        self.setStyleSheet("""
            InputBar {
                border: 0.5px solid rgba(0,0,0,0.15);
                border-radius: 10px;
                background: #1e2128;
            }
        """)

    def _add_attachment(self, path: Path):
        # 去重
        if any(a.path == path for a in self._attachments):
            return

        att = Attachment.from_path(path)
        self._attachments.append(att)

        chip = AttachmentChip(att)
        chip.remove_requested.connect(self._remove_chip)
        self._chips.append(chip)
        self.attach_bar.add_chip(chip)

    def _remove_chip(self, chip: AttachmentChip):
        if chip.attachment in self._attachments:
            self._attachments.remove(chip.attachment)
        if chip in self._chips:
            self._chips.remove(chip)
        self.attach_bar.remove_chip(chip)

    def submit(self):
        prompt = self.prompt_input.toPlainText().strip()
        if not prompt:
            return

        key = self._LABEL_TO_KEY.get(self.mode_combo.currentText(), "text")
        params = self.param_drawer.get_params()

        payload = InputPayload(
            mode=key,
            prompt=prompt,
            params=params,
            attachments=list(self._attachments),
        )

        self.submitted.emit(payload)
        self._clear()

    def _clear(self):
        self.prompt_input.clear()
        for chip in list(self._chips):
            self.attach_bar.remove_chip(chip)
        self._chips.clear()
        self._attachments.clear()

    def keyPressEvent(self, event: QKeyEvent):
        # Ctrl+V → 粘贴
        if event.modifiers() == Qt.ControlModifier and event.key() == Qt.Key_V:
            self._paste_clipboard()
        # Escape → 清空附件
        elif event.key() == Qt.Key_Escape:
            for chip in list(self._chips):
                self.attach_bar.remove_chip(chip)
            self._chips.clear()
            self._attachments.clear()
        else:
            super().keyPressEvent(event)

    @property
    def label_to_key(self):
        return self._LABEL_TO_KEY

    def showEvent(self, event):
        super().showEvent(event)
        if not self._initial_geometry_set:
            self._schedule_resize()
            self._initial_geometry_set = True

    def resizeEvent(self, event):
        width = event.size().width()
        self.input_page.setFixedWidth(width)
        self.param_page.setFixedWidth(width)
        super().resizeEvent(event)