from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from PySide6.QtCore import (
    Signal, Qt, QEvent, QRect, QTimer, QPropertyAnimation, QEasingCurve, QSize, Property, QPoint,
    QT_TRANSLATE_NOOP,
)
from PySide6.QtGui import (
    QDragEnterEvent, QDropEvent, QKeyEvent, QPen, QPixmap,
    QPainter, QPainterPath, QIcon, QColor
)
from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout,
    QPushButton, QLabel, QFileDialog,
    QScrollArea, QApplication, QTextEdit
)

from src.app.ui.base.action_button import ActionButton
from src.app.ui.param.param_factory import WidgetFactory
from src.app.ui.base.widget import BaseWidget
from src.app.ui.param.param_drawer import ParamDrawer
from src.app.ui.window_data import WindowData
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


_POPOVER_BORDER = QColor("#404048")
_POPOVER_FILL = QColor("#21242c")
_ACCENT = QColor("#7B9DBC")


def _paint_popover_frame(widget: QWidget):
    """和参数悬浮卡同款的圆角描边底板，让两种弹层观感一致。"""
    painter = QPainter(widget)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.setPen(QPen(_POPOVER_BORDER, 1))
    painter.setBrush(_POPOVER_FILL)
    path = QPainterPath()
    path.addRoundedRect(widget.rect().adjusted(1, 1, -1, -1), 8, 8)
    painter.drawPath(path)


class _PopoverRow(QWidget):
    """弹层里的一行。radio=True 是模式项（选中画对勾），否则是带复选框的开关项。"""

    clicked = Signal()
    _HEIGHT = 30

    def __init__(self, text: str, radio: bool, parent=None):
        super().__init__(parent)
        self._text = text
        self._radio = radio
        self._checked = False
        self._hover = False
        self.setFixedHeight(self._HEIGHT)
        self.setCursor(Qt.PointingHandCursor)

    def isChecked(self) -> bool:
        return self._checked

    def setChecked(self, checked: bool):
        self._checked = checked
        self.update()

    def enterEvent(self, event):
        self._hover = True
        self.update()

    def leaveEvent(self, event):
        self._hover = False
        self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.rect().contains(event.position().toPoint()):
            self.clicked.emit()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        if self._hover:
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(123, 157, 188, 40))
            painter.drawRoundedRect(self.rect().adjusted(4, 1, -4, -1), 5, 5)

        mark = QRect(14, (self.height() - 14) // 2, 14, 14)
        tick = QPainterPath()
        tick.moveTo(mark.left() + 3.5, mark.center().y() + 0.5)
        tick.lineTo(mark.left() + 6, mark.bottom() - 3.5)
        tick.lineTo(mark.right() - 3, mark.top() + 4)
        if self._radio:
            if self._checked:
                painter.setPen(QPen(_ACCENT, 1.8, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
                painter.setBrush(Qt.NoBrush)
                painter.drawPath(tick)
        else:
            box = mark.adjusted(0, 0, -1, -1)
            painter.setPen(QPen(_ACCENT if self._checked else QColor("#6a6d78"), 1.2))
            painter.setBrush(_ACCENT if self._checked else Qt.NoBrush)
            painter.drawRoundedRect(box, 3.5, 3.5)
            if self._checked:
                painter.setPen(QPen(_POPOVER_FILL, 1.8, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
                painter.setBrush(Qt.NoBrush)
                painter.drawPath(tick)

        painter.setPen(QColor("#DCDCDC"))
        painter.drawText(self.rect().adjusted(38, 0, -14, 0), Qt.AlignVCenter | Qt.AlignLeft, self._text)

    def sizeHint(self) -> QSize:
        width = self.fontMetrics().horizontalAdvance(self._text) + 38 + 14
        return QSize(max(width, 140), self._HEIGHT)


class _PopoverSeparator(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(9)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setPen(QPen(QColor(255, 255, 255, 25), 1))
        painter.drawLine(12, 4, self.width() - 12, 4)


class _ModePopover(QWidget):
    """向上弹出的模式菜单：外观沿用 ParamPopover，而不是系统 QMenu 的硬边框和粗对勾。"""

    def __init__(self, parent=None):
        super().__init__(parent, Qt.Popup | Qt.FramelessWindowHint | Qt.NoDropShadowWindowHint)
        self.setObjectName("mode_popover")
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.body = QVBoxLayout(self)
        self.body.setContentsMargins(5, 6, 5, 6)
        self.body.setSpacing(0)

    def paintEvent(self, event):
        _paint_popover_frame(self)


class ModeMenuButton(QPushButton):
    """模式选择按钮：点击在按钮上方弹出菜单，菜单里是互斥的模式项，
    其后可再追加带分隔线的可勾选选项（如"优化提示词"）。
    对外保持和 QComboBox 一致的 currentText / setCurrentText / currentTextChanged，
    这样生成页里按模式文字联动模型列表的代码不用改。"""

    currentTextChanged = Signal(str)

    def __init__(self, labels: list[str], parent=None):
        super().__init__(parent)
        self._text = labels[0]
        self._popover = _ModePopover(self)
        self._rows: dict[str, _PopoverRow] = {}
        for label in labels:
            row = _PopoverRow(label, radio=True)
            row.clicked.connect(lambda text=label: self._on_mode_clicked(text))
            self._popover.body.addWidget(row)
            self._rows[label] = row
        self._rows[self._text].setChecked(True)
        # 用图标画箭头：文字符号 ▴ 随字体渲染，大小和粗细都不受控。
        # RightToLeft 让图标排到文字右侧；菜单是子控件会继承方向，要单独改回来
        self.setIcon(QIcon(":svg/spin_up.svg"))
        self.setIconSize(QSize(10, 10))
        self.setLayoutDirection(Qt.RightToLeft)
        self._popover.setLayoutDirection(Qt.LeftToRight)
        self._refresh_caption()
        self.clicked.connect(self._popup_above)

    def add_separator(self) -> QWidget:
        separator = _PopoverSeparator()
        self._popover.body.addWidget(separator)
        return separator

    def add_option(self, text: str, tooltip: str) -> _PopoverRow:
        row = _PopoverRow(text, radio=False)
        row.setToolTip(tooltip)
        # 开关项点一下只翻转勾选，不收起菜单，方便连着勾几个
        row.clicked.connect(lambda: row.setChecked(not row.isChecked()))
        self._popover.body.addWidget(row)
        return row

    def currentText(self) -> str:
        return self._text

    def setCurrentText(self, text: str):
        if text not in self._rows or text == self._text:
            return
        self._rows[self._text].setChecked(False)
        self._text = text
        self._rows[text].setChecked(True)
        self._refresh_caption()
        self.currentTextChanged.emit(text)

    def _on_mode_clicked(self, text: str):
        self._popover.hide()
        self.setCurrentText(text)

    def _refresh_caption(self):
        self.setText(self._text)

    def _popup_above(self):
        # 输入栏贴着窗口底部，菜单向上弹才不会被屏幕边缘挤压
        self._popover.adjustSize()
        self._popover.move(self.mapToGlobal(QPoint(0, -self._popover.height() - 6)))
        self._popover.show()


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
        rm_btn = QPushButton()
        rm_btn.setFixedSize(16, 16)
        rm_btn.setObjectName("chip_remove")
        rm_btn.setIcon(QIcon(":svg/close.svg"))
        rm_btn.setIconSize(QSize(12, 12))
        rm_btn.setToolTip(self.tr("remove attachment"))
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
    BAR_HEIGHT = 52

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(self.BAR_HEIGHT)
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


class ParamPopover(QWidget):
    """快捷参数悬浮卡：临时借用 ParamDrawer 当前的参数面板，关闭时归还。"""

    closed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent, Qt.Popup | Qt.FramelessWindowHint | Qt.NoDropShadowWindowHint)
        self.setObjectName("param_popover")
        self.setAttribute(Qt.WA_TranslucentBackground)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(1, 1, 1, 1)
        outer.setSpacing(0)

        self._scroll = QScrollArea()
        self._scroll.setObjectName("param_popover_scroll")
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QScrollArea.NoFrame)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        outer.addWidget(self._scroll)

        self.setFixedWidth(320)
        self.setMaximumHeight(360)

    def show_widget(self, widget: QWidget):
        widget.setParent(self._scroll)
        self._scroll.setWidget(widget)
        widget.show()

    def take_widget(self) -> Optional[QWidget]:
        return self._scroll.takeWidget()

    def hideEvent(self, event):
        super().hideEvent(event)
        self.closed.emit()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        # 繪製一個邊框
        painter.setPen(QPen(QColor("#404048"), 1))
        painter.setBrush(QColor("#21242c"))

        rect = self.rect()
        path = QPainterPath()
        path.addRoundedRect(rect.adjusted(1, 1, -1, -1), 8, 8)
        painter.drawPath(path)

_STOP_ICON_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24">'
    '<rect x="6" y="6" width="12" height="12" rx="2.5" fill="#b0b0b8"/>'
    '</svg>'
)


class InputBar(BaseWidget):
    """ 统一输入栏 """
    submitted = Signal(object)   # InputPayload
    stop_requested = Signal()
    update_height = Signal(object)
    update_model = Signal(object)

    _MODE_CONFIGS = {
        "text": {
            "icon": "🗒️",
            "label": QT_TRANSLATE_NOOP("InputBar", "Text"),
            "placeholder": QT_TRANSLATE_NOOP(
                "InputBar",
                "Enter a topic, such as: write a background story for a game character."),
            "file_filter": "Text (*.txt *.md *.csv *.json)",
        },
        "image": {
            "icon": "📸",
            "label": QT_TRANSLATE_NOOP("InputBar", "Image"),
            "placeholder": QT_TRANSLATE_NOOP("InputBar", "image prompt"),
            "file_filter": "Images (*.png *.jpg *.jpeg *.webp)",
        },
        "animation": {
            "icon": "🎞️",
            "label": QT_TRANSLATE_NOOP("InputBar", "Animation"),
            "placeholder": QT_TRANSLATE_NOOP("InputBar", "side view warrior"),
            "file_filter": "Images (*.png *.jpg *.jpeg *.webp)",
        },
        "speech": {
            "icon": "🎙️",
            "label": QT_TRANSLATE_NOOP("InputBar", "Speech"),
            "placeholder": QT_TRANSLATE_NOOP(
                "InputBar",
                "Enter the text you want to read aloud; Chinese and English are supported..."),
            "file_filter": "Text (*.txt *.md)",
        }
    }

    _REFINABLE_MODES = ("image", "animation")


    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("input_bar")
        self.setAcceptDrops(True)
        self._attachments: list[Attachment] = []
        self._chips: list[AttachmentChip] = []
        # 模式标签既是显示文字又是反查模式的键，而类属性在翻译器装好之前就求值了，
        # 所以按实例翻译后再建表。
        self._mode_labels = {
            key: f"{cfg['icon']} {self.tr(cfg['label'])}"
            for key, cfg in self._MODE_CONFIGS.items()
        }
        self._label_to_key = {label: key for key, label in self._mode_labels.items()}
        self._build_ui()
        self._connect()
        self._on_mode_changed()

        self._resize_timer = QTimer(self)
        self._resize_timer.setSingleShot(True)
        self._resize_timer.setInterval(16)
        self._resize_timer.timeout.connect(self.on_input_bar_height_changed)

        self._geo_anim = QPropertyAnimation(self, b"animatedHeight")
        self._geo_anim.setDuration(16)
        self._geo_anim.setEasingCurve(QEasingCurve.Type.InCubic)

        self._initial_geometry_set = False
        self._param_popover: Optional[ParamPopover] = None

    def _get_animated_height(self) -> int:
        return self.height()

    def _set_animated_height(self, height: int):
        self.setFixedHeight(height)

    animatedHeight = Property(int, _get_animated_height, _set_animated_height)

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self.input_page = self.build_input_page()
        self.param_page = self.build_param_page()
        self.param_page.setParent(self)
        self.param_page.setVisible(False)

        root.addWidget(self.input_page)
        self._fit_mode_button_width()

    def _fit_mode_button_width(self):
        """各语言的模式名长短差很多（如俄语"Изображение"），按最长的一项定宽：
        既不截字，切换模式时输入框也不会跟着左右跳。
        QSS 里的字号挂在 #input_bar 祖先选择器上，所以要等按钮挂进输入栏后再 polish 取字宽。"""
        self.mode_combo.ensurePolished()
        metrics = self.mode_combo.fontMetrics()
        widest = max(metrics.horizontalAdvance(label) for label in self._mode_labels.values())
        icon_width = self.mode_combo.iconSize().width() + 6
        self.mode_combo.setFixedWidth(max(100, widest + icon_width + 24))

    def build_input_page(self):
        input_widget = QWidget()
        root = QVBoxLayout(input_widget)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # top row
        top_row = QWidget()
        top_layout = QHBoxLayout(top_row)
        top_layout.setContentsMargins(20, 10, -20, -10)
        top_layout.setSpacing(10)

        # 属性名沿用 mode_combo：生成页按它取当前模式
        self.mode_combo = ModeMenuButton(list(self._mode_labels.values()))
        self.mode_combo.setObjectName("mode_btn")
        self.mode_combo.setFixedHeight(44)

        # 模式之外的附加选项收进同一个菜单，不占输入栏空间；只对适用的模式显示
        self._option_separator = self.mode_combo.add_separator()
        self.refine_action = self.mode_combo.add_option(
            self.tr("optimize prompt"),
            self.tr("Rewrite the prompt with a local model for better results"))
        self.sprite_action = self.mode_combo.add_option(
            self.tr("export sprite sheet"),
            self.tr("Also export the frames as a sprite sheet with atlas JSON"))

        self.prompt_input = QTextEdit()
        self.prompt_input.setAcceptRichText(False)
        self.prompt_input.setFixedHeight(44)
        self.prompt_input.setObjectName("prompt_input")

        # tool bar
        toolbar = QWidget()
        toolbar.setObjectName("tool_bar")
        tbar_layout = QHBoxLayout(toolbar)
        tbar_layout.setContentsMargins(4, 4, 4, 4)
        tbar_layout.setSpacing(2)

        self.upload_btn = self._icon_button(":svg/upload.svg", self.tr("upload files / pic"))
        self.upload_btn.setObjectName("upload_btn")
        tbar_layout.addWidget(self.upload_btn)

        self.param_quick_btn = self._icon_button(":svg/setting.svg", self.tr("quick params"))
        self.param_quick_btn.setObjectName("param_btn")
        tbar_layout.addWidget(self.param_quick_btn)

        self.send_btn = ActionButton(":svg/up.svg", self.tr("send"), width=40, height=40)
        self.send_btn.set_color(QColor(120, 106, 75, 30), QColor(200, 106, 75, 255))
        self.send_btn.setObjectName("send_btn")
        self.send_btn.setEnabled(False)
        self._generating = False

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
        self.prompt_input.installEventFilter(self)
        self.send_btn.clicked.connect(self._on_send_clicked)
        self.upload_btn.clicked.connect(self._pick_files)
        self.param_quick_btn.clicked.connect(self._toggle_param_popover)

    def eventFilter(self, obj, event):
        if obj is self.prompt_input and event.type() == QEvent.Type.KeyPress:
            if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
                    # QTextEdit 默认不处理 Ctrl+Enter（不会自动换行），手动插入换行符
                    self.prompt_input.insertPlainText("\n")
                elif not self._generating:
                    self.submit()
                return True
        return super().eventFilter(obj, event)

    def _on_mode_changed(self):
        key = self._label_to_key.get(self.mode_combo.currentText(), "text")
        cfg = self._MODE_CONFIGS[key]
        self.prompt_input.setPlaceholderText(self.tr(cfg["placeholder"]))
        self.refine_action.setVisible(key in self._REFINABLE_MODES)
        self.sprite_action.setVisible(key == "animation")
        self._option_separator.setVisible(key in self._REFINABLE_MODES)

    def _schedule_resize(self):
        self._resize_timer.start()

    def calc_input_height(self):
        width = self.window().width()
        height = self.window().height()

        width_margin = 20
        height_margin = 25

        text_content_height = self.get_textedit_content_height()
        # attach_bar 隐藏时 Qt 尚未对其做过布局，height() 可能还是旧值/0，
        # 用是否有附件来判断而不是读取实时控件高度，避免刚显示时高度算少。
        attachment_height = AttachmentBar.BAR_HEIGHT if self._chips else 0
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

    def on_input_bar_height_changed(self):
        input_height = self.calc_input_height()
        if input_height == self.height():
            return

        if self._geo_anim.state() == QPropertyAnimation.Running:
            if self._geo_anim.endValue() == input_height:
                return
            self._geo_anim.stop()

        self._geo_anim.setStartValue(self.height())
        self._geo_anim.setEndValue(input_height)
        self._geo_anim.start()
        self.raise_()

        self.input_page.setFixedHeight(input_height)
        self.update_height.emit(input_height)

    def set_model(self, name):
        """切换模型时自动加载对应参数"""
        # 切模型前如果快捷参数卡还开着，先把旧控件收回 drawer，
        # 否则 load_schema 清空 drawer 时找不到它，旧面板会悬空。
        if self._param_popover is not None and self._param_popover.isVisible():
            self._param_popover.hide()

        type_str = self.label_to_key.get(self.mode_combo.currentText(), "")
        type_enum = FactoryType.convert_by_text(type_str)
        widget = WidgetFactory.build_widget(type_enum, name)
        self.param_drawer.load_schema(widget)

    def _toggle_param_popover(self):
        if self._param_popover is not None and self._param_popover.isVisible():
            self._param_popover.hide()
            return

        widget = self.param_drawer.param_widget
        if widget is None:
            return

        if self._param_popover is None:
            self._param_popover = ParamPopover(self)
            self._param_popover.closed.connect(self._on_param_popover_closed)

        self._param_popover.show_widget(widget)
        self._param_popover.adjustSize()

        # 卡片右边缘贴着按钮右边缘，底边贴在按钮上方（向上弹出），
        # 输入栏本来就在窗口底部，向下弹会被截断。
        gap = 8
        anchor = self.param_quick_btn.mapToGlobal(self.param_quick_btn.rect().topRight())
        x = anchor.x() - self._param_popover.width()
        y = anchor.y() - self._param_popover.height() - gap
        self._param_popover.move(x, y)
        self._param_popover.show()

    def _on_param_popover_closed(self):
        widget = self._param_popover.take_widget()
        if widget is not None:
            self.param_drawer.load_schema(widget)
        self.param_drawer.save_params()

    def _pick_files(self):
        key = self._label_to_key.get(self.mode_combo.currentText(), "text")
        flt = self._MODE_CONFIGS[key]["file_filter"]
        paths, _ = QFileDialog.getOpenFileNames(self, "Choose File", "", flt)
        for p in paths:
            self._add_attachment(Path(p))

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
        self._schedule_resize()

    def _remove_chip(self, chip: AttachmentChip):
        if chip.attachment in self._attachments:
            self._attachments.remove(chip.attachment)
        if chip in self._chips:
            self._chips.remove(chip)
        self.attach_bar.remove_chip(chip)
        self._schedule_resize()

    def _on_send_clicked(self):
        if self._generating:
            self.stop_requested.emit()
        else:
            self.submit()

    def set_generating(self, generating: bool):
        """生成进行中时，同一个按钮切换成"停止"，再点一次就中断当前生成。"""
        if generating == self._generating:
            return
        self._generating = generating
        if generating:
            self.send_btn.set_icon(_STOP_ICON_SVG)
            self.send_btn.setToolTip(self.tr("stop"))
            self.send_btn.setEnabled(True)
        else:
            self.send_btn.set_icon(":svg/up.svg")
            self.send_btn.setToolTip(self.tr("send"))
            self.send_btn.setEnabled(bool(self.prompt_input.toPlainText().strip()))

    @property
    def is_generating(self) -> bool:
        return self._generating

    def set_busy(self, busy: bool):
        """生成期间锁住除 send_btn 之外的输入控件（上传/粘贴/模式切换/文本框），
        send_btn 本身不禁用，而是切换成"停止"外观，交给 set_generating 处理。"""
        self.prompt_input.setEnabled(not busy)
        self.mode_combo.setEnabled(not busy)
        self.upload_btn.setEnabled(not busy)
        self.set_generating(busy)

    def submit(self):
        if self._generating:
            return
        prompt = self.prompt_input.toPlainText().strip()
        if not prompt:
            return

        key = self._label_to_key.get(self.mode_combo.currentText(), "text")
        self.param_drawer.save_params()
        params = self.param_drawer.get_params()
        if key in self._REFINABLE_MODES:
            params["refine_prompt"] = self.refine_action.isChecked()
        if key == "animation":
            params["export_sprites"] = self.sprite_action.isChecked()

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
            self._schedule_resize()
        else:
            super().keyPressEvent(event)

    @property
    def label_to_key(self):
        return self._label_to_key

    def showEvent(self, event):
        super().showEvent(event)
        if not self._initial_geometry_set:
            self._schedule_resize()
            self._initial_geometry_set = True

    def resizeEvent(self, event):
        width = event.size().width()
        self.input_page.setFixedWidth(width)
        super().resizeEvent(event)