from datetime import datetime

from PySide6.QtCore import Signal, Qt, QAbstractListModel, QModelIndex, QSize, QPoint, QRect, QObject, QTimer, QVariantAnimation, QEasingCurve
from PySide6.QtGui import QIcon, QBrush, QPainter, QPen, QColor
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                               QFrame, QScrollBar, QListView, QStyledItemDelegate, QAbstractItemView, QMenu, QStyle,
                               QSplitter, QMessageBox, QPushButton, QSizePolicy)

from src.app.work import ClearMemoryWorker, ModelListWorker
from src.app.ui.input.input_bar import InputBar, InputPayload
from src.app.ui.model_comb import ModelComboBox
from src.app.ui.base.action_button import ActionButton
from src.app.ui.chat.chat_session_manager import ChatSessionManager
from src.app.ui.chat.chat_widget import ChatWidget
from src.app.ui.window_data import WindowData
from src.shared.enum_type import FactoryType

def init_widget():
    from src.app.ui.param.core.text.llama_chat_panel import LlamaChatPanel
    from src.app.ui.param.core.image.flux_schnell_panel import FluxSchnellPanel
    from src.app.ui.param.core.image.sdxl_panel import SDXLPanel
    from src.app.ui.param.core.image.text2image_panels import (
        SD35MediumPanel, ZImageTurboPanel, QwenImageLightningPanel)
    from src.app.ui.param.core.image.image_edit_panel import BgRemovalPanel, UpscalePanel
    from src.app.ui.param.core.image_frame.film_interpolation_panel import FilmInterpolationPanel
    from src.app.ui.param.core.animation.ltx_video_panel import LTXVideoPanel
    from src.app.ui.param.core.animation.ltx2_video_panel import LTX2VideoPanel
    from src.app.ui.param.core.animation.wan2_video_panel import Wan2VideoPanel
    from src.app.ui.param.core.speech.ace_step_panel import AceStepMusicPanel
    from src.app.ui.param.core.speech.qwen3_tts_panel import Qwen3TTSPanel
    from src.app.ui.param.core.speech.stable_audio_open_panel import StableAudioOpenPanel
init_widget()


class HistoryModel(QAbstractListModel):
    """每行是一个 session 字典（session_id / title / created_at / ...），
    显示用 title，但删除/切换等操作都靠 session_id 定位。"""

    def __init__(self, sessions=None, parent=None):
        super().__init__(parent)
        self._data: list = sessions or []

    def rowCount(self, parent=QModelIndex()) -> int:
        return len(self._data)

    def data(self, index: QModelIndex, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        item = self._data[index.row()]
        if role == Qt.DisplayRole:
            return item.get("title") or item.get("session_id", "")
        if role == Qt.UserRole:
            return item
        return None

    def append(self, session: dict):
        row = len(self._data)
        self.beginInsertRows(QModelIndex(), row, row)
        self._data.append(session)
        self.endInsertRows()

    def remove(self, row: int):
        if not (0 <= row < len(self._data)):
            return
        self.beginRemoveRows(QModelIndex(), row, row)
        self._data.pop(row)
        self.endRemoveRows()

    def update_title(self, row: int, title: str):
        if not (0 <= row < len(self._data)):
            return
        self._data[row]["title"] = title
        idx = self.index(row)
        self.dataChanged.emit(idx, idx, [Qt.DisplayRole])

    def get(self, row: int) -> dict:
        return self._data[row]

    def find(self, session_id: str) -> int:
        """按 session_id 返回第一个匹配的行号，未找到返回 -1"""
        for row, item in enumerate(self._data):
            if item.get("session_id") == session_id:
                return row
        return -1

    def reset_all(self, sessions: list):
        self.beginResetModel()
        self._data = list(sessions)
        self.endResetModel()


class HistoryDelegateSignals(QObject):
    delete_requested = Signal(QModelIndex)
    select_requested = Signal(QModelIndex)


class HistoryDelegate(QStyledItemDelegate):
    BTN_SIZE   = 24
    BTN_MARGIN = 12
    BTN_HOVER_COLOR = QColor(70, 70, 70, 100)
    HOVER_COLOR = QColor(255, 255, 255, 14)
    SELECTED_COLOR = QColor(123, 157, 188, 40)   # 与全局强调色 #7B9DBC 呼应
    SELECTED_BAR_COLOR = QColor(123, 157, 188, 220)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.signals = HistoryDelegateSignals()
        self._hovered_index = QModelIndex()
        self._btn_hovered   = False

    def sizeHint(self, option, index) -> QSize:
        return QSize(option.rect.width(), 32)

    def _btn_rect(self, option_rect: QRect) -> QRect:
        r = option_rect
        x = r.right() - self.BTN_MARGIN - self.BTN_SIZE
        y = r.top()   + (r.height() - self.BTN_SIZE) // 2
        return QRect(x, y, self.BTN_SIZE, self.BTN_SIZE)

    def paint(self, painter: QPainter, option, index):
        painter.save()

        palette = option.palette
        bg_rect = option.rect
        is_selected = bool(option.state & QStyle.StateFlag.State_Selected)
        if is_selected:
            painter.fillRect(bg_rect, self.SELECTED_COLOR)
            painter.fillRect(QRect(bg_rect.left(), bg_rect.top(), 3, bg_rect.height()), self.SELECTED_BAR_COLOR)
            text_color = palette.text().color()
        elif index == self._hovered_index:
            painter.fillRect(bg_rect, self.HOVER_COLOR)
            text_color = palette.text().color()
        else:
            text_color = palette.text().color()

        text_rect = QRect(
            option.rect.left() + 12,
            option.rect.top(),
            option.rect.width() - self.BTN_SIZE - self.BTN_MARGIN - 20,
            option.rect.height(),
        )
        painter.setPen(QPen(text_color))
        painter.drawText(
            text_rect,
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
            index.data(Qt.ItemDataRole.DisplayRole) or "",
        )

        if index == self._hovered_index:
            btn = self._btn_rect(option.rect)
            if self._btn_hovered:
                painter.setBrush(self.BTN_HOVER_COLOR)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setRenderHint(QPainter.RenderHint.Antialiasing)
                painter.drawRoundedRect(btn, 4, 4)

            dot_color = text_color
            painter.setBrush(QBrush(dot_color))
            painter.setPen(Qt.PenStyle.NoPen)
            dot_r  = 2
            center_y = btn.center().y()

            for dx in (-6, 0, 6):
                painter.drawEllipse(
                    QPoint(btn.center().x() + dx, center_y),
                    dot_r, dot_r,
                )
        painter.restore()

    def editorEvent(self, event, model, option, index):
        from PySide6.QtCore import QEvent
        btn = self._btn_rect(option.rect)

        if event.type() == QEvent.Type.MouseMove:
            self._btn_hovered = btn.contains(event.pos())
            return False

        if event.type() == QEvent.Type.MouseButtonPress:
            if event.button() == Qt.MouseButton.LeftButton:
                if btn.contains(event.pos()):
                    return True
                else:
                    self.signals.select_requested.emit(index)

        if event.type() == QEvent.Type.MouseButtonRelease:
            if btn.contains(event.pos()):
                self._show_menu(index, option.widget.viewport().mapToGlobal(
                    btn.bottomLeft()
                ))
                return True

        return super().editorEvent(event, model, option, index)

    def _show_menu(self, index: QModelIndex, pos: QPoint):
        self._menu_open = True
        menu = QMenu()
        menu.setObjectName("history_item_menu")

        delete_act = menu.addAction("删除")
        delete_act.setIcon(QIcon.fromTheme("edit-delete"))

        action = menu.exec(pos)
        self._menu_open = False
        if action == delete_act:
            self.signals.delete_requested.emit(index)


class HistoryListView(QListView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)

    def mouseMoveEvent(self, event):
        index = self.indexAt(event.pos())
        delegate = self.itemDelegate()
        if isinstance(delegate, HistoryDelegate):
            old = delegate._hovered_index
            delegate._hovered_index = index

            if index.isValid():
                btn_rect = self.visualRect(index)
                delegate._btn_hovered = btn_rect.contains(event.pos())
            else:
                delegate._btn_hovered = False

            if old.isValid():
                self.viewport().update(self.visualRect(old))
            if index.isValid():
                self.viewport().update(self.visualRect(index))
        super().mouseMoveEvent(event)

    def leaveEvent(self, event):
        delegate = self.itemDelegate()
        if isinstance(delegate, HistoryDelegate):
            if getattr(delegate, '_menu_open', False):
                super().leaveEvent(event)
                return

            old = delegate._hovered_index
            delegate._hovered_index = QModelIndex()
            delegate._btn_hovered   = False
            if old.isValid():
                self.viewport().update(self.visualRect(old))
        super().leaveEvent(event)


class SettingSidePage(QFrame):
    delete_session = Signal(str)
    switch_session = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setObjectName("setting_side_page")
        self.build_ui()

    def build_ui(self):
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(5, 10, 5, 10)
        self.layout.setSpacing(5)

        self.new_session_btn = self.add_button(":/svg/new_session.svg", self.tr("New session"), self.tr("create new session"))
        self.setting_btn = self.add_button(":/svg/setting.svg", self.tr("Setting"), self.tr("set app parameter"))
        self.add_seperator()

        self.history_model = HistoryModel()
        self.history_delegate = HistoryDelegate()

        self.history_list = HistoryListView()
        self.history_list.setModel(self.history_model)
        self.history_list.setItemDelegate(self.history_delegate)
        self.layout.addWidget(self.history_list)
        self.layout.addStretch()

        self.history_delegate.signals.delete_requested.connect(
            self._on_delete_requested
        )
        self.history_delegate.signals.select_requested.connect(
            self._on_selected_requested
        )

    def _on_selected_requested(self, index: QModelIndex):
        session_id = self.history_model.get(index.row())["session_id"]
        self.switch_session.emit(session_id)

    def _on_delete_requested(self, index: QModelIndex):
        session_id = self.history_model.get(index.row())["session_id"]
        self.remove_session(session_id)
        self.delete_session.emit(session_id)

    def add_button(self, svg_path, text, tooltip, is_circle=False):
        btn = ActionButton(
            text=text,
            svg_str=svg_path,
            tooltip=tooltip,
            width=WindowData.SettingButtonWidth,
            height=WindowData.SettingButtonHeight,
            is_circle=is_circle,
        )
        # 和下方历史会话列表的行一致：撑满整行，配色也对齐列表项的 hover/按下效果。
        btn.setMinimumWidth(0)
        btn.setMaximumWidth(16777215)
        btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        btn.set_color(QColor(255, 255, 255, 0), QColor(255, 255, 255, 14))
        btn._press_bg = QColor(255, 255, 255, 24)
        self.layout.addWidget(btn)
        return btn

    def add_circle_button(self, svg_path, text, tooltip, is_circle=False):
        btn = ActionButton(
            text=text,
            svg_str=svg_path,
            tooltip=tooltip,
            width=WindowData.SettingButtonHeight,
            height=WindowData.SettingButtonHeight,
            is_circle=is_circle,
        )
        self.layout.addWidget(btn, Qt.AlignmentFlag.AlignLeft)
        return btn

    def add_seperator(self):
        frame = QFrame(self)
        frame.setFixedHeight(2)
        frame.setObjectName("frame_seperator")
        frame.setFrameShape(QFrame.Shape.HLine)
        self.layout.addWidget(frame)

    def update_history(self, list_session: list):
        self.history_model.reset_all(list_session)

    def remove_session(self, session_id: str):
        row = self.history_model.find(session_id)
        if row != -1:
            self.history_model.remove(row)

    def rename_session(self, session_id: str, new_title: str):
        row = self.history_model.find(session_id)
        if row != -1:
            self.history_model.update_title(row, new_title)

    def select_current(self, session_id: str):
        """让侧栏高亮和当前打开的会话保持一致（点击切换/程序切换都会调用）。"""
        row = self.history_model.find(session_id)
        if row == -1:
            self.history_list.clearSelection()
            return
        self.history_list.setCurrentIndex(self.history_model.index(row))

class GenerationPage(QWidget):
    generate_requested = Signal(object)
    stop_requested = Signal()
    retry_requested = Signal(str)
    session_changed = Signal(str)
    update_resized = Signal()
    setting_requested = Signal()

    save_message = Signal(dict)

    SESSION_ID = "default_session"

    """
        _history: [{
            "role": role,
            "content": {
                “content”: str,
                "attachments": list,
                "params": params      
            },
            "time": ts,
            "message_id": message_id,
            "model_type": model_type,
            "model_name": model_name,
        },……]
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.active_bubble = None
        self._background_workers = set()  # 持有运行中的线程，防止被 GC 提前销毁
        self.session_manager = ChatSessionManager()

        self.setup_ui()
        self.connection()
        self._load_initial_session()
        self.setEnabled(False)

    def setup_ui(self):
        outer  = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        self._splitter = main_splitter
        main_splitter.setContentsMargins(0, 0, 0, 0)
        main_splitter.setHandleWidth(6)  # 拖拽柄的宽度，可调
        main_splitter.setStyleSheet("""
            QSplitter::handle {
                background-color: #2d2d2d;
                margin: 0px 4px;
            }
            QSplitter::handle:hover {
                background-color: #3d8cff;
            }
        """)

        self._sidebar = SettingSidePage()

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)

        # Top bar：平时显示 Clear，进入多选模式后整条替换成选择计数 + 取消/删除
        topbar = QFrame()
        topbar.setObjectName("generation_top_bar")
        topbar.setFixedHeight(52)

        tb_layout = QHBoxLayout(topbar)
        tb_layout.setContentsMargins(16, 0, 16, 0)

        self._clear_btn = ActionButton(text=self.tr("Clear"), svg_str=":svg/clear.svg", width=80, height=40)

        self._normal_bar = QWidget()
        normal_layout = QHBoxLayout(self._normal_bar)
        normal_layout.setContentsMargins(0, 0, 0, 0)
        normal_layout.addWidget(self._clear_btn)

        self._selection_count_label = QLabel()
        self._selection_count_label.setObjectName("selection_count_label")
        self._cancel_selection_btn = QPushButton(self.tr("取消"))
        self._cancel_selection_btn.setObjectName("selection_cancel_btn")
        self._delete_selected_btn = QPushButton(self.tr("删除"))
        self._delete_selected_btn.setObjectName("selection_delete_btn")

        self._selection_bar = QWidget()
        self._selection_bar.setVisible(False)
        sel_layout = QHBoxLayout(self._selection_bar)
        sel_layout.setContentsMargins(0, 0, 0, 0)
        sel_layout.setSpacing(10)
        sel_layout.addWidget(self._selection_count_label)
        sel_layout.addStretch()
        sel_layout.addWidget(self._cancel_selection_btn)
        sel_layout.addWidget(self._delete_selected_btn)

        tb_layout.addStretch()
        tb_layout.addWidget(self._normal_bar)
        tb_layout.addWidget(self._selection_bar)

        self._chat = ChatWidget()
        self._input_bar = InputBar(self)

        # Model 选择器从顶栏挪到输入框下方，右侧对齐，左侧用提示文案填充，避免右重左轻
        self.model_combobox = ModelComboBox()
        self.model_combobox.setObjectName("model_combobox")
        self.model_combobox.setFixedWidth(180)

        bottom_info_bar = QWidget()
        bottom_info_bar.setObjectName("bottom_info_bar")
        bi_layout = QHBoxLayout(bottom_info_bar)
        bi_layout.setContentsMargins(6, 6, 6, 0)

        self._disclaimer_label = QLabel(self.tr("v1.0.1 © 2026 All rights reserved."))
        self._disclaimer_label.setObjectName("disclaimer_label")

        bi_layout.addWidget(self._disclaimer_label)
        bi_layout.addStretch()
        bi_layout.addWidget(QLabel(self.tr("Model：")))
        bi_layout.addWidget(self.model_combobox)

        input_panel = QWidget()
        input_layout = QVBoxLayout(input_panel)
        input_layout.setContentsMargins(100, 0, 100, 0)
        # 输入栏最小宽度会把 content 的最小宽度撑得很大，导致侧边栏无法拖宽，这里让 splitter 忽略它。
        input_panel.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        input_layout.setSpacing(0)
        input_layout.addWidget(self._input_bar)
        input_layout.addWidget(bottom_info_bar)

        content_layout.addWidget(topbar)
        content_layout.addWidget(self._chat, 1)
        content_layout.addWidget(input_panel)
        content_layout.addSpacing(20)

        self._chat_scrollbar = QScrollBar(Qt.Orientation.Vertical)
        self._chat_scrollbar.setFixedWidth(10)
        self._chat_scrollbar.setObjectName("chat_scroll_bar")

        # 双向绑定：自定义滚动条 ↔ ChatWidget 内部滚动条
        self.chat_vbar = self._chat.verticalScrollBar()

        main_splitter.addWidget(self._sidebar)
        main_splitter.addWidget(content)

        main_splitter.setStretchFactor(0, 1)
        main_splitter.setStretchFactor(1, 3)
        main_splitter.setSizes([260, 940])

        self._sidebar_width = 260
        self._sidebar_anim = QVariantAnimation(self)
        self._sidebar_anim.setDuration(220)
        self._sidebar_anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._sidebar_anim.valueChanged.connect(self._apply_sidebar_width)
        self._sidebar_anim.finished.connect(self._on_sidebar_anim_finished)
        self._sidebar_open = True
        outer.addWidget(main_splitter)

    def is_sidebar_open(self) -> bool:
        return self._sidebar_open

    def toggle_sidebar(self):
        self._sidebar_anim.stop()
        if self._sidebar_open:
            current = self._sidebar.width()
            if current > 0:
                self._sidebar_width = current
            self._sidebar_open = False
            self._sidebar_anim.setStartValue(current)
            self._sidebar_anim.setEndValue(0)
        else:
            self._sidebar_open = True
            self._sidebar_anim.setStartValue(self._sidebar.width())
            self._sidebar_anim.setEndValue(self._sidebar_width)
        self._sidebar_anim.start()

    def _apply_sidebar_width(self, w):
        w = int(w)
        # 动画期间靠 maximumWidth 压过侧边栏自身的最小宽度，结束后再还原。
        self._sidebar.setMaximumWidth(w)
        total = sum(self._splitter.sizes())
        self._splitter.setSizes([w, max(0, total - w)])

    def _on_sidebar_anim_finished(self):
        if self._sidebar_open:
            self._sidebar.setMaximumWidth(16777215)
            total = sum(self._splitter.sizes())
            self._splitter.setSizes([self._sidebar_width, max(0, total - self._sidebar_width)])

    def connection(self):
        self._clear_btn.clicked.connect(self.clear_chat)
        self._cancel_selection_btn.clicked.connect(lambda: self._chat.set_selection_mode(False))
        self._delete_selected_btn.clicked.connect(self._delete_selected_messages)
        self._chat.selection_mode_changed.connect(self._on_selection_mode_changed)
        self._chat.selection_changed.connect(self._on_chat_selection_changed)
        self._chat.retry_requested.connect(self._on_retry_requested)
        self._chat.edit_requested.connect(self._on_edit_requested)

        self._input_bar.submitted.connect(self._on_user_submit)
        self._input_bar.stop_requested.connect(self.stop_requested)
        self._input_bar.mode_combo.currentTextChanged.connect(self.on_changed_model_type)
        self._input_bar.prompt_input.textChanged.connect(self._on_text_changed)

        self._chat_scrollbar.valueChanged.connect(self.chat_vbar.setValue)
        self.chat_vbar.valueChanged.connect(self._chat_scrollbar.setValue)
        self.chat_vbar.rangeChanged.connect(
            lambda mn, mx: self._chat_scrollbar.setRange(mn, mx)
        )

        self.model_combobox.model_selected.connect(self._on_changed_model)
        self._sidebar.setting_btn.clicked.connect(self._on_go_to_setting)
        self._sidebar.new_session_btn.clicked.connect(self.create_new_session)
        self._sidebar.delete_session.connect(self.session_manager.delete_session)
        self._sidebar.switch_session.connect(self.session_manager.switch_session)

        self.session_manager.session_changed.connect(self._on_session_changed)
        self.session_manager.session_list_changed.connect(self._refresh_sidebar_history)

        self.save_message.connect(self._on_save_message)

    def resizeEvent(self, event, /):
        self.update_resized.emit()
        super().resizeEvent(event)

    def _load_initial_session(self):
        list_session = self.session_manager.list_sessions()
        if list_session:
            self.session_manager.switch_session(list_session[0]["session_id"])
        else:
            self.session_manager.create_new_session()

    def add_chat_message(self, role: str, content: str | dict, model_type: str = "text", item_count: int = 1):
        """追加一条完整消息"""
        ts = datetime.now().strftime("%H:%M")
        bubble = self._chat.add_message(role, content, ts, model_type=model_type, item_count=item_count)

        # 构造完整的历史记录
        model_text = self._input_bar.mode_combo.currentText()
        model_type = self._input_bar.label_to_key.get(model_text)
        model_name = self.model_combobox.currentText()
        history_item = {
            "session_id": self.session_manager.get_current_session_id(),
            "role": role,
            "content": content,
            "time": ts,
            "message_id": bubble.message_id,
            "model_type": model_type,
            "model_name": model_name,
        }
        return bubble, history_item

    def clear_chat(self):
        self._chat.clear()
        self.session_manager.clear_current_session()

        worker = ClearMemoryWorker(self.session_manager.get_current_session_id())
        worker.finished.connect(lambda w=worker: self._background_workers.discard(w))
        self._background_workers.add(worker)
        worker.start()

    def _on_selection_mode_changed(self, enabled: bool):
        self._normal_bar.setVisible(not enabled)
        self._selection_bar.setVisible(enabled)

    def _on_chat_selection_changed(self, count: int):
        self._selection_count_label.setText(self.tr(f"已选择 {count} 条"))

    def _delete_selected_messages(self):
        ids = self._chat.selected_ids()
        if not ids:
            return

        reply = QMessageBox.question(
            self,
            self.tr("删除消息"),
            self.tr(f"确定删除选中的 {len(ids)} 条消息吗？此操作无法撤销。"),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        deleted_ids = self._chat.delete_selected()
        self.session_manager.delete_messages(deleted_ids)

    def disable_ui(self):
        # 输入区其它控件锁住，但 send_btn 保持可点——切换成"停止"状态，
        # 这样生成过程中用户还能随时打断它。
        self._input_bar.set_busy(True)
        if not self.active_bubble.spinner_widget.isVisible():
            self.active_bubble.spinner_widget.setVisible(True)

    def enable_ui(self):
        self._input_bar.set_busy(False)
        self.active_bubble.hide_loading()

    def save_item_from_bubble(self, bubble):
        role = bubble.role
        content = bubble.get_persisted_content()
        ts = bubble.timestamp

        model_text = self._input_bar.mode_combo.currentText()
        model_type = self._input_bar.label_to_key.get(model_text)
        model_name = self.model_combobox.currentText()

        history_item = {
            "session_id": getattr(bubble, "generation_session_id", None)
                          or self.session_manager.get_current_session_id(),
            "role": role,
            "content": content,
            "time": ts,
            "message_id": bubble.message_id,
            "model_type": model_type,
            "model_name": model_name,
        }
        self.save_message.emit(history_item)
        self.finish_generation()

    def _on_go_to_setting(self):
        self.setting_requested.emit()

    def create_new_session(self):
        self.session_manager.create_new_session()

    def _mark_generating(self, bubble):
        bubble.generation_session_id = self.session_manager.get_current_session_id()
        bubble.generation_active = True

    def finish_generation(self):
        """生成结束（完成/取消/失败）：气泡不再需要跨会话保活。"""
        bubble = self.active_bubble
        if bubble is None:
            return
        bubble.generation_active = False

    def _on_session_changed(self):
        # 生成中的气泡还没写入历史，clear() 会把它销毁；先摘下来，回到原会话时再挂回去。
        pending = self.active_bubble
        if pending is not None and getattr(pending, "generation_active", False):
            self._chat.detach_message(pending.message_id)
        else:
            pending = None

        self._chat.clear()
        self._chat.load_history(self.session_manager.get_history())
        if pending is not None and pending.generation_session_id == self.session_manager.get_current_session_id():
            self._chat.attach_bubble(pending)
        self._refresh_sidebar_history()

    def _refresh_sidebar_history(self):
        self._sidebar.update_history(self.session_manager.list_sessions())
        self._sidebar.select_current(self.session_manager.get_current_session_id())

    def _on_changed_model(self, text):
        self._input_bar.set_model(text)

    def activate_model_type(self):
        self.on_changed_model_type(self._input_bar.mode_combo.currentText())

    def on_changed_model_type(self, text):
        type_str = self._input_bar.label_to_key.get(text)
        self.model_combobox.clear_models()

        # 请求走线程，避免后端忙（预加载/模型加载）时同步 HTTP 卡住主线程。
        worker = ModelListWorker(type_str)
        worker.loaded.connect(self._on_model_list_loaded)
        worker.finished.connect(lambda w=worker: self._background_workers.discard(w))
        self._background_workers.add(worker)
        worker.start()

    def _on_model_list_loaded(self, type_str, items):
        # 连续快速切换时，只采用与当前模式一致的最新结果，丢弃过期响应。
        current_type = self._input_bar.label_to_key.get(self._input_bar.mode_combo.currentText())
        if type_str != current_type or not items.ok:
            return
        self.model_combobox.clear_models()

        tags = items.tags
        for tag in tags.keys():
            for item in tags[tag]:
                self.model_combobox.add_model(item, tag=tag)

        name = self.model_combobox.currentText()
        self._input_bar.set_model(name)

    def _on_text_changed(self):
        if self._input_bar.is_generating:
            return
        if self._input_bar.prompt_input.toPlainText():
            self._input_bar.send_btn.setEnabled(True)
        else:
            self._input_bar.send_btn.setEnabled(False)

    @staticmethod
    def _item_count(model_type, params: dict) -> int:
        """本次请求预期产出几个结果，用于决定加载占位显示几个槽。"""
        if model_type == FactoryType.Image:
            return params.get("number", 1)
        if model_type == FactoryType.Speech:
            return params.get("batch_size", 1)
        return 1

    def on_load_stage(self, payload: dict):
        """后端推来的模型加载阶段（下载/加载），显示在当前气泡的占位行上。"""
        if self.active_bubble is None:
            return
        self.active_bubble.set_load_stage(
            payload.get("stage", ""),
            payload.get("progress") or 0.0,
            payload.get("detail", ""),
        )

    def on_partial_attachment(self, path: str):
        if self.active_bubble is not None:
            self.active_bubble.add_partial_attachment(path)

    def _on_user_submit(self, payload: "InputPayload"):
        user_content = {
            "model_name": self.model_combobox.currentText(),
            "content": payload.prompt,
            "attachments": [str(att.path) for att in payload.attachments],
            "extra": {**payload.params}
        }
        bubble, item = self.add_chat_message("user", user_content)
        model_type = FactoryType.convert_by_text(payload.mode)
        item_count = self._item_count(model_type, payload.params)
        self.active_bubble, _ = self.add_chat_message(
            "assistant", "", model_type=payload.mode, item_count=item_count
        )
        self._mark_generating(self.active_bubble)
        # 存历史是磁盘 IO（首条消息还会触发标题重命名 + 重新查询会话列表），
        # 挪到下一轮事件循环，先让两个新气泡画出来，发送感觉才是"立刻"的。
        QTimer.singleShot(0, lambda: self.save_message.emit(item))

        # 禁用输入，显示进度条
        self.disable_ui()

        # 发出信号给 controller
        self.generate_requested.emit({
            "session_id": self.session_manager.get_current_session_id(),
            "model_type": FactoryType.convert_by_text(payload.mode),
            "model_name": self.model_combobox.currentText(),
            "params": user_content,
        })

    def _on_retry_requested(self, message_id: str):
        user_msg, success = self.session_manager.prepare_for_retry(message_id)
        if not success or user_msg is None:
            return

        self._chat.clear_from_index(len(self.session_manager.get_history()))
        model_text = self._input_bar.mode_combo.currentText()
        model_type = self._input_bar.label_to_key.get(model_text)
        model_enum = FactoryType.convert_by_text(model_type)
        retry_params = (user_msg.get("content") or {}).get("extra", {})
        item_count = self._item_count(model_enum, retry_params)
        self.active_bubble, _ = self.add_chat_message(
            "assistant", "", model_type=model_type, item_count=item_count
        )
        self._mark_generating(self.active_bubble)

        # 禁用输入，显示进度条
        self.disable_ui()

        user_content = user_msg.get("content", "")

        # 发出重试请求
        self.generate_requested.emit({
            "session_id": self.session_manager.get_current_session_id(),
            "model_type": FactoryType.convert_by_text(model_type),
            "model_name": self.model_combobox.currentText(),
            "params": user_content,
        })

    def _on_edit_requested(self, message_id, content):
        self.session_manager.update_message_content(message_id, content)
        self._on_retry_requested(message_id)

    def _on_save_message(self, history_item):
        self.session_manager.add_message(history_item)
        self._input_bar.setEnabled(True)
