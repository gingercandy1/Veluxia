from datetime import datetime

from PySide6.QtCore import Signal, Qt
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                               QFrame, QScrollBar)

from app.client import ApiClient
from app.ui.input.input_bar import InputBar, InputPayload
from app.ui.model_comb import ModelComboBox
from src.shared.enum_type import FactoryType
from ui.base.action_button import ActionButton
from ui.chat.chat_session_manager import ChatSessionManager
from ui.chat.chat_widget import ChatWidget
from ui.window_data import WindowData

def init_widget():
    from src.app.ui.param.core.text.llama_chat_panel import LlamaChatPanel
    from src.app.ui.param.core.image.flux_schnell_panel import FluxSchnellPanel
    from src.app.ui.param.core.image_frame.film_interpolation_panel import FilmInterpolationPanel
    from src.app.ui.param.core.animation.ltx_video_panel import LTXVideoPanel
    from src.app.ui.param.core.animation.wan2_video_panel import Wan2VideoPanel
    from src.app.ui.param.core.speech.ace_step_panel import AceStepMusicPanel
    from src.app.ui.param.core.speech.qwen3_tts_panel import Qwen3TTSPanel
init_widget()


class SettingSidePage(QFrame):
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
        self.layout.addStretch()

    def add_button(self, svg_path, text, tooltip, is_circle=False):
        btn = ActionButton(
            text=text,
            svg_str=svg_path,
            tooltip=tooltip,
            width=WindowData.SettingButtonWidth,
            height=WindowData.SettingButtonHeight,
            is_circle=is_circle,
        )
        self.layout.addWidget(btn, Qt.AlignmentFlag.AlignCenter)
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



class GenerationPage(QWidget):
    generate_requested = Signal(object)
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
        self.session_manager = ChatSessionManager()

        self.setup_ui()
        self.connection()
        self._load_initial_session()

    def setup_ui(self):
        outer  = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self._sidebar = SettingSidePage()
        self._sidebar.setFixedWidth(WindowData.SettingWidth)

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)

        # Top bar
        topbar = QFrame()
        topbar.setObjectName("generation_top_bar")
        topbar.setFixedHeight(52)

        tb_layout = QHBoxLayout(topbar)
        tb_layout.setContentsMargins(16, 0, 16, 0)

        self.model_combobox = ModelComboBox()
        self.model_combobox.setObjectName("model_combobox")
        self.model_combobox.setFixedWidth(180)

        self._clear_btn = ActionButton(text=self.tr("Clear"), svg_str=":svg/clear.svg", width=80, height=40)

        tb_layout.addStretch()
        tb_layout.addWidget(QLabel(self.tr("Model：")))
        tb_layout.addWidget(self.model_combobox)
        tb_layout.addSpacing(12)
        tb_layout.addWidget(self._clear_btn)

        self._chat = ChatWidget()
        self._input_bar = InputBar(self)

        input_panel = QWidget()
        input_layout = QVBoxLayout(input_panel)
        input_layout.setContentsMargins(100, 0, 100, 0)
        input_layout.setSpacing(0)
        input_layout.addWidget(self._input_bar)

        content_layout.addWidget(topbar)
        content_layout.addWidget(self._chat, 1)
        content_layout.addWidget(input_panel)
        content_layout.addSpacing(20)

        self._chat_scrollbar = QScrollBar(Qt.Orientation.Vertical)
        self._chat_scrollbar.setFixedWidth(10)
        self._chat_scrollbar.setObjectName("chat_scroll_bar")

        # 双向绑定：自定义滚动条 ↔ ChatWidget 内部滚动条
        self.chat_vbar = self._chat.verticalScrollBar()

        outer.addWidget(self._sidebar)
        outer.addWidget(content, 1)
        outer.addWidget(self._chat_scrollbar)

    def connection(self):
        self._clear_btn.clicked.connect(self.clear_chat)
        self._chat.retry_requested.connect(self._on_retry_requested)
        self._chat.edit_requested.connect(self._on_edit_requested)

        self._input_bar.submitted.connect(self._on_user_submit)
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
        self.session_manager.session_changed.connect(self._on_session_changed)

        self.save_message.connect(self._on_save_message)

    def resizeEvent(self, event, /):
        self.update_resized.emit()
        super().resizeEvent(event)

    # ── 历史持久化 ────────────────────────────────────────────────────────────
    def _load_initial_session(self):
        self.session_manager.switch_session(self.SESSION_ID)

    def add_chat_message(self, role: str, content: str | dict):
        """追加一条完整消息"""
        ts = datetime.now().strftime("%H:%M")
        bubble = self._chat.add_message(role, content, ts)

        # 构造完整的历史记录
        model_text = self._input_bar.mode_combo.currentText()
        model_type = self._input_bar.label_to_key.get(model_text)
        model_name = self.model_combobox.currentText()
        history_item = {
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

        ApiClient.instance().clear_memory(
            session_id=self.session_manager.get_current_session_id()
        )
        # self.create_new_session()

    def disable_ui(self):
        if self._input_bar.isEnabled():
            self._input_bar.setEnabled(False)
        if not self.active_bubble.spinner_widget.isVisible():
            self.active_bubble.spinner_widget.setVisible(True)

    def enable_ui(self):
        if not self._input_bar.isEnabled():
            self._input_bar.setEnabled(True)
        if self.active_bubble.spinner_widget.isVisible():
            self.active_bubble.spinner_widget.setVisible(False)

    def _on_go_to_setting(self):
        self.setting_requested.emit()

    def create_new_session(self):
        self.session_manager.create_new_session()

    def _on_session_changed(self, session_id: str):
        self._chat.clear()
        self._chat.load_history(self.session_manager.get_history())

    def _on_changed_model(self, text):
        self._input_bar.set_model(text)

    def activate_model_type(self):
        self.on_changed_model_type(self._input_bar.mode_combo.currentText())

    def on_changed_model_type(self, text):
        type_str = self._input_bar.label_to_key.get(text)
        self.model_combobox.clear_models()

        items = ApiClient.instance().get_model_info(type_str)
        if not items.ok: return

        tags = items.tags
        for tag in tags.keys():
            for item in tags[tag]:
                self.model_combobox.add_model(item, tag=tag)

        name = self.model_combobox.currentText()
        self._input_bar.set_model(name)

    def _on_text_changed(self):
        if self._input_bar.prompt_input.toPlainText():
            self._input_bar.send_btn.setEnabled(True)
        else:
            self._input_bar.send_btn.setEnabled(False)

    def _on_user_submit(self, payload: "InputPayload"):
        user_content = {
            "model_name": self.model_combobox.currentText(),
            "content": payload.prompt,
            "attachments": [str(att.path) for att in payload.attachments],
            "extra": {**payload.params}
        }
        bubble, item = self.add_chat_message("user", user_content)
        self.active_bubble, _ = self.add_chat_message("assistant", "")
        self.save_message.emit(item)

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
        self.active_bubble, _ = self.add_chat_message("assistant", "")

        # 禁用输入，显示进度条
        self._input_bar.setEnabled(False)

        user_content = user_msg.get("content", "")
        model_text = self._input_bar.mode_combo.currentText()
        model_type = self._input_bar.label_to_key.get(model_text)

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
        # 恢复输入
        self._input_bar.setEnabled(True)
