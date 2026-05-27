import subprocess
import sys

from PySide6.QtWidgets import QMainWindow, QWidget, QHBoxLayout, QStackedWidget, QVBoxLayout, QFrame
from app.work import ApiWorker, BackendStartupWorker
from app.client import ApiClient
from app.param import GenerationRequest
from app.ui.base.action_button import ActionButton
from app.ui.gen_page import GenerationPage
from app.ui.setting.setting import SettingPage
from app.ui.window_data import WindowData
from src.shared.settings import ConfigManager
from src.shared.schemas import ImageResponse, AnimationResponse, SpeechResponse
from ui.setting.page.log_page import log_info, log_error


class TopBar(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(WindowData.TopBarHeight)
        self.setObjectName("top_bar")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(13, 0, 13, 0)
        layout.setSpacing(0)

        self.stack_widget = QStackedWidget()

        main_page = self.build_main_page()
        self.stack_widget.addWidget(main_page)

        setting_page = self.build_setting_page()
        self.stack_widget.addWidget(setting_page)

        layout.addWidget(self.stack_widget)

        self.back_btn.clicked.connect(lambda: self.set_index(0))

    def build_main_page(self):
        widget = QWidget()
        widget.setContentsMargins(0, 0, 0, 0)
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)

        self.side_btn = self.add_circle_button(":/svg/side.svg", "", self.tr("hide side"), is_circle=True)
        layout.addWidget(self.side_btn)
        layout.addStretch()
        return widget

    def build_setting_page(self):
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)

        self.back_btn = self.add_circle_button(":/svg/back.svg", "", self.tr("back"), is_circle=True)
        layout.addWidget(self.back_btn)
        layout.addStretch()
        return widget

    def add_circle_button(self, svg_path, text, tooltip, is_circle=False):
        btn = ActionButton(
            text=text,
            svg_str=svg_path,
            tooltip=tooltip,
            width=26,
            height=26,
            is_circle=is_circle,
        )
        return btn

    def set_index(self, index):
        self.stack_widget.setCurrentIndex(index)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(self.tr("Veluxia"))
        self.resize(WindowData.MainSize)
        self._sidebar_visible = True

        self._startup = BackendStartupWorker()
        self._startup.ready.connect(self._on_backend_ready)
        self._startup.timeout.connect(self._on_backend_timeout)
        self._startup.log.connect(lambda msg: log_info(msg))
        self._startup.start()

        self._client = ApiClient()

        root  = QWidget()
        root .setContentsMargins(0, 0, 0, 0)
        self.setCentralWidget(root)

        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)

        self._top_bar = TopBar()
        root_layout.addWidget(self._top_bar)

        self._stack = QStackedWidget()
        root_layout.addWidget(self._stack, 1)

        # Page 0
        self._gen_page = GenerationPage(self)
        self._gen_page.generate_requested.connect(self.on_generate_requested)
        self._stack.addWidget(self._gen_page)

        # Page 1
        self._setting_page = SettingPage()
        self._stack.addWidget(self._setting_page)

        self._connect()
        self.debug_paint_areas()

        self._output_started = False

    def debug_paint_areas(self):
        """ 调试方法：为主要区域和控件注入不同背景色，便于观察布局边界 """
        debug_styles = []
        target_objects = {
            # "chat_widget": "background-color: rgb(0, 100, 100); border: 2px solid red;",  # 顶部栏-淡粉红
            # "generation_top_bar": "background-color: rgb(255, 0, 0); border: 2px solid red;",  # 顶部栏-淡粉红
            # "setting_side_page": "background-color: rgba(173, 216, 230, 0.6); border: 2px solid blue;",  # 侧边栏-淡蓝色
            # "frame_seperator": "background-color: red;",  # 分割线-亮红
            # "prompt_input":  "background-color: rgb(100, 100, 0); border: 2px solid red;",  # 分割线-亮红
            # "input_page":  "background-color: rgb(100, 0, 0); border: 2px solid red;",  # 分割线-亮红
            # "user_bubble": "background-color: rgb(100, 0, 0); border: 2px solid red;",  # 分割线-亮红
            # "user_bubble_box": "background-color: rgb(200, 0, 0); border: 2px solid black;",  # 分割线-亮红
            # "assistant_bubble": "background-color: rgb(0, 100, 0); border: 2px solid red;",  # 分割线-亮红
            # "assistant_bubble_box": "background-color: rgb(0, 200, 0); border: 2px solid black;",  # 分割线-亮红
            # "text_content_widget": "background-color: rgb(0, 0, 200); border: 2px solid black;",  # 分割线-亮红
            # "content_container": "background-color: rgb(0, 200, 0); border: 2px solid black;",  # 分割线-亮红
            # "bubble_wrap": "background-color: rgb(200, 200, 200); border: 2px solid black;",  # 分割线-亮红
            # "collapse_container": "background-color: rgb(100, 100, 100); border: 2px solid black;",  # 分割线-亮红
        }

        # 将这些特定名字的样式加入列表
        for obj_name, style in target_objects.items():
            debug_styles.append(f"#{obj_name} {{ {style} }}")

        # 2. 自动递归遍历所有没有设置 objectName 的 QWidget/QStackedWidget 等容器，防止肉眼漏掉区域
        # 我们可以通过类名选择器来高亮它们
        # debug_styles.append(
        #     "GenerationPage { background-color: rgba(144, 238, 144, 0.4); border: 2px solid green; }")  # 生成页-淡绿
        # debug_styles.append(
        #     "SettingPage { background-color: rgba(221, 160, 221, 0.4); border: 2px solid purple; }")  # 设置页-淡紫

        # 3. 将所有调试样式组合，应用到全局窗口上
        full_style = "\n".join(debug_styles)
        self.setStyleSheet(full_style)
        log_info("🛠️ 界面区域彩色调试模式已开启...")

    def _connect(self):
        self._top_bar.back_btn.clicked.connect(self._on_back_btn_clicked)
        self._gen_page.setting_requested.connect(self._go_to_setting)
        self._startup.ready.connect(self._gen_page.activate_model_type)
        self._setting_page.install_requested.connect(lambda: self._startup.close())

    def _on_back_btn_clicked(self):
        self._stack.setCurrentIndex(0)
        self._top_bar.set_index(0)

    def _go_to_setting(self):
        self._stack.setCurrentIndex(1)
        self._top_bar.set_index(1)

    def on_generate_requested(self, params: dict):
        # self._gen_page.show_chat_progress()
        # 翻译提示词
        req = GenerationRequest.build(
            model_type=params["model_type"],
            model_name=params["model_name"],
            prompt=params["params"]["content"],
            model_params=params["params"]["extra"],
            attachments=params["params"]["attachments"],
            session_id=params["params"]["attachments"],
            setting=ConfigManager().get_backend_config(),
        )

        _client = ApiClient().instance()
        response = self._client.translate(req, is_default=True)
        GenerationRequest.open_translate(response.translate_result)

        # 创建 Worker 并启动
        self._worker = ApiWorker(self._client, req, params["model_type"])
        self._worker.finished_ok.connect(self.on_generate_finished)
        self._worker.error.connect(self._on_generate_error)

        self._worker.thinking_chunk.connect(self._on_think_chunk)
        self._worker.text_chunk.connect(self._on_text_chunk)
        self._worker.stream_done.connect(self._on_stream_done)
        self._worker.start()

    def on_generate_finished(self, result):
        self._gen_page.enable_ui()
        files = []
        if isinstance(result, ImageResponse):
            files = result.paths
        elif isinstance(result, AnimationResponse):
            files =  [result.video_path]
        elif isinstance(result, SpeechResponse):
            files = [result.audio_path]

        active_bubble = self._gen_page.active_bubble
        if not active_bubble:
            return
        active_bubble.load_attachments(files)
        log_info(f"生成完成:{result.session_id}")

    def _on_generate_error(self, msg: str):
        self._gen_page.enable_ui()
        log_error(f"生成出错:{msg}", )

    def _on_think_chunk(self, text: str):
        active_bubble = self._gen_page.active_bubble
        if not active_bubble:
            return
        active_bubble.append_thinking(text)

    def _on_text_chunk(self, text: str):
        active_bubble = self._gen_page.active_bubble
        if not active_bubble:
            return

        self._gen_page.enable_ui()

        if not self._output_started:
            self._output_started = True
            active_bubble.switch_to_generating()

        # print("front:", text)
        active_bubble.append_output(text)

    def _on_stream_done(self, is_think: bool):
        active_bubble = self._gen_page.active_bubble
        if not active_bubble:
            return

        active_bubble.finish()
        self._output_started = False
        # 非思考模型直接隐藏思考区
        if not is_think:
            active_bubble.hide_think_area()

    def _on_backend_ready(self):
        log_info("✅ Backend 已就緒")
        self._gen_page.setEnabled(True)

    def _on_backend_timeout(self):
        log_error("⚠️ Backend 啟動超時")

    def closeEvent(self, event):
        self._gen_page.closeEvent(event)
        self._client.close()
        if self._startup:
            self._startup.close()
            self._startup.close_guard()
        super().closeEvent(event)


if __name__ == '__main__':
    proc = subprocess.Popen(
        [sys.executable, "-m", "backend.server", "--port", str(8765)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )