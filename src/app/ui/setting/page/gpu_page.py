from PySide6.QtCore import Qt, Signal, QObject, QThread
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QGroupBox, QLabel, QPushButton,
    QRadioButton, QButtonGroup
)

from src.app.client import ApiGuardClient
from src.app.work import LibraryTaskWorker
from src.app.ui.setting.page.install_progress import InstallProgress, scrollable_layout
from src.app.ui.setting.page.log_page import log_error, log_success
from src.shared.settings import ConfigManager


# 思路
# 后台会单独运行一个服务，用来监听安装gpu版本还是cpu版本的
# 实现安装和获取安装进度状态的接口
# 在安装的时候先结束其它的进程，然后再执行安装，安装好后，然后再重新这个进程（可以再客户端中执行）


class _WorkerSignals(QObject):
    progress = Signal(str)
    done     = Signal(bool, str)


class _InstallWorker(QThread):
    def __init__(self, backend: str, client):
        super().__init__()
        self._backend = backend
        self._client  = client
        self.signals  = _WorkerSignals()

    def run(self):
        try:
            self._client.install_backend(
                extra=self._backend,
                on_progress=lambda line: self.signals.progress.emit(line),
            )
            self.signals.done.emit(True, self.tr("Installation complete"))
        except Exception as e:
            self.signals.done.emit(False, str(e))


class GpuPage(QWidget):
    install = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("gpu_page")
        self._config = ConfigManager()
        self._worker = None
        self._detect_worker = None
        self._build_ui()
        self.load()

    def showEvent(self, event):
        super().showEvent(event)
        # 只在打开这一页时检测：主窗口构造时守护服务还没起来，同步请求必然失败，
        # 而 Windows 上连一个没人监听的端口要卡约 2 秒，窗口会晚出来
        self._refresh_status()

    def _build_ui(self):
        layout = scrollable_layout(self)
        layout.setContentsMargins(32, 24, 32, 24)
        layout.setSpacing(20)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # 标题
        title = QLabel(self.tr("GPU Settings"))
        title.setObjectName("page_title")
        layout.addWidget(title)

        # 当前状态
        status_group = QGroupBox(self.tr("Current status"))
        status_group.setObjectName("setting_group")
        status_layout = QVBoxLayout(status_group)

        self._status_label = QLabel(self.tr("Detecting..."))
        self._status_label.setObjectName("gpu_status_label")
        self._status_label.setWordWrap(True)
        status_layout.addWidget(self._status_label)

        self._refresh_btn = QPushButton(self.tr("Refresh"))
        self._refresh_btn.setMinimumWidth(90)
        self._refresh_btn.clicked.connect(self._refresh_status)
        status_layout.addWidget(
            self._refresh_btn,
            alignment=Qt.AlignmentFlag.AlignLeft
        )
        layout.addWidget(status_group)

        # 后端选择
        backend_group = QGroupBox(self.tr("Compute backend"))
        backend_group.setObjectName("setting_group")
        backend_layout = QVBoxLayout(backend_group)

        self._btn_group = QButtonGroup(self)
        self._cpu_radio  = QRadioButton(self.tr("CPU (works on any device, slower)"))
        self._cuda_radio = QRadioButton(self.tr("CUDA (requires an NVIDIA GPU, faster)"))
        self._btn_group.addButton(self._cpu_radio,  0)
        self._btn_group.addButton(self._cuda_radio, 1)
        backend_layout.addWidget(self._cpu_radio)
        backend_layout.addWidget(self._cuda_radio)

        # 安装按钮
        btn_row = QHBoxLayout()
        self._install_btn = QPushButton(self.tr("Apply and install"))
        self._install_btn.setMinimumWidth(110)
        self._install_btn.clicked.connect(self._on_install)
        btn_row.addWidget(self._install_btn)
        btn_row.addStretch()
        backend_layout.addLayout(btn_row)

        # 安装进度和日志：没安装过时不显示
        self._progress = InstallProgress()
        backend_layout.addWidget(self._progress)

        layout.addWidget(backend_group)
        layout.addStretch()

        # 监听选择变化
        self._btn_group.buttonClicked.connect(self._on_backend_changed)

    def _refresh_status(self):
        """在后台线程里请求守护服务，界面不等它。"""
        if self._detect_worker is not None:
            return
        self._refresh_btn.setEnabled(False)
        self._status_label.setStyleSheet("")
        self._status_label.setText(self.tr("Detecting..."))
        self._detect_worker = LibraryTaskWorker(ApiGuardClient.instance().detect_device)
        self._detect_worker.finished_ok.connect(self._show_status)
        self._detect_worker.error.connect(lambda message: self._show_status(None, message))
        self._detect_worker.finished.connect(self._on_detect_finished)
        self._detect_worker.start()

    def _on_detect_finished(self):
        self._detect_worker = None
        self._refresh_btn.setEnabled(True)

    def _show_status(self, info: dict | None, error: str = ""):
        try:
            if info is None:
                raise RuntimeError(error)
            if not info:
                # detect_device 连不上时返回空字典：多半是刚启动、守护服务还没起来，别误报成"没有显卡"
                self._status_label.setText(self.tr(
                    "The local service is not ready yet. Click Refresh in a moment."))
                self._status_label.setStyleSheet("color: #e5c07b;")
                return
            if info.get("cuda_available"):
                gpus = info.get("gpus", [{}])[-1]
                print(gpus)
                gpu_name = gpus.get("name", self.tr("Unknown"))
                gpu_size = gpus.get('total_memory_gb', '?')
                text = self.tr("Backend: CUDA  |  GPU: {0}  |  VRAM: {1} GB").format(
                    gpu_name, gpu_size)
                self._status_label.setStyleSheet("color: #4ec994;")
            else:
                text = self.tr("Backend: CPU (no NVIDIA GPU detected, or CUDA is not installed)")
                self._status_label.setStyleSheet("color: #e5c07b;")
            self._status_label.setText(text)
        except Exception as e:
            self._status_label.setText(self.tr("Failed to get status: {0}").format(e))
            self._status_label.setStyleSheet("color: #e06c75;")

    def _on_install(self):
        backend = "cuda" if self._cuda_radio.isChecked() else "cpu"
        self._install_btn.setEnabled(False)
        self._progress.start(self.tr("Installing the {0} backend...").format(backend.upper()))

        try:
            self._worker = _InstallWorker(backend, ApiGuardClient.instance())
            self._worker.signals.progress.connect(self._progress.append)
            self._worker.signals.done.connect(self._on_install_done)
            self._worker.start()
        except Exception as e:
            self._progress.finish(False, self.tr("Error: {0}").format(e))
            self._install_btn.setEnabled(True)

    def _on_install_done(self, success: bool, msg: str):
        self._install_btn.setEnabled(True)
        self._progress.finish(success, msg)
        if success:
            log_success(f"后端安装完成: {msg}")
            self._refresh_status()
        else:
            log_error(f"后端安装失败: {msg}")

    def _on_backend_changed(self):
        self.collect()

    def load(self):
        backend = self._config.get("gpu", "backend", "cpu")
        if backend == "cuda":
            self._cuda_radio.setChecked(True)
        else:
            self._cpu_radio.setChecked(True)

    def collect(self):
        backend = "cuda" if self._cuda_radio.isChecked() else "cpu"
        self._config.set("gpu", "backend", backend)