from PySide6.QtCore import Qt, Signal, QObject, QThread
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QGroupBox, QLabel, QPushButton,
    QRadioButton, QButtonGroup, QTextEdit
)

from src.app.client import ApiGuardClient
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
            self.signals.done.emit(True, "安装完成")
        except Exception as e:
            self.signals.done.emit(False, str(e))


class GpuPage(QWidget):
    install = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("gpu_page")
        self._config = ConfigManager()
        self._worker = None
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(32, 24, 32, 24)
        layout.setSpacing(20)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # 标题
        title = QLabel(self.tr("Gpu Setting"))
        title.setObjectName("page_title")
        layout.addWidget(title)

        # 当前状态
        status_group = QGroupBox("当前状态")
        status_group.setObjectName("setting_group")
        status_layout = QVBoxLayout(status_group)

        self._status_label = QLabel("检测中...")
        self._status_label.setObjectName("gpu_status_label")
        self._status_label.setWordWrap(True)
        status_layout.addWidget(self._status_label)

        self._refresh_btn = QPushButton("刷新状态")
        self._refresh_btn.setFixedWidth(90)
        self._refresh_btn.clicked.connect(self._refresh_status)
        status_layout.addWidget(
            self._refresh_btn,
            alignment=Qt.AlignmentFlag.AlignLeft
        )
        layout.addWidget(status_group)

        # 后端选择
        backend_group = QGroupBox("计算后端")
        backend_group.setObjectName("setting_group")
        backend_layout = QVBoxLayout(backend_group)

        self._btn_group = QButtonGroup(self)
        self._cpu_radio  = QRadioButton("CPU（兼容所有设备，速度较慢）")
        self._cuda_radio = QRadioButton("CUDA（需要 NVIDIA GPU，速度快）")
        self._btn_group.addButton(self._cpu_radio,  0)
        self._btn_group.addButton(self._cuda_radio, 1)
        backend_layout.addWidget(self._cpu_radio)
        backend_layout.addWidget(self._cuda_radio)

        # 安装按钮
        btn_row = QHBoxLayout()
        self._install_btn = QPushButton("应用并安装")
        self._install_btn.setFixedWidth(110)
        self._install_btn.clicked.connect(self._on_install)
        btn_row.addWidget(self._install_btn)
        btn_row.addStretch()
        backend_layout.addLayout(btn_row)

        layout.addWidget(backend_group)

        # 安装日志
        log_group = QGroupBox("安装日志")
        log_group.setObjectName("setting_group")
        log_layout = QVBoxLayout(log_group)

        self._install_log = QTextEdit()
        self._install_log.setReadOnly(True)
        self._install_log.setFixedHeight(150)
        self._install_log.setObjectName("install_log")
        log_layout.addWidget(self._install_log)

        layout.addWidget(log_group)
        layout.addStretch()

        # 监听选择变化
        self._btn_group.buttonClicked.connect(self._on_backend_changed)

    def _refresh_status(self):
        try:
            info = ApiGuardClient.instance().detect_device()
            if info.get("cuda_available"):
                gpus = info.get("gpus", [{}])[-1]
                print(gpus)
                gpu_name = gpus.get("name", "未知")
                gpu_size = gpus.get('total_memory_gb', '?')
                text = (
                    f"后端：CUDA  ｜  "
                    f"GPU：{gpu_name}  ｜  "
                    f"显存：{gpu_size} GB"
                )
                self._status_label.setStyleSheet("color: #4ec994;")
            else:
                text = "后端：CPU（未检测到 NVIDIA GPU 或 CUDA 未安装）"
                self._status_label.setStyleSheet("color: #e5c07b;")
            self._status_label.setText(text)
        except Exception as e:
            self._status_label.setText(f"状态获取失败: {e}")
            self._status_label.setStyleSheet("color: #e06c75;")

    def _on_install(self):
        backend = "cuda" if self._cuda_radio.isChecked() else "cpu"
        self._install_log.clear()
        self._install_btn.setEnabled(False)
        self._append_log(f"开始安装 {backend.upper()} 后端...")

        try:
            self._worker = _InstallWorker(backend, ApiGuardClient.instance())
            self._worker.signals.progress.connect(self._append_log)
            self._worker.signals.done.connect(self._on_install_done)
            self._worker.start()
        except Exception as e:
            self._append_log(f"错误: {e}")
            self._install_btn.setEnabled(True)

    def _append_log(self, line: str):
        self._install_log.insertPlainText(line + "\n")
        self._install_log.ensureCursorVisible()

    def _on_install_done(self, success: bool, msg: str):
        self._install_btn.setEnabled(True)
        if success:
            self._append_log(f"✓ {msg}")
            log_success(f"后端安装完成: {msg}")
            self._refresh_status()
        else:
            self._append_log(f"✗ {msg}")
            log_error(f"后端安装失败: {msg}")

    def _on_backend_changed(self):
        self.collect()

    def load(self):
        backend = self._config.get("gpu", "backend", "cpu")
        if backend == "cuda":
            self._cuda_radio.setChecked(True)
        else:
            self._cpu_radio.setChecked(True)
        self._refresh_status()

    def collect(self):
        backend = "cuda" if self._cuda_radio.isChecked() else "cpu"
        self._config.set("gpu", "backend", backend)