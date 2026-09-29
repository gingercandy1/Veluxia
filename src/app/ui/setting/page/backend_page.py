from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from src.app import local_backend
from src.app.client import BackendStatus, probe_backend
from src.app.ui.setting.page.install_progress import InstallProgress, scrollable_layout
from src.app.ui.setting.page.log_page import log_error, log_success
from src.app.work import BackendInstallWorker, LibraryTaskWorker
from src.shared.settings import ConfigManager


class BackendPage(QWidget):
    """选择连本机后端还是远程服务（ADR 0005），并负责安装本机后端。"""

    install = Signal()            # 安装前要先停掉正在运行的本机后端，否则 python/ 目录被占用
    restart_requested = Signal()  # 装好后要重启前端才会按新后端重新走启动流程

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("backend_page")
        self._config = ConfigManager()
        self._probe_worker = None
        self._install_worker = None
        self._build_ui()
        self.load()

    def showEvent(self, event):
        super().showEvent(event)
        self._refresh_install_status()

    def _build_ui(self):
        layout = scrollable_layout(self)
        layout.setContentsMargins(32, 24, 32, 24)
        layout.setSpacing(20)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        title = QLabel(self.tr("Backend"))
        title.setObjectName("page_title")
        layout.addWidget(title)

        layout.addWidget(self._build_mode_group())
        layout.addWidget(self._build_remote_group())
        layout.addWidget(self._build_local_group())
        layout.addStretch()

    def _build_mode_group(self) -> QGroupBox:
        group = QGroupBox(self.tr("Connection"))
        group.setObjectName("setting_group")
        group_layout = QVBoxLayout(group)

        self._mode_group = QButtonGroup(self)
        self._local_radio = QRadioButton(self.tr("Local backend (runs on this computer)"))
        self._remote_radio = QRadioButton(self.tr("Remote backend (connect to a deployed service)"))
        self._mode_group.addButton(self._local_radio, 0)
        self._mode_group.addButton(self._remote_radio, 1)
        group_layout.addWidget(self._local_radio)
        group_layout.addWidget(self._remote_radio)

        hint = QLabel(self.tr("Changes take effect after saving and restarting the app."))
        hint.setObjectName("setting_hint")
        group_layout.addWidget(hint)

        self._mode_group.buttonClicked.connect(self._on_mode_changed)
        return group

    def _build_remote_group(self) -> QGroupBox:
        self._remote_box = QGroupBox(self.tr("Remote backend"))
        self._remote_box.setObjectName("setting_group")
        form = QFormLayout(self._remote_box)

        self._url_edit = QLineEdit()
        self._url_edit.setPlaceholderText("http://192.168.1.10:8765")
        self._token_edit = QLineEdit()
        # token 等同于远程服务的密码，不明文显示
        self._token_edit.setEchoMode(QLineEdit.EchoMode.Password)
        form.addRow(self.tr("Address"), self._url_edit)
        form.addRow(self.tr("Token"), self._token_edit)

        row = QHBoxLayout()
        self._test_btn = QPushButton(self.tr("Test connection"))
        self._test_btn.clicked.connect(self._on_test_connection)
        self._test_label = QLabel()
        self._test_label.setWordWrap(True)
        row.addWidget(self._test_btn)
        row.addWidget(self._test_label, 1)
        form.addRow(row)
        return self._remote_box

    def _build_local_group(self) -> QGroupBox:
        self._local_box = QGroupBox(self.tr("Local backend"))
        self._local_box.setObjectName("setting_group")
        group_layout = QVBoxLayout(self._local_box)

        dir_row = QHBoxLayout()
        self._dir_edit = QLineEdit()
        self._dir_edit.setPlaceholderText(str(local_backend.default_install_dir()))
        self._dir_edit.textChanged.connect(self._refresh_install_status)
        browse_btn = QPushButton(self.tr("Browse..."))
        browse_btn.clicked.connect(self._on_browse_dir)
        dir_row.addWidget(QLabel(self.tr("Install directory")))
        dir_row.addWidget(self._dir_edit, 1)
        dir_row.addWidget(browse_btn)
        group_layout.addLayout(dir_row)

        self._install_status = QLabel()
        self._install_status.setWordWrap(True)
        group_layout.addWidget(self._install_status)

        btn_row = QHBoxLayout()
        self._download_btn = QPushButton(self.tr("Download and install"))
        self._download_btn.clicked.connect(self._on_download_install)
        self._zip_btn = QPushButton(self.tr("Install from local zip..."))
        self._zip_btn.clicked.connect(self._on_zip_install)
        btn_row.addWidget(self._download_btn)
        btn_row.addWidget(self._zip_btn)
        btn_row.addStretch()
        group_layout.addLayout(btn_row)

        note = QLabel(self.tr(
            "The first install downloads PyTorch (about 3 GB) and needs an internet connection."))
        note.setObjectName("setting_hint")
        note.setWordWrap(True)
        group_layout.addWidget(note)

        self._progress = InstallProgress()
        group_layout.addWidget(self._progress)
        return self._local_box

    # ── 连接方式 ──

    def _on_mode_changed(self):
        remote = self._remote_radio.isChecked()
        self._remote_box.setEnabled(remote)
        self._local_box.setEnabled(not remote)
        self.collect()

    def _on_test_connection(self):
        url = self._url_edit.text().strip()
        if not url:
            self._test_label.setText(self.tr("Enter an address first."))
            return
        self._test_btn.setEnabled(False)
        self._test_label.setText(self.tr("Connecting..."))
        self._probe_worker = LibraryTaskWorker(probe_backend, url, self._token_edit.text().strip())
        self._probe_worker.finished_ok.connect(self._on_probe_done)
        self._probe_worker.error.connect(self._on_probe_error)
        self._probe_worker.start()

    def _on_probe_done(self, status: BackendStatus):
        self._test_btn.setEnabled(True)
        messages = {
            BackendStatus.OK: (self.tr("Connected."), "#4ec994"),
            BackendStatus.UNAUTHORIZED: (self.tr("Connected, but the token was rejected."),
                                         "#e5c07b"),
            BackendStatus.UNREACHABLE: (self.tr("Cannot connect to this address."), "#e06c75"),
        }
        text, color = messages[status]
        self._test_label.setStyleSheet(f"color: {color};")
        self._test_label.setText(text)

    def _on_probe_error(self, msg: str):
        self._test_btn.setEnabled(True)
        self._test_label.setStyleSheet("color: #e06c75;")
        self._test_label.setText(msg)

    # ── 本机安装 ──

    def _install_root(self) -> Path:
        text = self._dir_edit.text().strip()
        return Path(text) if text else local_backend.default_install_dir()

    def _refresh_install_status(self):
        root = self._install_root()
        if local_backend.is_installed(root):
            self._install_status.setStyleSheet("color: #4ec994;")
            self._install_status.setText(self.tr("Installed: {0}").format(root))
        else:
            self._install_status.setStyleSheet("color: #e5c07b;")
            self._install_status.setText(self.tr("Not installed: {0}").format(root))

    def _on_browse_dir(self):
        path = QFileDialog.getExistingDirectory(self, self.tr("Choose install directory"),
                                                str(self._install_root()))
        if path:
            self._dir_edit.setText(path)

    def _on_download_install(self):
        self._start_install(None)

    def _on_zip_install(self):
        path, _ = QFileDialog.getOpenFileName(
            self, self.tr("Choose backend package"), "", "veluxia-backend (*.zip)")
        if path:
            self._start_install(Path(path))

    def _start_install(self, archive):
        self.collect()
        self.install.emit()
        self._set_install_enabled(False)
        self._progress.start(self.tr("Preparing to install into {0}…").format(self._install_root()))
        self._install_worker = BackendInstallWorker(self._install_root(), archive)
        self._install_worker.progress.connect(self._progress.append)
        self._install_worker.finished_ok.connect(self._on_install_done)
        self._install_worker.error.connect(self._on_install_error)
        self._install_worker.start()

    def _set_install_enabled(self, enabled: bool):
        self._download_btn.setEnabled(enabled)
        self._zip_btn.setEnabled(enabled)
        self._dir_edit.setEnabled(enabled)

    def _on_install_done(self):
        self._set_install_enabled(True)
        self._progress.finish(True, self.tr("Installation complete"))
        self._refresh_install_status()
        self._config.save()  # 安装目录要落盘，重启后启动流程才找得到
        log_success(f"本机后端安装完成: {self._install_root()}")
        reply = QMessageBox.question(
            self, self.tr("Backend installed"),
            self.tr("The local backend is installed. Restart the app now to use it?"))
        if reply == QMessageBox.StandardButton.Yes:
            self.restart_requested.emit()

    def _on_install_error(self, msg: str):
        self._set_install_enabled(True)
        self._refresh_install_status()
        self._progress.finish(False, msg)
        log_error(f"本机后端安装失败: {msg}")

    # ── 配置读写 ──

    def load(self):
        server = self._config.get_section("server")
        if server.get("mode") == "remote":
            self._remote_radio.setChecked(True)
        else:
            self._local_radio.setChecked(True)
        self._url_edit.setText(server.get("url", ""))
        self._token_edit.setText(server.get("token", ""))
        self._dir_edit.setText(server.get("install_dir", ""))
        self._remote_box.setEnabled(self._remote_radio.isChecked())
        self._local_box.setEnabled(self._local_radio.isChecked())

    def collect(self):
        self._config.set("server", "mode", "remote" if self._remote_radio.isChecked() else "local")
        self._config.set("server", "url", self._url_edit.text().strip())
        self._config.set("server", "token", self._token_edit.text().strip())
        self._config.set("server", "install_dir", self._dir_edit.text().strip())
