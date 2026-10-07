from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from src.app.client import ApiClient
from src.app.work import LibraryTaskWorker
from src.shared.schemas import ModelStorageEntry, ModelStorageResponse
from src.shared.settings import ConfigManager

_UNITS = ("B", "KB", "MB", "GB", "TB")


def format_size(size: int) -> str:
    value = float(size)
    for unit in _UNITS:
        if value < 1024 or unit == _UNITS[-1]:
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{size} B"


class StoragePage(QWidget):
    """模型文件占了多少磁盘、删掉不用的。数据来自后端：远程后端时看的是 GPU 机器上的磁盘。
    首次切到这一页才统计，遍历几百 GB 的目录要一会儿，不拖慢打开设置。"""
    COL_LOCATION, COL_NAME, COL_SIZE, COL_ACTION = range(4)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("storage_page")
        self._client = ApiClient.instance()
        self._workers: set[LibraryTaskWorker] = set()
        self._loaded = False
        self._busy = False
        self._entries: list[ModelStorageEntry] = []
        self._buttons: dict[int, QPushButton] = {}
        self._config = ConfigManager()
        self._build_ui()
        self._dir_edit.setText(self._config.get("models", "dir", ""))
        self._dir_edit.textChanged.connect(self.collect)

    def _build_folder_group(self) -> QGroupBox:
        """模型存放目录：权重一两百 GB，常常要放到别的盘上。只影响本机后端，远程后端在它自己的机器上配置。"""
        group = QGroupBox(self.tr("Model folder"))
        group.setObjectName("setting_group")
        group_layout = QVBoxLayout(group)
        row = QHBoxLayout()
        self._dir_edit = QLineEdit()
        self._dir_edit.setPlaceholderText(self.tr("Default: the models folder of the backend"))
        browse_btn = QPushButton(self.tr("Browse..."))
        browse_btn.clicked.connect(self._on_browse_dir)
        row.addWidget(self._dir_edit, 1)
        row.addWidget(browse_btn)
        group_layout.addLayout(row)
        note = QLabel(self.tr(
            "Models are read from and downloaded into this folder. Leave it empty to use the "
            "default. Takes effect after restarting the app. Models you already downloaded are "
            "not moved: move the old models folder here by hand to keep them. Only applies to "
            "the backend on this computer."))
        note.setObjectName("setting_hint")
        note.setWordWrap(True)
        group_layout.addWidget(note)
        return group

    def _on_browse_dir(self):
        path = QFileDialog.getExistingDirectory(
            self, self.tr("Choose model folder"), self._dir_edit.text().strip())
        if path:
            self._dir_edit.setText(path)

    def collect(self):
        self._config.set("models", "dir", self._dir_edit.text().strip())

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(32, 24, 32, 24)
        layout.setSpacing(12)

        title = QLabel(self.tr("Model Storage"))
        title.setObjectName("page_title")
        layout.addWidget(title)
        hint = QLabel(self.tr(
            "Disk space used by downloaded model weights. A deleted model downloads again "
            "the next time you use it."))
        hint.setObjectName("page_hint")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setObjectName("page_separator")
        layout.addWidget(sep)
        layout.addWidget(self._build_folder_group())

        row = QHBoxLayout()
        self.total_label = QLabel()
        self.total_label.setObjectName("storage_total")
        row.addWidget(self.total_label, 1)
        self.refresh_btn = QPushButton(self.tr("Refresh"))
        self.refresh_btn.setMinimumWidth(90)
        self.refresh_btn.clicked.connect(self.refresh)
        row.addWidget(self.refresh_btn)
        layout.addLayout(row)

        self.error_label = QLabel()
        self.error_label.setObjectName("storage_error")
        self.error_label.setWordWrap(True)
        self.error_label.hide()
        layout.addWidget(self.error_label)

        self.table = QTableWidget(0, 4)
        self.table.setObjectName("model_table")
        self.table.setHorizontalHeaderLabels(
            [self.tr("Type"), self.tr("Name"), self.tr("Size"), ""])
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        # 行里放了删除按钮，默认行高放不下
        self.table.verticalHeader().setDefaultSectionSize(38)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(self.COL_NAME, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(self.COL_ACTION, QHeaderView.ResizeMode.Fixed)
        self.table.setColumnWidth(self.COL_ACTION, 96)
        layout.addWidget(self.table, 1)

    # ---- 数据 ----
    def showEvent(self, event):
        super().showEvent(event)
        if not self._loaded:
            self.refresh()

    def refresh(self):
        if self._busy:
            return
        self._set_busy(True, self.tr("Calculating…"))
        self._start(self._client.list_model_storage)

    def _delete(self, entry: ModelStorageEntry):
        if self._busy:
            return
        message = self.tr("Delete \"{0}\" ({1})?").format(entry.name, format_size(entry.size))
        if entry.manual:
            message += "\n\n" + self.tr(
                "This model is NOT downloaded automatically. You will have to put the files "
                "back by hand before frame interpolation works again.")
        else:
            message += "\n\n" + self.tr("It downloads again the next time you use it.")
        reply = QMessageBox.question(
            self, self.tr("Delete model files"), message,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No)
        if reply != QMessageBox.StandardButton.Yes:
            return
        self._set_busy(True, self.tr("Deleting {0}…").format(entry.name))
        self._start(self._client.delete_model_storage, entry.id)

    def _on_response(self, response: ModelStorageResponse):
        self._set_busy(False)
        if not response.ok:
            self._show_error(response.error or "")
            # 删除失败时列表没变；首次统计失败下次切回来再试
            self.total_label.setText(self._total_text())
            return
        self._loaded = True
        self._show_error("")
        self._entries = response.entries
        self._fill_table()
        self.total_label.setText(self._total_text(response.total))

    def _total_text(self, total: int | None = None) -> str:
        if total is None:
            if not self._loaded:
                return ""
            total = sum(e.size for e in self._entries)
        return self.tr("Total: {0} in {1} items").format(format_size(total), len(self._entries))

    def _fill_table(self):
        self._buttons: dict[int, QPushButton] = {}
        self.table.setRowCount(len(self._entries))
        for row, entry in enumerate(self._entries):
            location = QTableWidgetItem(self._location(entry))
            name = QTableWidgetItem(entry.name)
            # 并了附属目录的条目把各目录列出来，删除时它们会一起删
            name.setToolTip("\n".join([entry.id, *entry.parts[1:]]))
            size = QTableWidgetItem(format_size(entry.size))
            size.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            for column, item in ((self.COL_LOCATION, location), (self.COL_NAME, name),
                                 (self.COL_SIZE, size)):
                self.table.setItem(row, column, item)
            button = QPushButton(self.tr("Delete"))
            button.setObjectName("storage_delete_btn")
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.setEnabled(not self._busy)
            button.clicked.connect(lambda _=False, e=entry: self._delete(e))
            # 包一层居中：按钮直接放进单元格会被拉满整格，样式表的 margin 在格子里不可靠
            cell = QWidget()
            cell_layout = QHBoxLayout(cell)
            cell_layout.setContentsMargins(0, 0, 0, 0)
            cell_layout.addWidget(button, 0, Qt.AlignmentFlag.AlignCenter)
            self.table.setCellWidget(row, self.COL_ACTION, cell)
            self._buttons[row] = button

    def _location(self, entry: ModelStorageEntry) -> str:
        if entry.root == "ace_step":
            return self.tr("Music (ACE-Step)")
        if entry.root == "rembg":
            return self.tr("Background removal")
        labels = {
            "text": self.tr("Text"),
            "image": self.tr("Image"),
            "speech": self.tr("Voice and audio"),
            "animation": self.tr("Animation"),
            "image_frame": self.tr("Frame interpolation"),
            "transcription": self.tr("Speech-to-text"),
            "upscale": self.tr("Upscaling"),
            "lora": "LoRA",
            "prompt_refiner": self.tr("Prompt optimizer"),
        }
        return labels.get(entry.category, self.tr("Other"))

    # ---- 状态 ----
    def _set_busy(self, busy: bool, message: str = ""):
        self._busy = busy
        self.refresh_btn.setEnabled(not busy)
        for button in self._buttons.values():
            button.setEnabled(not busy)
        if busy:
            self.total_label.setText(message)

    def _show_error(self, message: str):
        self.error_label.setText(message)
        self.error_label.setVisible(bool(message))

    def _start(self, fn, *args):
        worker = LibraryTaskWorker(fn, *args)
        worker.finished_ok.connect(self._on_response)
        worker.error.connect(lambda message: self._on_response(
            ModelStorageResponse.from_error(message)))
        worker.finished.connect(lambda: self._workers.discard(worker))
        self._workers.add(worker)
        worker.start()
