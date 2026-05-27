import json
import os

from PySide6.QtCore import Qt, QAbstractTableModel, QModelIndex
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QTableView, QPushButton,
    QHeaderView, QAbstractItemView,
    QMessageBox, QFrame
)

from src.shared.settings import PROJECT_ROOT
from src.app.ui.setting.page.log_page import log_success, log_error
MODELS_PATH = os.path.join(PROJECT_ROOT, "models.json")

# 表格列定义
COLUMNS = ["类别", "模型名称", "repo_id", "filename", "tag", "note"]
COL_MAP  = ["_category", "name", "repo_id", "filename", "tag", "note"]

class _ModelTableModel(QAbstractTableModel):
    """
    把 models.json 的嵌套结构展平成表格行：
    { "text": { "ModelA": {...}, "ModelB": {...} } }
    → 每行: [category, name, repo_id, filename, tag, note]
    """

    def __init__(self, data: dict, parent=None):
        super().__init__(parent)
        self._raw   = data
        self._rows: list[dict] = []
        self._dirty_rows: set[int] = set()
        self._flatten()

    def _flatten(self):
        self._rows.clear()
        for category, models in self._raw.items():
            if not isinstance(models, dict):
                continue
            for name, info in models.items():
                if not isinstance(info, dict):
                    continue
                self._rows.append({
                    "_category": category,
                    "name":      name,
                    "repo_id":   info.get("repo_id", ""),
                    "filename":  info.get("filename", ""),
                    "tag":       info.get("tag", ""),
                    "note":      info.get("note", ""),
                })

    def rowCount(self, parent=QModelIndex()):
        return len(self._rows)

    def columnCount(self, parent=QModelIndex()):
        return len(COLUMNS)

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole:
            if orientation == Qt.Orientation.Horizontal:
                return COLUMNS[section]
        return None

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        row = self._rows[index.row()]
        col = COL_MAP[index.column()]

        if role == Qt.ItemDataRole.DisplayRole:
            return row.get(col, "")

        if role == Qt.ItemDataRole.EditRole:
            return row.get(col, "")

        # 类别列背景色区分
        if role == Qt.ItemDataRole.BackgroundRole:
            if index.column() == 0:
                from PySide6.QtGui import QColor
                colors = {
                    "text":       "#2d3250",
                    "image":      "#2d4a3e",
                    "animation":  "#3d2d4a",
                    "speech":     "#4a3d2d",
                    "image_frame":"#2d404a",
                }
                return QColor(colors.get(row["_category"], "#2d2d2d"))

        # 已修改行高亮
        if role == Qt.ItemDataRole.ForegroundRole:
            if index.row() in self._dirty_rows:
                from PySide6.QtGui import QColor
                return QColor("#e5c07b")

        return None

    def flags(self, index):
        base = Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
        # 类别列和模型名不可编辑
        if index.column() in (0, 1):
            return base
        return base | Qt.ItemFlag.ItemIsEditable

    def setData(self, index, value, role=Qt.ItemDataRole.EditRole):
        if not index.isValid() or role != Qt.ItemDataRole.EditRole:
            return False
        col = COL_MAP[index.column()]
        self._rows[index.row()][col] = value
        self._dirty_rows.add(index.row())
        self.dataChanged.emit(index, index)
        return True

    def to_dict(self) -> dict:
        result: dict = {}
        for row in self._rows:
            cat  = row["_category"]
            name = row["name"]
            if cat not in result:
                result[cat] = {}
            result[cat][name] = {
                "repo_id":  row["repo_id"],
                "filename": row["filename"],
                "tag":      row["tag"],
                "note":     row["note"],
            }
        return result

    @property
    def is_dirty(self) -> bool:
        return bool(self._dirty_rows)

    def mark_clean(self):
        self._dirty_rows.clear()
        self.layoutChanged.emit()


class ModelPage(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("model_page")
        self._table_model = None
        self._build_ui()
        self.load()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(32, 24, 32, 24)
        layout.setSpacing(12)

        # 标题 + 说明
        title = QLabel("模型设置")
        title.setObjectName("page_title")
        layout.addWidget(title)

        hint = QLabel("双击单元格可编辑 repo_id / filename / tag / note，类别和名称不可修改。")
        hint.setObjectName("page_hint")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setObjectName("page_separator")
        layout.addWidget(sep)

        # 表格
        self._table_view = QTableView()
        self._table_view.setObjectName("model_table")
        self._table_view.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self._table_view.setAlternatingRowColors(True)
        self._table_view.setEditTriggers(
            QAbstractItemView.EditTrigger.DoubleClicked
        )
        self._table_view.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )
        self._table_view.horizontalHeader().setStretchLastSection(True)
        self._table_view.verticalHeader().setVisible(False)
        layout.addWidget(self._table_view, 1)

        # 底部按钮
        btn_row = QHBoxLayout()
        self._reload_btn = QPushButton("重新加载")
        self._reload_btn.setFixedWidth(90)
        self._reload_btn.clicked.connect(self.load)
        btn_row.addWidget(self._reload_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

    def load(self):
        try:
            with open(MODELS_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            QMessageBox.warning(self, "加载失败", f"无法读取 models.json:\n{e}")
            return

        self._table_model = _ModelTableModel(data)
        self._table_view.setModel(self._table_model)

    def collect(self):
        if self._table_model is None or not self._table_model.is_dirty:
            return
        try:
            with open(MODELS_PATH, "w", encoding="utf-8") as f:
                json.dump(
                    self._table_model.to_dict(), f,
                    ensure_ascii=False, indent=2
                )
            self._table_model.mark_clean()
            log_success("models.json 已保存")
        except Exception as e:
            log_error(f"models.json 保存失败: {e}")