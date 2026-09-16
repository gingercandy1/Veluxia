from typing import Optional

from PySide6.QtCore import (
    Qt, Signal, QModelIndex, QSize, QPropertyAnimation,
    QEasingCurve, QPoint, Property
)
from PySide6.QtGui import (
    QStandardItemModel, QStandardItem, QPainter, QColor,
    QPainterPath, QFont, QPaintEvent, QPolygon
)
from PySide6.QtWidgets import (
    QComboBox, QListView, QStyledItemDelegate, QWidget,
    QStyleOptionViewItem, QApplication, QStyle,
    QHBoxLayout, QSizePolicy, QFrame, QStyleOptionComboBox
)

# UserRole
ROLE_IS_GROUP  = Qt.ItemDataRole.UserRole + 20
ROLE_MODEL_KEY = Qt.ItemDataRole.UserRole + 21
ROLE_NOTE      = Qt.ItemDataRole.UserRole + 22
ROLE_TAGS      = Qt.ItemDataRole.UserRole + 23

ITEM_H  = 48
GROUP_H = 26
MAX_POPUP_H = 500

class GroupedDelegate(QStyledItemDelegate):

    def paint(self, painter: QPainter,
              option: QStyleOptionViewItem,
              index: QModelIndex) -> None:
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        if index.data(ROLE_IS_GROUP):
            self._draw_group(painter, option.rect, index)
        else:
            self._draw_item(painter, option, option.rect, index)

        painter.restore()

    def _draw_group(self, painter, rect, index):
        background = QColor("#1c1f25").darker(150)
        # background.setAlpha(0)
        painter.fillRect(rect, background)

        # 标签文字背景块（让文字盖住线）
        text  = (index.data(Qt.DisplayRole) or "").upper()
        font  = QFont()
        font.setPointSize(8)
        font.setBold(True)
        font.setLetterSpacing(QFont.AbsoluteSpacing, 1.0)
        painter.setFont(font)

        fm = painter.fontMetrics()
        tw = fm.horizontalAdvance(text) + 12
        tx = rect.left() + 14
        ty = rect.top()
        painter.fillRect(tx - 4, ty, tw + 4, rect.height(), background)
        painter.setPen(QColor("#333333"))
        painter.drawText(rect.adjusted(14, 0, -8, 0),
                         Qt.AlignLeft | Qt.AlignVCenter, text)

    def _draw_item(self, painter, option, rect, index):
        is_selected = bool(option.state & QStyle.State_Selected)
        is_hovered  = bool(option.state & QStyle.State_MouseOver)

        if is_selected:
            bg = QColor("#1c1f25").lighter(130)
        elif is_hovered:
            bg = QColor("#1c1f25")
        else:
            bg = QColor("#1c1f25")
            bg.setAlpha(0)

        path = QPainterPath()
        path.addRoundedRect(rect.adjusted(4, 1, -4, -1), 6, 6)
        painter.fillPath(path, bg)

        # 选中蓝条
        if is_selected:
            bar = QPainterPath()
            bar.addRoundedRect(rect.left() + 4, rect.top() + 8,
                               3, rect.height() - 16, 2, 2)
            painter.fillPath(bar, QColor("#3b3b3b"))

        name = index.data(Qt.DisplayRole) or ""
        note = index.data(ROLE_NOTE) or ""

        # 模型名
        nf = QFont()
        nf.setPointSize(10)
        nf.setBold(is_selected)
        painter.setFont(nf)
        painter.setPen(QColor("#e2e2ef") if is_selected else QColor("#cbcbcb"))

        if note:
            painter.drawText(rect.adjusted(16, 4, -8, -rect.height() // 2),
                             Qt.AlignLeft | Qt.AlignVCenter, name)
            sf = QFont()
            sf.setPointSize(8)
            painter.setFont(sf)
            painter.setPen(QColor("#474747"))
            painter.drawText(rect.adjusted(16, rect.height() // 2, -8, -3),
                             Qt.AlignLeft | Qt.AlignVCenter, note)
        else:
            painter.drawText(rect.adjusted(16, 0, -8, 0),
                             Qt.AlignLeft | Qt.AlignVCenter, name)

    def sizeHint(self, option, index) -> QSize:
        if index.data(ROLE_IS_GROUP):
            return QSize(option.rect.width(), GROUP_H)
        note = index.data(ROLE_NOTE) or ""
        return QSize(option.rect.width(), ITEM_H + (6 if note else 0))


class _ListView(QListView):
    def hideEvent(self, event):
        pass

    def wheelEvent(self, event):
        # 滚轮直接驱动垂直滚动条
        delta = event.angleDelta().y()
        bar = self.verticalScrollBar()
        bar.setValue(bar.value() - delta // 120)
        event.accept()


class ModelComboBox(QComboBox):
    """
    分组模型下拉框
    """
    model_selected = Signal(str)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._all_models: list[dict] = []
        self._angle = 180
        self._is_popup_shown = False
        self._confirmed_row = -1
        self._setup_view()
        self._setup_animation()

    def _setup_view(self):
        self._std_model = QStandardItemModel(self)

        self._view = _ListView()
        self._view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._view.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self._view.setItemDelegate(GroupedDelegate())
        self._view.setModel(self._std_model)
        self._view.setMouseTracking(True)
        self._view.clicked.connect(self._on_item_clicked)

        self.setModel(self._std_model)
        self.setView(self._view)
        self.setMinimumWidth(240)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def _setup_animation(self):
        self._anim_angle = QPropertyAnimation(self, b"_angle_prop", self)
        self._anim_angle.setStartValue(180)
        self._anim_angle.setEndValue(360)
        self._anim_angle.setDuration(180)
        self._anim_angle.setEasingCurve(QEasingCurve.InOutCubic)

    # ── Property ────────────────────────────────
    def _get_angle(self): return self._angle
    def _set_angle(self, v):
        self._angle = v
        self.update()
    _angle_prop = Property(int, _get_angle, _set_angle)

    # ── 公开 API ─────────────────────────────────
    def add_model(self,
                  key: str,
                  display_name: Optional[str] = None,
                  tag: str = "small",
                  note: str = "") -> None:
        """添加单个模型条目"""
        self._all_models.append({
            "key":          key,
            "display_name": display_name or key,
            "tag":          tag,
            "note":         note,
        })
        self._rebuild()

    def add_models(self, models: list[dict]) -> None:
        """
        批量添加，每项格式：
        """
        self._all_models.extend(models)
        self._rebuild()

    def clear_models(self) -> None:
        self._all_models.clear()
        self._std_model.clear()

    def set_group_order(self, order: list[str]) -> None:
        """切换分组维度顺序，调用后自动刷新列表"""
        self._group_order = order
        self._rebuild()

    def current_model_key(self) -> str:
        idx = self.currentIndex()
        if idx < 0:
            return ""
        item = self._std_model.item(idx)
        if item and not item.data(ROLE_IS_GROUP):
            return item.data(ROLE_MODEL_KEY) or ""
        return ""

    # ── 内部重建 ─────────────────────────────────
    def _rebuild(self) -> None:
        saved_key = self.current_model_key()
        self._std_model.clear()

        # 聚合分组
        groups: dict[str, list[dict]] = {}
        for m in self._all_models:
            groups.setdefault(m["tag"], []).append(m)

        ordered = [g for g in groups]
        restore_row = -1

        for gk in ordered:
            # 分组标题
            header = QStandardItem(gk)
            header.setData(True, ROLE_IS_GROUP)
            header.setData("",   ROLE_MODEL_KEY)
            header.setFlags(Qt.NoItemFlags)
            self._std_model.appendRow(header)

            for m in groups[gk]:
                item = QStandardItem(m["display_name"])
                item.setData(False,       ROLE_IS_GROUP)
                item.setData(m["key"],    ROLE_MODEL_KEY)
                item.setData(m["note"],   ROLE_NOTE)
                item.setData(m["tag"],   ROLE_TAGS)
                item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                self._std_model.appendRow(item)
                if m["key"] == saved_key:
                    restore_row = self._std_model.rowCount() - 1

        if restore_row >= 0:
            self.setCurrentIndex(restore_row)
            self._confirmed_row = restore_row
        else:
            self._select_first_valid()

    def _select_first_valid(self) -> None:
        for row in range(self._std_model.rowCount()):
            item = self._std_model.item(row)
            if item and not item.data(ROLE_IS_GROUP):
                self.setCurrentIndex(row)
                self._confirmed_row = row
                return

    def _on_item_clicked(self, index: QModelIndex) -> None:
        if index.data(ROLE_IS_GROUP):
            return

        key = index.data(ROLE_MODEL_KEY) or ""
        if not key:
            return

        self._confirmed_row = index.row()
        self.setCurrentIndex(index.row())
        self.model_selected.emit(key)

    def showPopup(self) -> None:
        super().showPopup()
        self._is_popup_shown = True
        popup = self.findChild(QFrame)
        if popup:
            # 创建样式选项
            opt = QStyleOptionComboBox()
            self.initStyleOption(opt)

            # 获取 ComboBox 下拉框应该出现的位置（下方）
            rect = self.style().subControlRect(
                QStyle.CC_ComboBox,
                opt,
                QStyle.SC_ComboBoxListBoxPopup,
                self
            )

            global_pos = self.mapToGlobal(rect.bottomLeft())
            popup.move(global_pos.x(), global_pos.y() + 10)
            popup.setFixedWidth(self.width())
            popup.setMaximumHeight(MAX_POPUP_H + 8)

            # ← 让 popup 容器的滚轮也转发给 _view
            popup.wheelEvent = lambda e: self._view.wheelEvent(e)
        # 窗口样式设置
        self._anim_angle.setDirection(QPropertyAnimation.Backward)
        if self._anim_angle.state() != QPropertyAnimation.Running:
            self._anim_angle.start()

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        path = QPainterPath()
        path.addRoundedRect(self.rect().adjusted(1, 1, -1, -1), 8, 8)
        painter.fillPath(path, QColor("#1c1f25"))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawPath(path)

        # 当前选中文字
        key     = self.current_model_key()
        display = next((m["display_name"] for m in self._all_models
                        if m["key"] == key), "")

        f = QFont()
        f.setPointSize(10)
        painter.setFont(f)
        painter.setPen(QColor("#e2e8e2") if display else QColor("#474747"))
        painter.drawText(self.rect().adjusted(12, 0, -32, 0),
                         Qt.AlignLeft | Qt.AlignVCenter,
                         display or "Choose model...")

        # 箭头
        cx, cy = self.width() - 18, self.height() // 2
        painter.translate(cx, cy)
        painter.rotate(self._angle - 180)
        painter.translate(-cx, -cy)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor("#64748b"))
        painter.drawPolygon(QPolygon([
            QPoint(cx - 4, cy - 2),
            QPoint(cx + 4, cy - 2),
            QPoint(cx,     cy + 3),
        ]))

    def enterEvent(self, event) -> None:
        self.setCursor(Qt.PointingHandCursor)

    def leaveEvent(self, event) -> None:
        self.unsetCursor()


if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    w = QWidget()
    w.setFixedSize(320, 60)
    w.setStyleSheet("background:#0d1117;")
    layout = QHBoxLayout(w)
    layout.setContentsMargins(16, 12, 16, 12)

    cb = ModelComboBox(w)

    cb.add_model("SmolLM2-135M", tag="tiny", note="135M，~90MB")
    cb.add_model("Qwen2.5-0.5B", tag="tiny", note="0.5B，~0.4GB")
    cb.add_model("Qwen2.5-1.5B", tag="small", note="1.5B，~1GB")
    cb.add_model("Llama-3.2-3B", tag="small", note="3B，~2GB")
    cb.add_model("Qwen3-8B",     tag="medium", note="8B，支持思考模式")
    cb.add_model("Llama-3.1-8B", tag="medium", note="8B，~5GB")
    cb.add_model("Qwen3-14B",    tag="large", note="14B，需 12GB VRAM")
    cb.add_model("Llama-3.3-70B", tag="xlarge", note="70B，需 48GB+ VRAM")

    cb.model_selected.connect(lambda k: print(f"选中: {k}"))

    layout.addWidget(cb)
    w.show()
    sys.exit(app.exec())