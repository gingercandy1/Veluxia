"""一段的图层面板：左边预览（按眼睛叠出可见的层），右边图层列表（合并 / 分层两组）。

只负责展示和收集操作；图从外面传进来，合并、删除交给页面去调后端。叠放顺序规则在
src.shared.layer_stack，和后端合并时一致。

列表用逐行控件而不是 QTreeWidget：树的子行有缩进、段图行没有，眼睛列对不齐；
点眼睛重建整棵树还会丢选中和滚动位置。"""
import tempfile
from pathlib import Path

from PySide6.QtCore import QRect, QSize, Qt, Signal
from PySide6.QtGui import QColor, QIcon, QImage, QPainter, QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from src.shared.layer_stack import stack_key
from src.shared.schemas import MergedLayer

REFERENCE_ID = "ref"
ROW_THUMB = QSize(72, 44)
ROW_HEIGHT = 52
EYE_BUTTON = 30
EYE_ICON = QSize(20, 20)
CHECKER = 10
CANVAS_MIN_HEIGHT = 160
LIST_MIN_WIDTH = 250


class LayerCanvas(QWidget):
    """棋盘格底上按从底到顶叠出可见的层；宽度铺满、高度按段图比例，双击放大。"""
    composite_opened = Signal(QImage)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("ring_layer_canvas")
        self.setMinimumHeight(CANVAS_MIN_HEIGHT)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._images: list[QImage] = []

    def set_images(self, images: list[QImage]):
        self._images = [image for image in images if not image.isNull()]
        self.update()

    def composite(self) -> QImage:
        """按原始尺寸叠出的结果，双击放大时用。"""
        if not self._images:
            return QImage()
        result = QImage(self._images[0].size(), QImage.Format.Format_ARGB32)
        result.fill(Qt.GlobalColor.transparent)
        painter = QPainter(result)
        for image in self._images:
            painter.drawImage(0, 0, image)
        painter.end()
        return result

    def _target(self) -> QRect:
        """图在控件里的位置：按比例放进控件，居中。"""
        if not self._images:
            return QRect()
        size = self._images[0].size().scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio)
        return QRect((self.width() - size.width()) // 2, (self.height() - size.height()) // 2,
                     size.width(), size.height())

    def paintEvent(self, _event):
        target = self._target()
        if target.isNull():
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.setClipRect(target)
        for row in range(target.height() // CHECKER + 1):
            for column in range(target.width() // CHECKER + 1):
                shade = 58 if (row + column) % 2 else 74
                painter.fillRect(target.left() + column * CHECKER, target.top() + row * CHECKER,
                                 CHECKER, CHECKER, QColor(shade, shade, shade))
        painter.setClipping(False)
        for image in self._images:
            painter.drawImage(target, image)

    def mouseDoubleClickEvent(self, event):
        if self._images:
            self.composite_opened.emit(self.composite())
        super().mouseDoubleClickEvent(event)


class LayerRow(QFrame):
    """一行：眼睛 + 缩略图 + 名字。点行本身选中，点眼睛只切换显示。"""
    clicked = Signal(str, object)          # 层编号, 键盘修饰键
    eye_toggled = Signal(str)
    opened = Signal(str)                   # 层编号

    def __init__(self, layer_id: str, name: str, selectable: bool, parent=None):
        super().__init__(parent)
        self.layer_id = layer_id
        self.setObjectName("ring_layer_row")
        self.setFixedHeight(ROW_HEIGHT)
        self.setProperty("selected", False)
        self.setProperty("selectable", selectable)
        self.setCursor(Qt.CursorShape.PointingHandCursor if selectable
                       else Qt.CursorShape.ArrowCursor)

        self.eye = QToolButton()
        self.eye.setObjectName("ring_layer_eye")
        self.eye.setFixedSize(EYE_BUTTON, EYE_BUTTON)
        self.eye.setIconSize(EYE_ICON)
        self.eye.setToolTip(self.tr("Show or hide this layer in the preview"))
        self.eye.setCursor(Qt.CursorShape.PointingHandCursor)
        self.eye.clicked.connect(lambda: self.eye_toggled.emit(self.layer_id))
        self.thumb = QLabel()
        self.thumb.setObjectName("ring_layer_row_thumb")
        self.thumb.setFixedSize(ROW_THUMB)
        self.thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.thumb.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.label = QLabel(name)
        self.label.setObjectName("ring_layer_row_name")
        self.label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setToolTip(self.tr("Click to select · Ctrl/Shift to select several · "
                                "double-click to enlarge"))

        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 3, 8, 3)
        layout.setSpacing(8)
        layout.addWidget(self.eye)
        layout.addWidget(self.thumb)
        layout.addWidget(self.label, 1)

    def set_visible_eye(self, visible: bool):
        self.eye.setIcon(QIcon(":/svg/eye.svg" if visible else ":/svg/eye_off.svg"))

    def set_selected(self, selected: bool):
        if self.property("selected") == selected:
            return
        self.setProperty("selected", selected)
        self.style().unpolish(self)
        self.style().polish(self)

    def set_image(self, image: QImage | None):
        if image is None or image.isNull():
            self.thumb.clear()
            return
        # 按设备像素比放大再标注比例：高分屏上缩略图才不糊
        ratio = self.devicePixelRatioF()
        pixmap = QPixmap.fromImage(image).scaled(
            ROW_THUMB * ratio, Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation)
        pixmap.setDevicePixelRatio(ratio)
        self.thumb.setPixmap(pixmap)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self.layer_id, event.modifiers())
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.opened.emit(self.layer_id)
        super().mouseDoubleClickEvent(event)


class GroupHeader(QPushButton):
    """分组标题：点一下折叠 / 展开这一组。"""

    def __init__(self, title: str, folded: bool, parent=None):
        super().__init__(parent)
        self._title = title
        self.setObjectName("ring_layer_group")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.set_folded(folded)

    def set_folded(self, folded: bool):
        self.setText(f"{'▸' if folded else '▾'}  {self._title}")


class LayerPanel(QWidget):
    """merge_btn / delete_btn 由使用方摆在按钮行里（紧挨着“重做此段”和“分层”），面板只管它们的启用。"""
    merge_requested = Signal(list)         # 选中的层编号
    delete_requested = Signal(str)         # 要删除的合并层编号
    layer_opened = Signal(str)             # 双击一行：该层文件的相对路径
    image_opened = Signal(str)             # 双击预览：叠好的图存成的临时文件路径

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("ring_layer_stack")
        self._segment = -1
        self._reference = ""
        self._layers: list[str] = []
        self._merged: list[MergedLayer] = []
        self._images: dict[str, QImage] = {}
        self._hidden: set[str] = set()
        self._selected: list[str] = []
        self._anchor = ""
        self._idle = True
        self._rows: dict[str, LayerRow] = {}
        self._order: list[str] = []            # 列表里从上到下的可选层
        self._groups: list[tuple[str, list[str]]] = []
        self._collapsed: set[str] = set()
        # 叠好的图放进临时目录再交给查看窗口；目录随面板销毁一起清掉
        self._tmp = tempfile.TemporaryDirectory(prefix="veluxia_layers_")
        self._tmp_count = 0

        self.canvas = LayerCanvas()
        self.canvas.composite_opened.connect(self._open_composite)

        self.scroll = QScrollArea()
        self.scroll.setObjectName("ring_layer_scroll")
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll.setMinimumWidth(LIST_MIN_WIDTH)
        # 4 层 + 合并层 + 段图，至少露出五行再滚动
        self.scroll.setMinimumHeight(ROW_HEIGHT * 5)
        self.merge_btn = QPushButton(self.tr("Merge"))
        self.merge_btn.setToolTip(self.tr("Merge the selected layers into a new layer"))
        self.delete_btn = QPushButton(self.tr("Delete merged layer"))
        self.merge_btn.clicked.connect(lambda: self.merge_requested.emit(self.selected_ids()))
        self.delete_btn.clicked.connect(self._emit_delete)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        layout.addWidget(self.canvas, 3)
        layout.addWidget(self.scroll, 2)
        self._update_buttons()

    # ---- 对外 ----
    def set_segment(self, segment: int, reference: str, layers: list[str],
                    merged: list[MergedLayer]):
        """换了一段就让所有眼睛恢复睁开、清掉选中；同一段刷新（如刚合并出一层）保留。"""
        if segment != self._segment:
            self._hidden, self._selected, self._anchor = set(), [], ""
        self._segment, self._reference = segment, reference
        self._layers, self._merged = list(layers), list(merged)
        alive = {REFERENCE_ID, *self._ids()}
        self._hidden &= alive
        self._selected = [layer_id for layer_id in self._selected if layer_id in alive]
        self._rebuild()

    def set_images(self, images: dict[str, QImage]):
        """键是相对路径；缩略图和预览都用这份。"""
        self._images = images
        for layer_id, row in self._rows.items():
            row.set_image(self._images.get(self._path(layer_id)))
        self._refresh_canvas()

    def set_idle(self, idle: bool):
        """后端在做别的长任务时不能改图层。"""
        self._idle = idle
        self._update_buttons()

    def selected_ids(self) -> list[str]:
        """选中的层，按列表从上到下的顺序。"""
        return [layer_id for layer_id in self._order if layer_id in self._selected]

    def visible_ids(self) -> list[str]:
        """预览里叠着的层，从底到顶；段图排在最底下。"""
        ids = [] if REFERENCE_ID in self._hidden or not self._reference else [REFERENCE_ID]
        return ids + [layer_id for layer_id in self._stack() if layer_id not in self._hidden]

    def group_layout(self) -> list[tuple[str, list[str]]]:
        """(分组标题, 该组从上到下的层编号)。"""
        return list(self._groups)

    def row(self, layer_id: str) -> LayerRow | None:
        return self._rows.get(layer_id)

    def select(self, layer_ids: list[str]):
        self._selected = [layer_id for layer_id in layer_ids if layer_id in self._order]
        self._anchor = self._selected[-1] if self._selected else ""
        self._sync_selection()

    def toggle_visible(self, layer_id: str):
        self._hidden ^= {layer_id}
        row = self._rows.get(layer_id)
        if row is not None:
            row.set_visible_eye(layer_id not in self._hidden)
        self._refresh_canvas()

    # ---- 内部 ----
    def _ids(self) -> list[str]:
        return [f"L{number}" for number in range(1, len(self._layers) + 1)] + [
            merged.id for merged in self._merged]

    def _info(self) -> dict[str, tuple[str, int]]:
        """层编号 -> (文件, 叠放位置)。"""
        info = {f"L{number}": (path, number) for number, path in enumerate(self._layers, start=1)}
        info.update({merged.id: (merged.path, merged.z) for merged in self._merged})
        return info

    def _stack(self) -> list[str]:
        """从底到顶。"""
        info = self._info()
        return sorted(info, key=lambda layer_id: stack_key(layer_id, info[layer_id][1]))

    def _path(self, layer_id: str) -> str:
        return self._reference if layer_id == REFERENCE_ID else self._info()[layer_id][0]

    def _rebuild(self):
        scroll_value = self.scroll.verticalScrollBar().value()
        body = QWidget()
        column = QVBoxLayout(body)
        column.setContentsMargins(0, 0, 4, 0)
        column.setSpacing(2)
        self._rows, self._order, self._groups = {}, [], []
        info = self._info()
        stack = self._stack()
        for title, kind in ((self.tr("Merged"), "M"), (self.tr("Layers"), "L")):
            ids = [layer_id for layer_id in reversed(stack) if layer_id.startswith(kind)]
            if not ids:
                continue
            self._groups.append((title, ids))
            group_body = QWidget()
            group_column = QVBoxLayout(group_body)
            group_column.setContentsMargins(0, 0, 0, 0)
            group_column.setSpacing(2)
            for layer_id in ids:
                group_column.addWidget(self._make_row(layer_id, info))
            folded = title in self._collapsed
            header = GroupHeader(title, folded, body)
            group_body.setVisible(not folded)
            header.clicked.connect(
                lambda _=False, t=title, h=header, w=group_body: self._toggle_fold(t, h, w))
            column.addWidget(header)
            column.addWidget(group_body)
            self._order.extend(ids)
        if self._reference:
            row = LayerRow(REFERENCE_ID, self.tr("Segment image"), selectable=False)
            self._wire(row)
            self._rows[REFERENCE_ID] = row
            column.addWidget(row)
        column.addStretch(1)
        self.scroll.setWidget(body)
        self.scroll.verticalScrollBar().setValue(scroll_value)
        for layer_id, row in self._rows.items():
            row.set_visible_eye(layer_id not in self._hidden)
            row.set_image(self._images.get(self._path(layer_id)))
        self._sync_selection()
        self._refresh_canvas()

    def _toggle_fold(self, title: str, header: GroupHeader, body: QWidget):
        self._collapsed ^= {title}
        folded = title in self._collapsed
        header.set_folded(folded)
        body.setVisible(not folded)

    def _make_row(self, layer_id: str, info: dict[str, tuple[str, int]]) -> LayerRow:
        merged = next((m for m in self._merged if m.id == layer_id), None)
        if merged:
            name = (self.tr("Merged {0}").format(layer_id[1:]) + "\n"
                    + self.tr("from {0}").format(" + ".join(merged.sources)))
        else:
            name = self.tr("Layer {0}").format(layer_id[1:])
        row = LayerRow(layer_id, name, selectable=True)
        self._wire(row)
        self._rows[layer_id] = row
        return row

    def _wire(self, row: LayerRow):
        row.clicked.connect(self._on_row_clicked)
        row.eye_toggled.connect(self.toggle_visible)
        row.opened.connect(self._on_row_opened)

    def _on_row_clicked(self, layer_id: str, modifiers):
        if layer_id == REFERENCE_ID:
            return
        if modifiers & Qt.KeyboardModifier.ShiftModifier and self._anchor in self._order:
            start, end = sorted((self._order.index(self._anchor), self._order.index(layer_id)))
            self._selected = self._order[start:end + 1]
        elif modifiers & Qt.KeyboardModifier.ControlModifier:
            self._selected = ([i for i in self._selected if i != layer_id]
                              if layer_id in self._selected else [*self._selected, layer_id])
            self._anchor = layer_id
        else:
            self._selected, self._anchor = [layer_id], layer_id
        self._sync_selection()

    def _on_row_opened(self, layer_id: str):
        path = self._path(layer_id)
        if path:
            self.layer_opened.emit(path)

    def _sync_selection(self):
        for layer_id, row in self._rows.items():
            row.set_selected(layer_id in self._selected)
        self._update_buttons()

    def _refresh_canvas(self):
        images = [self._images.get(self._path(layer_id)) for layer_id in self.visible_ids()]
        self.canvas.set_images([image for image in images if image is not None])

    def _emit_delete(self):
        merged = [layer_id for layer_id in self.selected_ids() if layer_id.startswith("M")]
        if len(merged) == 1:
            self.delete_requested.emit(merged[0])

    def _update_buttons(self):
        ids = self.selected_ids()
        self.merge_btn.setEnabled(self._idle and len(ids) >= 2)
        self.delete_btn.setEnabled(
            self._idle and len(ids) == 1 and ids[0].startswith("M"))

    def _open_composite(self, image: QImage):
        self._tmp_count += 1
        path = Path(self._tmp.name) / f"segment_{self._tmp_count}.png"
        image.save(str(path))
        self.image_opened.emit(str(path))
