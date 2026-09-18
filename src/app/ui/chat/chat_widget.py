from PySide6.QtCore import Signal, Qt, QEvent, QTimer, QRect, QPoint, QPropertyAnimation, QEasingCurve, Property
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QScrollArea, QFrame

from src.app.ui.message.message_bubble import (
    MessageBubble, create_message_bubble, SpinnerWidget, FadeMask, SelectionDot,
)
from src.app.ui.setting.page.log_page import log_info


class SelectionColumn(QWidget):
    """聊天区左侧的多选列：每条消息对应一个圆形勾选框，宽度动画展开/收起。"""
    WIDTH = 36

    dot_toggled = Signal(str, bool)   # message_id, checked

    def __init__(self, parent=None):
        super().__init__(parent)
        self._dots: dict[str, SelectionDot] = {}
        self._col_width = 0
        self.setFixedWidth(0)

        self._anim = QPropertyAnimation(self, b"colWidth")
        self._anim.setDuration(220)
        self._anim.setEasingCurve(QEasingCurve.Type.OutCubic)

    def _get_col_width(self) -> int:
        return self._col_width

    def _set_col_width(self, w: int):
        self._col_width = w
        self.setFixedWidth(w)

    colWidth = Property(int, _get_col_width, _set_col_width)

    def add_dot(self, message_id: str) -> SelectionDot:
        dot = SelectionDot(self)
        dot.move((self.WIDTH - dot.width()) // 2, 0)
        dot.toggled.connect(lambda checked, mid=message_id: self.dot_toggled.emit(mid, checked))
        dot.show()
        self._dots[message_id] = dot
        return dot

    def remove_dot(self, message_id: str):
        dot = self._dots.pop(message_id, None)
        if dot is not None:
            dot.deleteLater()

    def clear_dots(self):
        for dot in self._dots.values():
            dot.deleteLater()
        self._dots.clear()

    def set_checked(self, message_id: str, checked: bool, animate: bool = True):
        dot = self._dots.get(message_id)
        if dot is not None:
            dot.setChecked(checked, animate=animate, emit=False)

    def toggle(self, message_id: str):
        dot = self._dots.get(message_id)
        if dot is not None:
            dot.toggle()

    def uncheck_all(self):
        for dot in self._dots.values():
            dot.setChecked(False, animate=False, emit=False)

    def place_dot(self, message_id: str, center_y: int):
        dot = self._dots.get(message_id)
        if dot is not None:
            dot.move(dot.x(), center_y - dot.height() // 2)

    def set_open(self, opened: bool):
        self._anim.stop()
        self._anim.setStartValue(self._col_width)
        self._anim.setEndValue(self.WIDTH if opened else 0)
        self._anim.start()


class ChatWidget(QScrollArea):
    """
    可滚动的聊天消息列表。
    维护一个 QWidget 作为内容容器，消息依次追加。
    """
    retry_requested = Signal(str)
    edit_requested = Signal(str, str)
    selection_mode_changed = Signal(bool)
    selection_changed = Signal(int)   # 当前选中条数

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("chat_widget")
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self._container = QWidget()
        outer = QHBoxLayout(self._container)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self._column = SelectionColumn()
        self._column.dot_toggled.connect(self._on_dot_toggled)
        self._content = QWidget()
        outer.addWidget(self._column)
        outer.addWidget(self._content, 1)

        self._layout = QVBoxLayout(self._content)
        self._layout.setContentsMargins(0, 16, 0, 16)
        self._layout.setSpacing(15)

        self._layout.addStretch()
        self._layout.addSpacing(20)

        self.setWidget(self._container)
        self._bubbles: dict[str, MessageBubble] = {}
        self._selection_mode = False
        self._selected_ids: set[str] = set()

        self._sync_timer = QTimer(self)
        self._sync_timer.setSingleShot(True)
        self._sync_timer.setInterval(0)
        self._sync_timer.timeout.connect(self._sync_column)
        self._content.installEventFilter(self)

        self._drag_origin: QPoint | None = None
        self._drag_active = False
        self._band = QFrame(self._content)
        self._band.setStyleSheet(
            "background-color: rgba(123, 157, 188, 0.15);"
            "border: 1px solid rgba(123, 157, 188, 0.6); border-radius: 3px;"
        )
        self._band.hide()

        fade_mask_color = QColor("#101215")
        fade_mask_color.setAlpha(255)
        self.fade_mask = FadeMask(fade_mask_color, self)
        self.fade_mask.set_reverse(False)
        self.fade_mask.setVisible(True)

    def add_message(self, role, content, timestamp, message_id=None):
        bubble = create_message_bubble(role=role, content=content, timestamp=timestamp, message_id=message_id)
        bubble.retry_requested.connect(self._on_retry_requested)
        bubble.content_edited.connect(self.on_message_edited)
        bubble.selection_clicked.connect(self._column.toggle)
        bubble.set_selection_mode(self._selection_mode)
        bubble.installEventFilter(self)

        self._bubbles[bubble.message_id] = bubble
        self._column.add_dot(bubble.message_id)
        self._layout.insertWidget(self._layout.count() - 2, bubble)
        self._sync_timer.start()
        return bubble

    def remove_message(self, message_id):
        self._bubbles[message_id].deleteLater()
        del self._bubbles[message_id]
        self._column.remove_dot(message_id)
        self._selected_ids.discard(message_id)
        self._sync_timer.start()

    def has_message(self, message_id) -> bool:
        return message_id in self._bubbles

    def detach_message(self, message_id):
        """把气泡从列表摘下但不销毁（生成中切换会话时保活），返回该气泡。"""
        bubble = self._bubbles.pop(message_id, None)
        if bubble is None:
            return None
        self._layout.removeWidget(bubble)
        bubble.hide()
        self._column.remove_dot(message_id)
        self._selected_ids.discard(message_id)
        self._sync_timer.start()
        return bubble

    def attach_bubble(self, bubble: MessageBubble):
        """把 detach_message 摘下的气泡重新追加到列表末尾。"""
        bubble.set_selection_mode(self._selection_mode)
        self._bubbles[bubble.message_id] = bubble
        self._column.add_dot(bubble.message_id)
        self._layout.insertWidget(self._layout.count() - 2, bubble)
        bubble.show()
        self._sync_timer.start()

    # ==================== 多选删除 ====================
    def is_selection_mode(self) -> bool:
        return self._selection_mode

    def set_selection_mode(self, enabled: bool):
        if enabled == self._selection_mode:
            return
        self._selection_mode = enabled
        for bubble in self._bubbles.values():
            bubble.set_selection_mode(enabled)
        self._column.set_open(enabled)
        if not enabled:
            self._selected_ids.clear()
            self._column.uncheck_all()
            self.selection_changed.emit(0)
        self.selection_mode_changed.emit(enabled)

    def selected_ids(self) -> list[str]:
        return list(self._selected_ids)

    def delete_selected(self) -> list[str]:
        """删除当前选中的消息气泡，返回被删除的 message_id 列表（供上层同步持久化存储）。"""
        ids = list(self._selected_ids)
        for message_id in ids:
            if message_id in self._bubbles:
                self.remove_message(message_id)
        self._selected_ids.clear()
        self.set_selection_mode(False)
        return ids

    def _on_dot_toggled(self, message_id: str, checked: bool):
        if checked:
            self._selected_ids.add(message_id)
        else:
            self._selected_ids.discard(message_id)
        self.selection_changed.emit(len(self._selected_ids))

        # 取消到一条都不剩时自动退出多选模式。
        if not self._selected_ids and self._selection_mode:
            self.set_selection_mode(False)

    def _sync_column(self):
        for message_id, bubble in self._bubbles.items():
            self._column.place_dot(message_id, bubble.y() + bubble.height() // 2)

    # ==================== 空白处拖拽框选 ====================
    _DRAG_THRESHOLD = 6
    _drag_origin: QPoint | None = None
    _drag_active = False

    def _press_on_bubble_body(self, content_pos: QPoint) -> bool:
        for bubble in self._bubbles.values():
            wrap = bubble._bubble_wrap
            if QRect(bubble.mapTo(self._content, wrap.pos()), wrap.size()).contains(content_pos):
                return True
        return False

    def _update_drag(self, content_pos: QPoint):
        if not self._drag_active:
            if (content_pos - self._drag_origin).manhattanLength() < self._DRAG_THRESHOLD:
                return
            self._drag_active = True
            self._band.show()
            self._band.raise_()

        rect = QRect(self._drag_origin, content_pos).normalized()
        self._band.setGeometry(rect)

        hits = {mid for mid, b in self._bubbles.items() if b.geometry().intersects(rect)}
        if hits and not self._selection_mode:
            self.set_selection_mode(True)
        if not self._selection_mode:
            return
        for mid in self._bubbles:
            checked = mid in hits
            if checked != (mid in self._selected_ids):
                self._column.set_checked(mid, checked)
                if checked:
                    self._selected_ids.add(mid)
                else:
                    self._selected_ids.discard(mid)
        self.selection_changed.emit(len(self._selected_ids))

    def _end_drag(self):
        self._drag_origin = None
        self._drag_active = False
        self._band.hide()
        if self._selection_mode and not self._selected_ids:
            self.set_selection_mode(False)

    def eventFilter(self, obj, event):
        if obj is self._container:
            return super().eventFilter(obj, event)
        t = event.type()
        if t in (QEvent.Type.Resize, QEvent.Type.Move, QEvent.Type.LayoutRequest):
            self._sync_timer.start()
            return False

        if t == QEvent.Type.MouseButtonPress and event.button() == Qt.MouseButton.LeftButton:
            pos = obj.mapTo(self._content, event.position().toPoint())
            if not self._press_on_bubble_body(pos):
                self._drag_origin = pos
                self._drag_active = False
                return True
        elif self._drag_origin is not None and t == QEvent.Type.MouseMove:
            self._update_drag(obj.mapTo(self._content, event.position().toPoint()))
            return True
        elif self._drag_origin is not None and t == QEvent.Type.MouseButtonRelease:
            self._end_drag()
            return True
        return super().eventFilter(obj, event)

    def clear_from_index(self, start_index: int):
        if start_index < 0:
            return

        keys_to_remove = []
        for i, (msg_id, bubble) in enumerate(list(self._bubbles.items())):
            if i >= start_index:
                keys_to_remove.append(msg_id)

        for msg_id in keys_to_remove:
            if msg_id in self._bubbles:
                self.remove_message(msg_id)

    def clear(self):
        for b in self._bubbles.values():
            b.deleteLater()
        self._bubbles.clear()
        self._column.clear_dots()
        self._selected_ids.clear()
        if self._selection_mode:
            self._selection_mode = False
            self._column.set_open(False)
            self.selection_changed.emit(0)
            self.selection_mode_changed.emit(False)

    def load_history(self, messages: list[dict]):
        """从历史记录列表恢复聊天"""
        for msg in messages:
            try:
                content =dict(msg["content"])
                self.add_message(msg["role"], content, msg["time"], msg["message_id"])
            except Exception as e:
                self.add_message(msg["role"], msg["content"], msg["time"], msg["message_id"])

    def _on_retry_requested(self, message_id: str):
        if message_id not in self._bubbles:
            return
        log_info("正在重新生成...")
        self.retry_requested.emit(message_id)

    def on_message_edited(self, msg_id: str, new_content: str):
        self.edit_requested.emit(msg_id, new_content)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.fade_mask.setGeometry(-1, -10, self.width()+2, 60)



if __name__ == '__main__':

    from PySide6.QtWidgets import QApplication
    import sys
    app = QApplication(sys.argv)

    widget = QWidget()
    widget.resize(800, 600)
    # 创建
    spinner = SpinnerWidget(
        parent=widget,
        size=48,  # 尺寸
        ring_width=3,  # 圆弧宽度
        color="#a0a0a0",  # 颜色
        speed=6,  # 旋转速度
        fade_duration=250  # 渐变时长 ms
    )

    widget.show()

    # 显示（渐显 + 开始旋转）
    spinner.show_spinner()
    app.exec()


    # 隐藏（渐隐 + 自动停止）
    # spinner.hide_spinner()