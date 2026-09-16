from PySide6.QtCore import Signal, Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QWidget, QVBoxLayout, QScrollArea

from src.app.ui.message.message_bubble import MessageBubble, create_message_bubble, SpinnerWidget, FadeMask
from src.app.ui.setting.page.log_page import log_info


class ChatWidget(QScrollArea):
    """
    可滚动的聊天消息列表。
    维护一个 QWidget 作为内容容器，消息依次追加。
    """
    retry_requested = Signal(str)
    edit_requested = Signal(str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("chat_widget")
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self._container = QWidget()
        self._layout = QVBoxLayout(self._container)
        self._layout.setContentsMargins(0, 16, 0, 16)
        self._layout.setSpacing(15)

        self._layout.addStretch()
        self._layout.addSpacing(20)

        self.setWidget(self._container)
        self._bubbles: dict[str, MessageBubble] = {}

        fade_mask_color = QColor("#101215")
        fade_mask_color.setAlpha(255)
        self.fade_mask = FadeMask(fade_mask_color, self)
        self.fade_mask.set_reverse(False)
        self.fade_mask.setVisible(True)

    def add_message(self, role, content, timestamp, message_id=None):
        bubble = create_message_bubble(role=role, content=content, timestamp=timestamp, message_id=message_id)
        bubble.retry_requested.connect(self._on_retry_requested)
        bubble.content_edited.connect(self.on_message_edited)

        self._bubbles[bubble.message_id] = bubble
        self._layout.insertWidget(self._layout.count() - 2, bubble)
        return bubble

    def remove_message(self, message_id):
        self._bubbles[message_id].deleteLater()
        del self._bubbles[message_id]

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