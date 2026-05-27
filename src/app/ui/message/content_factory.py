from pathlib import Path
from typing import Optional, List
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout

from ui.message.message_widgets import ImageWidget, VideoWidget, FileWidget, AudioWidget
from ui.message.text.markdown_widget import render_markdown

# 支持的文件扩展名分组
_IMAGE_EXTS = frozenset({".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"})
_VIDEO_EXTS = frozenset({".mp4", ".mov", ".avi", ".webm", ".gif"})
_AUDIO_EXTS = frozenset({'.mp3', '.wav', '.flac', '.ogg', '.m4a'})


class ContentLoader:

    def __init__(self, layout: QVBoxLayout):
        self._layout = layout

    def _add_widgets_in_pairs(self, widgets: list):
        """将 widgets 两两一行添加到 layout"""
        for i in range(0, len(widgets), 2):
            row = QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(4)
            row.addWidget(widgets[i])
            if i + 1 < len(widgets):
                row.addWidget(widgets[i + 1])
            else:
                row.addStretch()  # 奇数时末尾补空
            self._layout.addLayout(row)

    def load(self, attachments: Optional[List[Path]| List[str]] = None):
        sum_num = 0
        sum_height = 0

        if attachments:
            buckets: dict[type, list] = {}
            for att_path in attachments:
                if isinstance(att_path, (str, Path)):
                    widget = ContentLoader._make_file_widget(Path(att_path))
                    if not widget:
                        continue
                    bucket_key = type(widget)
                    buckets.setdefault(bucket_key, []).append(widget)

                    sum_num += 1
                    sum_height += widget.height()

            for widgets in buckets.values():
                self._add_widgets_in_pairs(widgets)

    @staticmethod
    def _make_file_widget(path: Path) -> QWidget:
        ext = path.suffix.lower()
        if ext in _IMAGE_EXTS:
            return ImageWidget(str(path))
        elif ext in _AUDIO_EXTS:
            return AudioWidget(str(path))
        elif ext in _VIDEO_EXTS:
            return VideoWidget(str(path))
        return FileWidget(str(path))



class ContentBuilder:
    """
    静态内容构建器，用于 UserBubble。
    一次性渲染完整 Markdown 文本 + 加载所有附件。
    """

    def __init__(self, layout: QVBoxLayout):
        self._layout = layout
        self._content_loader = ContentLoader(layout)

    def build(self, text: str, attachments: list[str] | None = None) -> None:
        """一次性构建所有内容。"""
        # 1. 渲染文字
        if text and text.strip():
            widgets = render_markdown.render_markdown_to_widgets(text)
            for w in widgets:
                self._layout.addWidget(w)

        # 2. 加载附件媒体
        if attachments:
            self._content_loader.load(attachments)