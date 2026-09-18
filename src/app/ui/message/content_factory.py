from pathlib import Path
from typing import Optional, List
from PySide6.QtCore import QPropertyAnimation, QEasingCurve
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QGraphicsOpacityEffect

from src.app.ui.message.message_widgets import ImageWidget, VideoWidget, FileWidget, AudioWidget
from src.app.ui.message.text.markdown_widget import render_markdown

# 支持的文件扩展名分组
_IMAGE_EXTS = frozenset({".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"})
_VIDEO_EXTS = frozenset({".mp4", ".mov", ".avi", ".webm", ".gif"})
_AUDIO_EXTS = frozenset({'.mp3', '.wav', '.flac', '.ogg', '.m4a'})


def fade_in(widget: QWidget, duration: int = 300):
    """淡入；结束后移除透明度特效（特效会让控件走离屏渲染，常驻很吃性能）。"""
    effect = QGraphicsOpacityEffect(widget)
    effect.setOpacity(0.0)
    widget.setGraphicsEffect(effect)
    anim = QPropertyAnimation(effect, b"opacity", widget)
    anim.setDuration(duration)
    anim.setStartValue(0.0)
    anim.setEndValue(1.0)
    anim.setEasingCurve(QEasingCurve.Type.OutCubic)
    anim.finished.connect(lambda: widget.setGraphicsEffect(None))
    anim.start(QPropertyAnimation.DeletionPolicy.DeleteWhenStopped)


class ContentLoader:

    def __init__(self, layout: QVBoxLayout):
        self._layout = layout
        self._open_row: Optional[QHBoxLayout] = None   # append() 正在填的、只放了一张图的那一行

    def append(self, path, animate: bool = True) -> Optional[QWidget]:
        """增量追加一个附件（图片按两张一行接着上一行排），用于生成过程中逐张显示。"""
        widget = ContentLoader._make_file_widget(Path(path))
        if not widget:
            return None
        if isinstance(widget, ImageWidget):
            if self._open_row is not None:
                self._open_row.insertWidget(self._open_row.count() - 1, widget)
                self._open_row = None
            else:
                row = QHBoxLayout()
                row.setContentsMargins(0, 0, 0, 0)
                row.setSpacing(4)
                row.addWidget(widget)
                row.addStretch()
                self._layout.addLayout(row)
                self._open_row = row
        else:
            self._open_row = None
            self._layout.addWidget(widget)
        # QVideoWidget 套透明度特效会渲染异常，视频不做淡入。
        if animate and not isinstance(widget, VideoWidget):
            fade_in(widget)
        return widget

    def _add_widgets_in_pairs(self, widgets: list):
        """将 widgets 两两一行添加到 layout。
        两张图都是固定尺寸、都不参与拉伸时，QHBoxLayout 会把多余宽度平摊在
        前面/中间/后面，图片之间反而比设定的 4px 松得多；末尾补一个 stretch
        把多余空间统一吸收到右侧，图片才会紧贴在一起。"""
        for i in range(0, len(widgets), 2):
            row = QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(4)
            row.addWidget(widgets[i])
            if i + 1 < len(widgets):
                row.addWidget(widgets[i + 1])
            row.addStretch()
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