from enum import Enum, auto
from PySide6.QtWidgets import QVBoxLayout

from src.app.ui.message.text.text_block_widget import TextBlockWidget
from src.app.ui.message.text.code_block_widget import CodeBlockWidget


class _State(Enum):
    TEXT = auto()
    CODE = auto()


class StreamingRenderer:
    """ 流式 Markdown 渲染状态机。"""

    def __init__(self, layout: QVBoxLayout):
        self._layout = layout
        self._state = _State.TEXT

        self._active: TextBlockWidget | CodeBlockWidget | None = None # 当前活跃控件
        self._all_widgets: list[TextBlockWidget | CodeBlockWidget] = []

        self._tail_buf: str = ""            # 尾部暂存区：最多保留 2 个字符，用于跨 chunk 的 ``` 检测
        self._lang_buf: str = ""            # CODE 状态下：语言标识符缓冲（读取 ```lang 这一行时用）
        self._reading_lang: bool = False    # 是否正在读 ```lang\n 这一行

    def append_chunk(self, chunk: str) -> None:
        data = self._tail_buf + chunk
        self._tail_buf = ""
        self._scan(data)

    def finish(self) -> None:
        # 冲刷跨 chunk 暂存的尾部碎片
        if self._tail_buf:
            self._dispatch(self._tail_buf)
            self._tail_buf = ""

    def _scan(self, data: str) -> None:
        """
        逐字符扫描 data，根据状态机决定分发给哪个控件。
        TEXT 状态：监视 ``` 的出现
        CODE 状态：监视结束 ``` 的出现
        """
        i = 0
        while i < len(data):
            # TEXT 状态：寻找 ``` 开始
            if self._state == _State.TEXT:
                fence_pos = data.find("```", i)
                if fence_pos == -1:
                    # 没有 ```，但末尾可能是残缺的 ` 或 ``
                    # 保留最后 2 个字符到 tail_buf 防止跨 chunk 漏检
                    safe_end = max(i, len(data) - 2)
                    if safe_end > i:
                        self._dispatch(data[i:safe_end])
                    self._tail_buf = data[safe_end:]
                    break
                else:
                    # ``` 之前的普通文本先分发
                    if fence_pos > i:
                        self._dispatch(data[i:fence_pos])

                    # 跳过 ``` 本身，开始读语言标识符
                    i = fence_pos + 3
                    self._lang_buf = ""
                    self._reading_lang = True
                    self._state = _State.CODE

            # ── CODE 状态：寻找 ``` 结束 ──────────────────────────────────────
            else:
                # 还在读 ```lang\n 这一行（语言标识符）
                if self._reading_lang:
                    newline_pos = data.find("\n", i)
                    if newline_pos == -1:
                        # 这一 chunk 里 lang 行还没结束，先暂存
                        self._lang_buf += data[i:]
                        self._tail_buf = ""
                        break
                    else:
                        self._lang_buf += data[i:newline_pos]
                        lang = self._lang_buf.strip()
                        self._reading_lang = False
                        i = newline_pos + 1  # 跳过 \n

                        # 创建 CodeBlockWidget，成为新的活跃控件
                        self._switch_to_code(lang)
                        continue
                # 正式代码内容，寻找结束 ```
                close_pos = data.find("```", i)
                if close_pos == -1:
                    # 代码还没结束，保留末尾 2 字符防止跨 chunk 漏检
                    safe_end = max(i, len(data) - 2)
                    if safe_end > i:
                        self._dispatch(data[i:safe_end])
                    self._tail_buf = data[safe_end:]
                    break
                else:
                    # ``` 之前的代码内容先分发
                    if close_pos > i:
                        self._dispatch(data[i:close_pos])

                    # 代码块结束，切回 TEXT 状态
                    i = close_pos + 3
                    # 跳过结束 ``` 后面可能紧跟的换行
                    if i < len(data) and data[i] == "\n":
                        i += 1
                    self._switch_to_text()

    def _switch_to_text(self) -> None:
        if self._active is not None:
            self._active.finish()
        self._state = _State.TEXT
        self._active = None

    def _switch_to_code(self, lang: str) -> None:
        """切换到 CODE 状态，立即创建 CodeBlockWidget。"""
        widget = CodeBlockWidget(code="", lang=lang)
        self._layout.addWidget(widget)
        self._all_widgets.append(widget)
        self._active = widget

    def _dispatch(self, text: str) -> None:
        """
        把文本分发给当前活跃控件。
        TEXT 状态下懒创建 TextBlockWidget（避免只有空白时创建无意义的 widget）。
        """
        if not text:
            return

        if self._state == _State.TEXT:
            if self._active is None:
                # 懒创建：第一个非空字符到来时才建 widget
                widget = TextBlockWidget()
                widget.set_layout_ref(self._layout)
                self._layout.addWidget(widget)
                self._all_widgets.append(widget)
                self._active = widget

        if self._active is not None:
            self._active.append_chunk(text)