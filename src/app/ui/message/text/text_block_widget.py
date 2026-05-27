from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont, QTextCursor
from PySide6.QtWidgets import QTextBrowser, QFrame

# 与 RenderMarkDown 保持一致的样式常量
_FONT_FAMILY = "JetBrains Mono, Cascadia Code, Consolas, monospace"
_BODY_SIZE   = 11
_CODE_SIZE   = 12
_COLOR_TEXT  = "#DCDCDC"
_COLOR_HEAD  = "#9AC5EC"
_COLOR_CODE  = "#E06C75"
_BG_CODE     = "#1e2433"

_BASE_CSS = f"""
body {{
    font-family: 'Noto Sans SC', 'Microsoft YaHei', 'PingFang SC', sans-serif;
    font-size: {_BODY_SIZE}pt;
    color: {_COLOR_TEXT};
    line-height: 1.3;
    margin: 0; padding: 0;
}}
h1 {{ font-size: 22pt; color: {_COLOR_HEAD}; margin: 12px 0 6px 0; }}
h2 {{ font-size: 19pt; color: {_COLOR_HEAD}; margin: 10px 0 5px 0; }}
h3 {{ font-size: 17pt; color: {_COLOR_HEAD}; margin: 8px 0 4px 0; }}
h4, h5, h6 {{ font-size: {_BODY_SIZE}pt; color: {_COLOR_HEAD}; margin: 6px 0 3px 0; }}
p {{ margin: 4px 0; }}
b, strong {{ color: #FFFFFF; }}
i, em {{ color: #d8c8a0; }}
code {{
    font-family: {_FONT_FAMILY};
    font-size: {_CODE_SIZE}pt;
    color: {_COLOR_CODE};
    background: {_BG_CODE};
    padding: 1px 5px;
    border-radius: 3px;
}}
pre {{
    background: {_BG_CODE};
    border-radius: 6px;
    padding: 8px 12px;
    margin: 6px 0;
    overflow-x: auto;
}}
pre code {{
    background: transparent;
    padding: 0;
    font-size: {_CODE_SIZE}pt;
}}
a {{ color: #58A6FF; text-decoration: none; }}
ul, ol {{ margin: 4px 0 4px 16px; padding: 0; }}
li {{ margin: 3px 0; line-height: 1.3; }}
hr {{ border: none; border-top: 1px solid #30363d; margin: 8px 0; }}
blockquote {{
    border-left: 3px solid #3b5070;
    margin: 4px 0 4px 8px;
    padding: 2px 12px;
    color: #8b949e;
}}
"""


def _wrap_html(body: str) -> str:
    return f"<html><head><style>{_BASE_CSS}</style></head><body>{body}</body></html>"


def _raw_to_html(text: str) -> str:
    """
    把未完成的流式原始文本转成 HTML：
    - 用 markdown-it 解析已完成的部分（不含末尾未闭合代码块）
    - 末尾未闭合的代码块/普通文本直接 <pre> 或转义后追加，保持实时可见
    """
    # 判断是否有未闭合的代码块
    fence_count = text.count("```")
    if fence_count % 2 == 1:
        # 有未闭合的 ```：把已闭合部分交给 md 解析，未闭合部分用 <pre> 展示
        last_fence = text.rfind("```")
        closed_part = text[:last_fence]
        open_part   = text[last_fence + 3:]  # ``` 之后的内容（含语言行+代码）

        closed_html = _md_to_html(closed_part) if closed_part.strip() else ""

        # 提取语言行
        first_nl = open_part.find("\n")
        if first_nl != -1:
            lang      = open_part[:first_nl].strip()
            code_body = open_part[first_nl + 1:]
        else:
            lang      = open_part.strip()
            code_body = ""

        lang_attr = f' class="language-{lang}"' if lang else ""
        escaped   = code_body.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        open_html = f'<pre><code{lang_attr}>{escaped}</code></pre>'
        return _wrap_html(closed_html + open_html)
    else:
        return _wrap_html(_md_to_html(text))


def _md_to_html(text: str) -> str:
    """用 markdown-it 把文本转成 HTML 片段（不含 <html>/<body>）。"""
    from markdown_it import MarkdownIt
    md = MarkdownIt("commonmark", {"breaks": True, "html": True})
    md.enable(["table"])
    return md.render(text)


class TextBlockWidget(QTextBrowser):
    """ 流式文本块 """
    _THROTTLE_MS = 80

    def __init__(self, parent=None):
        super().__init__(parent)
        self._raw_text    = ""   # 累积的完整原始文本
        self._dirty       = False
        self._layout_ref = None  # 持有父 layout 引用，finish 时用于替换自身
        self._setup_style()
        # 复用游标，避免每次 append 都构造新 QTextCursor
        self._cursor = QTextCursor(self.document())

        self._flush_timer = QTimer(self)
        self._flush_timer.setInterval(self._THROTTLE_MS)
        self._flush_timer.timeout.connect(self._flush_to_ui)
        self._flush_timer.start()

    def _setup_style(self):
        self.setObjectName("streaming_text_widget")
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setOpenExternalLinks(True)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setReadOnly(True)

        font = QFont()
        font.setFamilies(["Noto Sans SC", "Microsoft YaHei", "Segoe UI", "sans-serif"])
        font.setPointSize(11)
        font.setStyleStrategy(QFont.StyleStrategy.PreferAntialias)
        font.setHintingPreference(QFont.HintingPreference.PreferNoHinting)
        self.setFont(font)

        self.setStyleSheet("""
            QTextBrowser {

            }
        """)
        self.document().setDocumentMargin(0)

    def append_chunk(self, text: str) -> None:
        self._raw_text += text
        self._dirty = True

    def finish(self) -> None:
        pass

    def set_layout_ref(self, layout) -> None:
        self._layout_ref = layout

    def _flush_to_ui(self) -> None:
        """定时器回调：若有新内容则重新渲染 HTML 并刷新显示。"""
        if not self._dirty:
            return
        self._dirty = False
        html = _raw_to_html(self._raw_text)

        # 暂停更新减少闪烁
        self.setUpdatesEnabled(False)
        self.setHtml(html)
        self.document().setDocumentMargin(0)
        self._relax_height()
        self.setUpdatesEnabled(True)

    def _relax_height(self) -> None:
        doc = self.document()
        doc.setTextWidth(self.viewport().width() or 600)
        h = int(doc.size().height()) + 15
        print("height:", h)
        self.setFixedHeight(h)
