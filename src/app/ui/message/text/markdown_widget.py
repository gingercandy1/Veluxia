from typing import List, Tuple

from PySide6.QtGui import QFont, QFontMetrics, QTextBlock, Qt
from PySide6.QtWidgets import QWidget, QTextBrowser
from markdown_it import MarkdownIt
from markdown_it.token import Token

from src.app.ui.message.text.code_block_widget import CodeBlockWidget


class RenderMarkDown:
    # 字体常量
    FONT_FAMILY = "JetBrains Mono, Cascadia Code, Consolas, monospace"
    BODY_SIZE = 11
    CODE_SIZE = 12
    COLOR_TEXT = "#DCDCDC"
    COLOR_HEAD = "#9AC5EC"
    COLOR_CODE = "#E06C75"
    BG_CODE = "#1e2433"

    def __init__(self):
        # 模块级解析器
        self.md_parser = MarkdownIt("commonmark", {"breaks": True, "html": True})
        self.md_parser.enable(["table"])

    def _preprocess(self, text: str) -> str:
        """
        合并连续的单行代码块为一个完整代码块。
        AI 流式输出时有时每行单独包一个 ```，例如：
            ```python
            def foo():
            ```
            ```
                pass
            ```
        预处理后统一合并为：
            ```python
            def foo():
                pass
            ```
        """
        import re
        lines = text.splitlines()
        result: list[str] = []
        i = 0
        while i < len(lines):
            fence_open = re.match(r'^(`{3,})([\w\-]*)$', lines[i].strip())
            if fence_open:
                fence_mark = fence_open.group(1)
                lang = fence_open.group(2)
                body: list[str] = []
                i += 1
                while i < len(lines):
                    if re.match(r'^`{3,}$', lines[i].strip()):
                        i += 1
                        # 向前看：跳过空行后是否又是开启 fence
                        j = i
                        while j < len(lines) and lines[j].strip() == "":
                            j += 1
                        if j < len(lines) and re.match(r'^(`{3,})([\w\-]*)$', lines[j].strip()):
                            i = j + 1  # 跳过空行 + 下一个开启行，继续合并
                            continue
                        break
                    body.append(lines[i])
                    i += 1
                result.append(f"{fence_mark}{lang}")
                result.extend(body)
                result.append(fence_mark)
            else:
                result.append(lines[i])
                i += 1
        return "\n".join(result)


    def _inline_to_html(self,inline_token: Token) -> str:
        """
        把一个 inline token 的子 token 列表转换成 HTML 字符串，
        让 QLabel(RichText) 能正确渲染 bold / italic / code / link 等。
        """
        if not inline_token or not inline_token.children:
            return inline_token.content if inline_token else ""

        parts: List[str] = []
        for t in inline_token.children:
            if t.type == "text":
                parts.append(self._escape(t.content))
            elif t.type == "softbreak":
                parts.append("<br/>")
            elif t.type == "hardbreak":
                parts.append("<br/>")
            elif t.type == "strong_open":
                parts.append("<b>")
            elif t.type == "strong_close":
                parts.append("</b>")
            elif t.type == "em_open":
                parts.append("<i>")
            elif t.type == "em_close":
                parts.append("</i>")
            elif t.type == "code_inline":
                parts.append(
                    f'<code style="font-family:{self.FONT_FAMILY};font-size:{self.CODE_SIZE}pt;'
                    f'color:{self.COLOR_CODE};background:{self.BG_CODE};'
                    f'padding:1px 4px;border-radius:3px;">'
                    f'{self._escape(t.content)}</code>'
                )
            elif t.type == "link_open":
                href = dict(t.attrs or {}).get("href", "#")
                parts.append(f'<a href="{href}" style="color:#58A6FF;">')
            elif t.type == "link_close":
                parts.append("</a>")
            elif t.type == "image":
                src = dict(t.attrs or {}).get("src", "")
                alt = t.content or ""
                parts.append(f'<img src="{src}" alt="{self._escape(alt)}" />')
            else:
                # 其余未知 token 直接输出文本内容
                if t.content:
                    parts.append(self._escape(t.content))

        return "".join(parts)


    def _escape(self, text: str) -> str:
        return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    def _body_font(self) -> QFont:
        f = QFont()
        f.setFamilies(["JetBrains Mono", "Cascadia Code", "Segoe UI", "sans-serif"])
        f.setPointSize(self.BODY_SIZE)
        return f


    def _make_text_browser(self, html: str) -> "QTextBrowser":
        browser = QTextBrowser()
        browser.setObjectName("markdown_widget")
        browser.setOpenExternalLinks(True)
        browser.setFrameShape(QTextBrowser.NoFrame)
        browser.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        browser.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        font = self._body_font()  # 获取当前字体
        font.setStyleStrategy(QFont.StyleStrategy.PreferAntialias)
        font.setHintingPreference(QFont.HintingPreference.PreferNoHinting)
        browser.setFont(font)
        # browser.setStyleSheet("QTextBrowser { background: #ff0000; border: none; }")
        # 注入全局 CSS，QLabel 不支持这些，QTextBrowser 完全支持
        full_html = f"""
        <html><head><style>
            body {{
                font-family: 'Noto Sans SC', 'Microsoft YaHei', 'PingFang SC', sans-serif;
                font-size: {self.BODY_SIZE}pt;
                color: {self.COLOR_TEXT};
                line-height: 1.3;
                margin: 0; padding: 0;
            }}
            h1 {{ font-size: 22pt; color: {self.COLOR_HEAD}; margin: 12px 0 6px 0; }}
            h2 {{ font-size: 19pt; color: {self.COLOR_HEAD}; margin: 10px 0 5px 0; }}
            h3 {{ font-size: 17pt; color: {self.COLOR_HEAD}; margin: 8px 0 4px 0; }}
            h4, h5, h6 {{ font-size: {self.BODY_SIZE}pt; color: {self.COLOR_HEAD}; margin: 6px 0 3px 0; }}
            p {{    
                margin: 4px 0;
            }}
            b, strong {{ color: #FFFFFF; }}
            i, em {{ color: #d8c8a0; }}
            code {{
                font-family: 'JetBrains Mono', 'Cascadia Code', monospace;
                font-size: {self.CODE_SIZE}pt;
                color: {self.COLOR_CODE};
                background: {self.BG_CODE};
                padding: 1px 5px;
                border-radius: 3px;
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
        </style></head><body>{html}</body></html>
        """
        browser.setHtml(full_html)

        doc = browser.document()
        doc.setDocumentMargin(0)

        def get_longest_line_info(doc) -> Tuple[int, str]:
            """返回 (最长行像素宽度, 最长行纯文本内容)"""
            fm = QFontMetrics(browser.font())
            max_width = 0
            longest_text = ""

            block: QTextBlock = doc.begin()
            while block.isValid():
                text = block.text()
                if not text:
                    block = block.next()
                    continue

                line_width = fm.boundingRect(text).width()
                if line_width > max_width:
                    max_width = line_width
                    longest_text = text

                block = block.next()
            return max_width, longest_text

        def _calc_height(b) -> int:
            doc = b.document()
            doc.setTextWidth(browser.viewport().width())
            doc.adjustSize()

            raw_h = int(doc.size().height())
            fm = b.fontMetrics()

            multiply_line = raw_h > fm.lineSpacing() * 2
            if multiply_line:
                extra = -20
            else:
                extra = 8
                browser.wheelEvent = lambda e: e.ignore()
            return raw_h + extra

        max_pixel_width, longest_content = get_longest_line_info(doc)
        suggested_width = min(920, max_pixel_width + 80)
        # print(suggested_width, longest_content)
        # browser.setFixedWidth(suggested_width)

        # 自动撑高，消除内部滚动条
        height = _calc_height(browser)
        browser.setFixedHeight(height)
        return browser

    def _tokens_to_html(self, tokens, start: int, end: int) -> str:
        """将 start..end 范围内的 token 转成 HTML 字符串（跳过 fence）。"""
        html_parts: list[str] = []
        i = start
        while i < end:
            t = tokens[i]
            if t.type == "heading_open":
                level = int(t.tag[1])
                if i + 1 < end and tokens[i + 1].type == "inline":
                    i += 1
                    html_parts.append(f"<h{level}>{self._inline_to_html(tokens[i])}</h{level}>")
            elif t.type == "paragraph_open":
                if i + 1 < end and tokens[i + 1].type == "inline":
                    i += 1
                    inline_html = self._inline_to_html(tokens[i])
                    html_parts.append(
                        f'<p style="text-indent: 2em; margin: 6px 0 10px 0;">'
                        f'{inline_html}</p>'
                    )
            elif t.type == "bullet_list_open":
                html_parts.append("<ul>")
            elif t.type == "bullet_list_close":
                html_parts.append("</ul>")
            elif t.type == "ordered_list_open":
                html_parts.append("<ol>")
            elif t.type == "ordered_list_close":
                html_parts.append("</ol>")
            elif t.type == "list_item_open":
                html_parts.append("<li>")
            elif t.type == "list_item_close":
                html_parts.append("</li>")
            elif t.type == "inline":
                html_parts.append(self._inline_to_html(t))
            elif t.type == "hr":
                html_parts.append("<hr/>")
            elif t.type == "blockquote_open":
                html_parts.append("<blockquote>")
            elif t.type == "blockquote_close":
                html_parts.append("</blockquote>")
            i += 1
        return "".join(html_parts)

    def render_markdown_to_widgets(self, text: str) -> List[QWidget]:
        """将 Markdown 解析成 QWidget 列表（完整内联格式支持）。"""
        tokens: List[Token] = self.md_parser.parse(self._preprocess(text))
        widgets: List[QWidget] = []

        # 收集连续非 fence token 的范围，批量转 QTextBrowser
        pending_start: int | None = None

        def _flush(end: int):
            nonlocal pending_start
            if pending_start is None:
                return
            html = self._tokens_to_html(tokens, pending_start, end)
            if html.strip():
                widgets.append(self._make_text_browser(html))
            pending_start = None

        for i, t in enumerate(tokens):
            if t.type == "fence":
                _flush(i)
                widgets.append(CodeBlockWidget(t.content, t.info or ""))
            else:
                if pending_start is None:
                    pending_start = i

        _flush(len(tokens))
        return widgets


    def _collect_list_items(self,
            tokens: List[Token], start: int, close_type: str
    ) -> tuple[List[str], int]:
        items: List[str] = []
        i = start
        while i < len(tokens):
            t = tokens[i]
            if t.type == close_type:
                return items, i - start + 1
            if t.type == "inline":
                items.append(self._inline_to_html(t))
            i += 1
        return items, i - start

render_markdown = RenderMarkDown()

