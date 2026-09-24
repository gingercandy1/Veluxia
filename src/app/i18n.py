"""界面语言：启动时按设置装载 resource/translations 下的语言包。

源文案统一写英文（Qt 惯例），中文等各语言都由 .ts 提供译文，
维护流程见 script/build_i18n.bat。
"""
from pathlib import Path

from PySide6.QtCore import QCoreApplication, QLibraryInfo, QLocale, QTranslator

TRANSLATIONS_DIR = Path(__file__).parent.parent.parent / "resource" / "translations"

# 设置里存短码（沿用 DEFAULT_CONFIG 里已有的 "zh"），语言包按 Qt locale 命名。
# 语言名用各自的母语写，这样切错语言后用户仍能认出该选哪一项。
LANGUAGES: dict[str, tuple[str, str]] = {
    "zh": ("zh_CN", "简体中文"),
    "en": ("en_US", "English"),
    "ja": ("ja_JP", "日本語"),
    "ko": ("ko_KR", "한국어"),
    "ru": ("ru_RU", "Русский"),
    "es": ("es_ES", "Español"),
}
DEFAULT_LANGUAGE = "zh"


def app_translation_path(code: str) -> Path:
    return TRANSLATIONS_DIR / f"app_{LANGUAGES[code][0]}.qm"


def install_language(app: QCoreApplication, code: str) -> list[str]:
    """装载界面语言包和 Qt 自带控件（标准按钮、文件对话框）的语言包。

    QTranslator 被回收后翻译会失效，所以挂在 app 上保持引用。
    返回装载失败的说明，由调用方写日志；缺语言包时界面退回英文源文案，不影响使用。
    """
    if code not in LANGUAGES:
        code = DEFAULT_LANGUAGE
    locale = QLocale(LANGUAGES[code][0])
    problems = []

    app_translator = QTranslator(app)
    app_qm = app_translation_path(code)
    if app_translator.load(str(app_qm)):
        app.installTranslator(app_translator)
    else:
        problems.append(f"界面语言包加载失败：{app_qm}")

    qt_translator = QTranslator(app)
    qt_dir = QLibraryInfo.path(QLibraryInfo.LibraryPath.TranslationsPath)
    # Qt 没有英文的 qtbase 包（源文案就是英文），其余语言缺失才算问题
    if qt_translator.load(locale, "qtbase", "_", qt_dir):
        app.installTranslator(qt_translator)
    elif code != "en":
        problems.append(f"Qt 自带语言包缺失：qtbase_{LANGUAGES[code][0]} ({qt_dir})")

    app._translators = [app_translator, qt_translator]
    return problems
