"""语音识别可选的语言：设置页（麦克风口述）和转写参数面板共用。

代码要和 models.json 里 transcription 各模型的 languages 字段对得上，后端据此给 Auto 挑模型。
"""
from PySide6.QtCore import QCoreApplication, QT_TRANSLATE_NOOP
from PySide6.QtWidgets import QComboBox

_CONTEXT = "TranscriptionLanguage"
TRANSCRIPTION_LANGUAGES = [
    ("auto", QT_TRANSLATE_NOOP("TranscriptionLanguage", "Auto detect")),
    ("zh", QT_TRANSLATE_NOOP("TranscriptionLanguage", "Chinese")),
    ("yue", QT_TRANSLATE_NOOP("TranscriptionLanguage", "Cantonese")),
    ("en", QT_TRANSLATE_NOOP("TranscriptionLanguage", "English")),
    ("ja", QT_TRANSLATE_NOOP("TranscriptionLanguage", "Japanese")),
    ("ko", QT_TRANSLATE_NOOP("TranscriptionLanguage", "Korean")),
    ("ru", QT_TRANSLATE_NOOP("TranscriptionLanguage", "Russian")),
    ("es", QT_TRANSLATE_NOOP("TranscriptionLanguage", "Spanish")),
    ("fr", QT_TRANSLATE_NOOP("TranscriptionLanguage", "French")),
    ("de", QT_TRANSLATE_NOOP("TranscriptionLanguage", "German")),
    ("it", QT_TRANSLATE_NOOP("TranscriptionLanguage", "Italian")),
    ("pt", QT_TRANSLATE_NOOP("TranscriptionLanguage", "Portuguese")),
    ("nl", QT_TRANSLATE_NOOP("TranscriptionLanguage", "Dutch")),
    ("pl", QT_TRANSLATE_NOOP("TranscriptionLanguage", "Polish")),
    ("uk", QT_TRANSLATE_NOOP("TranscriptionLanguage", "Ukrainian")),
    ("th", QT_TRANSLATE_NOOP("TranscriptionLanguage", "Thai")),
    ("vi", QT_TRANSLATE_NOOP("TranscriptionLanguage", "Vietnamese")),
    ("id", QT_TRANSLATE_NOOP("TranscriptionLanguage", "Indonesian")),
    ("tr", QT_TRANSLATE_NOOP("TranscriptionLanguage", "Turkish")),
    ("ar", QT_TRANSLATE_NOOP("TranscriptionLanguage", "Arabic")),
    ("hi", QT_TRANSLATE_NOOP("TranscriptionLanguage", "Hindi")),
]


def fill_language_combo(combo: QComboBox, current: str = "auto"):
    """显示译名、itemData 存语言代码，保存和恢复都按代码，不受界面语言影响。"""
    for code, name in TRANSCRIPTION_LANGUAGES:
        combo.addItem(QCoreApplication.translate(_CONTEXT, name), code)
    combo.setCurrentIndex(max(combo.findData(current), 0))
