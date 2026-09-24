"""界面多语言的契约测试：语言包齐全、译文完整、占位符不丢。

不需要显示器：qapp 夹具（conftest.py）走离屏平台。
"""
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
from PySide6.QtCore import QCoreApplication, QTranslator

from src.app.i18n import LANGUAGES, TRANSLATIONS_DIR, app_translation_path

ROOT = Path(__file__).resolve().parents[1]
PLACEHOLDER = re.compile(r"\{\d+\}")
# Qt 复数规则：每种语言的 numerusform 个数
PLURAL_FORMS = {"zh": 1, "ja": 1, "ko": 1, "en": 2, "es": 2, "ru": 3}


def _messages(code):
    tree = ET.parse(TRANSLATIONS_DIR / f"app_{LANGUAGES[code][0]}.ts")
    for context in tree.iter("context"):
        for msg in context.iter("message"):
            yield context.findtext("name"), msg


@pytest.mark.parametrize("code", LANGUAGES)
def test_every_language_has_ts_and_qm(code):
    assert (TRANSLATIONS_DIR / f"app_{LANGUAGES[code][0]}.ts").exists()
    assert app_translation_path(code).exists(), "改完 .ts 要运行 script/build_pro.bat 生成 .qm"


@pytest.mark.parametrize("code", LANGUAGES)
def test_translations_are_complete_and_keep_placeholders(code):
    for context, msg in _messages(code):
        source = msg.findtext("source")
        tr = msg.find("translation")
        where = f"{code} {context}: {source!r}"
        assert tr.get("type") not in ("unfinished", "vanished", "obsolete"), f"未翻译 {where}"
        if msg.get("numerus") == "yes":
            forms = [f.text or "" for f in tr.iter("numerusform")]
            assert len(forms) == PLURAL_FORMS[code], f"复数形式个数不对 {where}"
            assert any("%n" in f for f in forms), f"复数译文丢了 %n {where}"
        else:
            text = tr.text or ""
            assert text.strip(), f"译文为空 {where}"
            assert sorted(PLACEHOLDER.findall(text)) == sorted(PLACEHOLDER.findall(source)), \
                f"占位符不一致 {where}: {text!r}"


@pytest.mark.parametrize("code, expected", [
    ("zh", "通用设置"), ("en", "General Settings"), ("ja", "一般設定"),
    ("ko", "일반 설정"), ("ru", "Общие настройки"), ("es", "Ajustes generales"),
])
def test_qm_translates_at_runtime(qapp, code, expected):
    translator = QTranslator()
    assert translator.load(str(app_translation_path(code)))
    assert translator.translate("GeneralPage", "General Settings") == expected


def test_russian_plural_forms_pick_by_count(qapp):
    translator = QTranslator()
    assert translator.load(str(app_translation_path("ru")))
    qapp.installTranslator(translator)
    try:
        source = "Delete the %n selected message(s)? This cannot be undone."
        one = QCoreApplication.translate("GenerationPage", source, "", 1)
        few = QCoreApplication.translate("GenerationPage", source, "", 3)
        many = QCoreApplication.translate("GenerationPage", source, "", 5)
        assert "1 выбранное сообщение" in one
        assert "3 выбранных сообщения" in few
        assert "5 выбранных сообщений" in many
    finally:
        qapp.removeTranslator(translator)


def test_no_fstring_passed_to_tr():
    """tr(f"...") 先插值再查表，永远查不到译文，lupdate 也提取不到。"""
    offenders = []
    for path in (ROOT / "src" / "app").rglob("*.py"):
        for no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if re.search(r"""\b(tr|translate)\((\s*"[^"]*",\s*)?f["']""", line):
                offenders.append(f"{path.relative_to(ROOT)}:{no}")
    assert not offenders, offenders
