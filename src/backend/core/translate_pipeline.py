import re
from typing import Optional


class TranslationPipeline:
    """
    基于 deep-translator 的翻译管道
    主引擎: Google Translate（质量最好）
    备选引擎: MyMemory（完全免费，无限制，离线友好）
    """

    _instance  = None
    # 支持的语言列表
    SUPPORTED_LANGUAGES = {
        "zh-CN": "中文（简体）",
        "zh-TW": "中文（繁体）",
        "en": "英语",
        "ja": "日语",
        "ko": "韩语",
        "es": "西班牙语",
        "fr": "法语",
        "de": "德语",
        "it": "意大利语",
        "pt": "葡萄牙语",
        "ru": "俄语",
        "ar": "阿拉伯语",
        "hi": "印地语",
        "th": "泰语",
        "vi": "越南语",
        "id": "印尼语",
        "nl": "荷兰语",
        "pl": "波兰语",
        "tr": "土耳其语",
        "sv": "瑞典语",
        "da": "丹麦语",
        "fi": "芬兰语",
        "no": "挪威语",
        "cs": "捷克语",
        "hu": "匈牙利语",
        "ro": "罗马尼亚语",
        "uk": "乌克兰语",
        "he": "希伯来语",
        "fa": "波斯语",
        "ms": "马来语",
    }

    # MyMemory 语言代码映射（格式不同）
    _MYMEMORY_LANG_MAP = {
        "zh-CN": "zh-CN",
        "zh-TW": "zh-TW",
        "en": "en-US",
        "ja": "ja-JP",
        "ko": "ko-KR",
        "es": "es-ES",
        "fr": "fr-FR",
        "de": "de-DE",
        "it": "it-IT",
        "pt": "pt-PT",
        "ru": "ru-RU",
        "ar": "ar-SA",
        "hi": "hi-IN",
        "th": "th-TH",
        "vi": "vi-VN",
        "tr": "tr-TR",
        "pl": "pl-PL",
        "nl": "nl-NL",
        "sv": "sv-SE",
    }

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance


    def __init__(self):
        if self._initialized:
            return
        from deep_translator import GoogleTranslator, MyMemoryTranslator
        self._GoogleTranslator = GoogleTranslator
        self._MyMemoryTranslator = MyMemoryTranslator
        # 缓存已创建的 translator 实例，避免重复创建
        self._translator_cache: dict = {}
        self._initialized = True

    def _get_google_translator(self, source: str, target: str):
        key = f"google:{source}:{target}"
        if key not in self._translator_cache:
            self._translator_cache[key] = self._GoogleTranslator(
                source=source, target=target
            )
        return self._translator_cache[key]

    def _get_mymemory_translator(self, source: str, target: str):
        src = self._MYMEMORY_LANG_MAP.get(source, source)
        tgt = self._MYMEMORY_LANG_MAP.get(target, target)
        key = f"mymemory:{src}:{tgt}"
        if key not in self._translator_cache:
            self._translator_cache[key] = self._MyMemoryTranslator(
                source=src, target=tgt
            )
        return self._translator_cache[key]

    def _is_chinese(self, text: str) -> bool:
        zh = len(re.findall(r'[\u4e00-\u9fff]', text))
        return zh / max(len(text), 1) > 0.15

    def _detect_source(self, text: str) -> str:
        """简单语种检测，返回语言代码"""
        zh = len(re.findall(r'[\u4e00-\u9fff]', text))
        ja = len(re.findall(r'[\u3040-\u309f\u30a0-\u30ff]', text))
        ko = len(re.findall(r'[\uac00-\ud7af]', text))
        ar = len(re.findall(r'[\u0600-\u06ff]', text))
        total = max(len(text), 1)
        if zh / total > 0.15:
            return "zh-CN"
        if ja / total > 0.15:
            return "ja"
        if ko / total > 0.15:
            return "ko"
        if ar / total > 0.15:
            return "ar"
        return "auto"

    def translate(
            self,
            text: str,
            target: str = "en",
            source: str = "auto",
    ) -> str:
        """
        通用翻译接口
        Args:
            text:   要翻译的文本
            target: 目标语言代码，如 'en' 'zh-CN' 'ja' 等
            source: 源语言代码，默认 'auto' 自动检测

        Returns:
            翻译结果，失败时返回原文
        """
        if not text or not text.strip():
            return text

        if target not in self.SUPPORTED_LANGUAGES:
            raise ValueError(
                f"不支持的目标语言: {target}\n"
                f"支持的语言: {self.get_supported_languages()}"
            )

        # 主引擎
        try:
            translator = self._get_google_translator(source, target)
            result = translator.translate(text)
            if result:
                print(f"🔤 Google [{source}→{target}]: {text[:20]}... → {result[:20]}...")
                return result
        except Exception as e:
            print(f"⚠️ Google翻译失败，切换备选: {e}")

        # 备选引擎
        try:
            detected_source = source if source != "auto" else self._detect_source(text)
            fallback = self._get_mymemory_translator(detected_source, target)
            result = fallback.translate(text)
            if result:
                print(f"🔤 MyMemory [{detected_source}→{target}]: {result[:20]}...")
                return result
        except Exception as e:
            print(f"⚠️ 备选翻译失败: {e}")

        print("⚠️ 翻译失败，返回原文")
        return text

    def translate_to_english(self, text: str, source: str = "auto") -> str:
        if not self._is_chinese(text) and source == "auto":
            return text
        return self.translate(text, target="en", source=source)

    def translate_to_chinese(self, text: str, source: str = "auto") -> str:
        return self.translate(text, target="zh-CN", source=source)

    def translate_to_japanese(self, text: str, source: str = "auto") -> str:
        return self.translate(text, target="ja", source=source)

    def translate_to_korean(self, text: str, source: str = "auto") -> str:
        return self.translate(text, target="ko", source=source)

    @classmethod
    def get_supported_languages(cls) -> dict:
        return cls.SUPPORTED_LANGUAGES.copy()

    @classmethod
    def get_language_name(cls, code: str) -> Optional[str]:
        return cls.SUPPORTED_LANGUAGES.get(code)

    @classmethod
    def is_language_supported(cls, code: str) -> bool:
        return code in cls.SUPPORTED_LANGUAGES

def translate(text: str, target: str = "en", source: str = "auto") -> str:
    return TranslationPipeline().translate(text, target=target, source=source)

def translate_to_english(text: str) -> str:
    return TranslationPipeline().translate_to_english(text)

def translate_to_chinese(text: str) -> str:
    return TranslationPipeline().translate_to_chinese(text)

def get_supported_languages() -> dict:
    """获取所有支持的语言"""
    return TranslationPipeline.get_supported_languages()


if __name__ == '__main__':
    pipeline = TranslationPipeline()

    print("支持的语言：")
    for code, name in pipeline.get_supported_languages().items():
        print(f"  {code}: {name}")


    test_text = "今天天气很好，我很开心。"

    # 翻译到不同语言
    targets = ["en", "ja", "ko", "es", "fr", "de"]
    for target in targets:
        result = pipeline.translate(test_text, target=target)
        lang_name = pipeline.get_language_name(target)
        print(f"\n→ {lang_name}({target}): {result}")

    # 通用接口
    print("\n通用接口测试：")
    print(translate("Hello World", target="zh-CN"))
    print(translate("こんにちは", target="en"))

