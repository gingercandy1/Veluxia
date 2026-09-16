import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, List, Dict
from src.shared.enum_type import FactoryType


@dataclass
class GenerationRequest:
    """
    统一管理生成任务的所有参数，
    贯穿 UI → Client → Backend 整个链路。
    """

    model_type:   FactoryType
    model_name:   str
    prompt:       str                        # 用户原始输入

    translated:   Optional[str] = None      # 翻译后的提示词，None 表示未翻译

    attachments:  list[str] = field(default_factory=list)  # 文件路径列表

    model_params: dict[str, Any] = field(default_factory=dict)

    setting:      dict[str, Any] = field(default_factory=dict)

    session_id:   str = field(default_factory=lambda: str(uuid.uuid4()))
    request_id:   str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id:      Optional[str] = None

    output_dir:   Optional[str] = None      # None 时由后端自动生成

    def auto_output_dir(self) -> str:
        """自动生成输出目录"""
        if self.output_dir:
            return self.output_dir
        type_text = FactoryType.convert_to_text(self.model_type)
        return f"output_{type_text}/{self.model_name}"

    def to_api_payload(self) -> dict:
        return {
            "request_id":  self.request_id,
            "session_id":  self.session_id,
            "model_type":  FactoryType.convert_to_text(self.model_type),
            "model_name":  self.model_name,
            "extra": {
                "content":    self.translated or self.prompt ,
                "original":   self.prompt,
                "output_dir": self.auto_output_dir(),
                **self.model_params,        # 展开模型细节参数
            },
            "attachments": self.attachments,
            "setting":     self.setting,
        }

    def to_history_record(self) -> dict:
        return {
            "request_id": self.request_id,
            "session_id": self.session_id,
            "prompt":     self.prompt,
            "translated": self.translated,
            "model_type": FactoryType.convert_to_text(self.model_type),
            "model_name": self.model_name,
            "attachments": self.attachments,
        }

    @classmethod
    def build(
        cls,
        model_type:   FactoryType,
        model_name:   str,
        prompt:       str,
        model_params: dict[str, Any] = None,
        attachments:  list[str] = None,
        session_id:   str = None,
        user_id:      str = None,
        setting:      dict[str, Any] = None,
    ) -> "GenerationRequest":
        """
        标准构建入口，自动注入全局设置
        """

        req = cls(
            model_type   = model_type,
            model_name   = model_name,
            prompt       = prompt,
            model_params = model_params or {},
            attachments  = attachments or [],
            setting      = setting or {},
            user_id      = user_id or str(uuid.uuid4())
        )
        if session_id:
            req.session_id = session_id
        return req

    @classmethod
    def open_translate(cls, translated):
        print(f"翻译的内容：{translated}")
        if translated:
            cls.translated = translated
        else:
            cls.translated = ""


    @staticmethod
    def _sanitize_str(value: Any, max_len: int = 10000) -> str:
        """確保是字符串，去除首尾空白，限制長度"""
        if value is None:
            return ""
        return str(value).strip()[:max_len]

    @staticmethod
    def _sanitize_attachments(attachments: Any) -> List[str]:
        """
        過濾 attachments：
        - 必須是列表
        - 每項必須是字符串
        - 路徑必須存在且是文件
        - 擴展名必須在白名單內
        """
        ALLOWED_EXTENSIONS = {
            ".png", ".jpg", ".jpeg", ".webp", ".gif",  # 圖片
            ".mp4", ".mov", ".avi",  # 視頻
            ".mp3", ".wav", ".flac",  # 音頻
            ".txt", ".pdf",  # 文檔
        }
        if not attachments:
            return []
        if not isinstance(attachments, (list, tuple)):
            return []

        result = []
        for item in attachments:
            if not isinstance(item, str):
                continue
            item = item.strip()
            if not item:
                continue
            try:
                p = Path(item).resolve()
                if not p.exists() or not p.is_file():
                    print(f"⚠️ attachment 跳過（文件不存在）: {item}")
                    continue

                if p.suffix.lower() not in ALLOWED_EXTENSIONS:
                    print(f"⚠️ attachment 跳過（不支持的類型）: {p.suffix}")
                    continue
                result.append(str(p))
            except Exception as e:
                print(f"⚠️ attachment 跳過（路徑解析失敗）: {item} → {e}")
                continue
        return result

    @staticmethod
    def _sanitize_extra(extra: Any) -> Dict[str, Any]:
        ALLOWED_TYPES = (str, int, float, bool, list, tuple, dict, type(None))
        if not isinstance(extra, dict):
            return {}
        result = {}
        for k, v in extra.items():
            if not isinstance(k, str):
                continue
            if not isinstance(v, ALLOWED_TYPES):
                print(f"⚠️ extra[{k}] 跳過（不支持的類型: {type(v).__name__}）")
                continue
            result[k.strip()] = v
        return result

