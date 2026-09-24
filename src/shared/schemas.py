from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# 公共基类
class BaseRequest(BaseModel):
    """所有请求的公共字段"""
    model_name: str = Field(..., description="模型名称，需已注册到 GeneratorFactory")
    user_id: str = Field("user", description="用户 ID")
    session_id: str = Field("assistant", description="会话 ID，原样透传给 UI")

    extra: Dict[str, Any] = Field(default_factory=dict, description="模型特有的额外参数")
    setting: Dict[str, Any] = Field(default_factory=dict, description="設置中的參數")

class BaseResponse(BaseModel):
    """所有响应的公共字段"""
    ok: bool = True
    session_id: str = ""
    error: Optional[str] = None

    @classmethod
    def from_error(cls, msg: str, session_id: str = ""):
        return cls(ok=False, session_id=session_id, error=msg)

# Text（LLM / 文本生成）
class TextResponse(BaseResponse):
    content: str = ""

# Image（文生图 / 图生图）
class ImageResponse(BaseResponse):
    paths: List[str] = Field(default_factory=list, description="生成图片的绝对路径列表")

# Animation（图像 → 动画）
class AnimationResponse(BaseResponse):
    video_path: Optional[str] = None
    frame_paths: List[str] = Field(default_factory=list)
    export_paths: List[str] = Field(default_factory=list, description="精灵图 / atlas 等附加导出文件")


# 序列帧导出（精灵图 + atlas + 编号 PNG）
class SpriteSheetResponse(BaseResponse):
    sheet_path: Optional[str] = None
    atlas_path: Optional[str] = None
    frame_paths: List[str] = Field(default_factory=list, description="编号 PNG 序列")
    columns: int = 0
    rows: int = 0


# Speech（文本 → 语音 / 音乐）
class SpeechResponse(BaseResponse):
    audio_path: Optional[str] = None


# Transcription（语音 → 文本）
class TranscriptSegment(BaseModel):
    start: float = Field(..., description="起始时间（秒）")
    end: float = Field(..., description="结束时间（秒）")
    text: str = ""


class TranscriptionResponse(BaseResponse):
    text: str = ""
    language: str = ""
    model: str = Field("", description="实际执行识别的模型（Auto 会解析成具体模型）")
    segments: List[TranscriptSegment] = Field(default_factory=list)
    srt_path: Optional[str] = Field(None, description="SRT 字幕的媒体 URL；没识别出内容时为空")


# 模型信息（供 UI 初始化下拉列表）
class ModelInfoResponse(BaseResponse):
    """返回某个 FactoryType 下已注册的模型名称列表"""
    type: str = ""
    names: List[str] = Field(default_factory=list)
    tags: Dict[str, List[str]] = Field(default_factory=dict, description="tag → [name, ...]")

class TranslateResponse(BaseResponse):
    translate_result: str = ""

class RefineResponse(BaseResponse):
    refined: str = ""


# 异步生成任务（图片 / 动画 / 语音耗时不固定，提交后轮询状态）
class JobSubmitResponse(BaseModel):
    job_id: str
    status: str = "pending"


class JobStatusResponse(BaseModel):
    job_id: str
    status: str  # pending / running / done / error
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    partial: List[str] = Field(default_factory=list, description="已产出的中间结果（媒体 URL）")
