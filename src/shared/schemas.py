from typing import Any, Dict, List, Literal, Optional
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


# 资料库 / 资源包（ADR 0004）：manifest 是资源包进度的唯一来源，前后端都按这套结构解析
StepStatus = Literal["pending", "running", "done", "error"]
# 资源包分类，界面按这个顺序分区排列：按游戏里的东西分，而不是按文件类型分
PackType = Literal["scene", "character", "item", "effect", "dialogue", "audio"]
PACK_TYPES: tuple[str, ...] = ("scene", "character", "item", "effect", "dialogue", "audio")
# 剧本台词的情绪：后端写进 LLM 提示词，界面审阅剧本时做下拉选项，两边必须一致
DIALOGUE_EMOTIONS: tuple[str, ...] = ("平静", "开心", "生气", "悲伤", "惊讶", "害怕", "坚定")


class StepState(BaseModel):
    """一个条目在某一步上的状态。"""
    status: StepStatus = "pending"
    outputs: List[str] = Field(default_factory=list, description="产物路径，相对于资源包目录")
    error: Optional[str] = None
    meta: Dict[str, Any] = Field(
        default_factory=dict, description="生成细节：模型、实际提示词、参数、尺寸、耗时等")
    # 只对模板里标了 review 的步骤有意义：完成后要用户确认，下游才会继续执行
    approved: bool = False


class CollectionItem(BaseModel):
    """资源包里的一个待生成对象，逐一走完模板的各个步骤。"""
    id: str = Field(..., description="条目 ID，同时作为产物子目录名")
    prompt: str = ""
    fields: Dict[str, str] = Field(default_factory=dict, description="模板声明的附加字段")
    steps: Dict[str, StepState] = Field(default_factory=dict, description="步骤 id → 状态")


class CollectionStyle(BaseModel):
    """风格锁：自动拼到每个条目的提示词上。"""
    prompt: str = ""
    negative: str = ""


class CastMember(BaseModel):
    """对话包的出场角色。绑定了角色包条目时用它的设定和声线；否则按描述现场设计声音。"""
    name: str
    description: str = Field("", description="性格 / 音色描述；绑定角色包时可留空")
    character: str = Field("", description="绑定的角色包条目，格式 <资源包 id>/<条目 id>")


class Manifest(BaseModel):
    id: str
    name: str = ""
    type: PackType = "scene"
    template: str
    template_version: int = 1
    style: CollectionStyle = Field(default_factory=CollectionStyle)
    # 建包时选的项目级风格预设 id：style 是复制过来的内容，预设之后被改也不影响已生成的包
    style_preset: str = ""
    cast: List[CastMember] = Field(default_factory=list)
    # 模板声明了 source 时绑定的来源角色（ADR 0006），如动作包的角色，格式同 CastMember.character
    source: str = ""
    items: List[CollectionItem] = Field(default_factory=list)


class NewCollectionItem(BaseModel):
    id: str = Field("", description="留空时按序号自动生成")
    prompt: str = ""
    fields: Dict[str, str] = Field(default_factory=dict)


class CreatePackRequest(BaseModel):
    name: str = ""
    template: str
    style: CollectionStyle = Field(default_factory=CollectionStyle)
    style_preset: str = Field("", description="风格预设 id；style 须已填成该预设的内容")
    cast: List[CastMember] = Field(default_factory=list)
    source: str = Field("", description="来源角色 <资源包 id>/<条目 id>；只有模板声明了 source 才需要")
    items: List[NewCollectionItem] = Field(default_factory=list)


class StylePreset(BaseModel):
    """项目级风格预设（ADR 0006）：存在资料库目录里，建包时选用，保证不同资源包画风统一。"""
    id: str = Field("", description="留空表示新建")
    name: str
    prompt: str = ""
    negative: str = ""

    def style(self) -> CollectionStyle:
        return CollectionStyle(prompt=self.prompt, negative=self.negative)


class StyleListResponse(BaseResponse):
    styles: List[StylePreset] = Field(default_factory=list)


class DraftItemsRequest(BaseModel):
    """让文本模型按模板字段起草一批条目，填回新建表单由用户修改后再建包。"""
    template: str
    theme: str = Field("", description="用户给的主题，如“奥日风格的森林地面元素”")
    count: int = 10
    style: str = Field("", description="资源包的风格锁，只作参考")
    cast: List[CastMember] = Field(default_factory=list, description="对话包的出场角色")
    exclude: List[str] = Field(default_factory=list, description="表格里已有的条目，避免重复")
    model_name: str = Field("", description="文本模型；空表示用默认模型")


class DraftItemsResponse(BaseResponse):
    items: List[NewCollectionItem] = Field(default_factory=list)


class RunPackRequest(BaseModel):
    pack_id: str


class ApproveStepRequest(BaseModel):
    """确认一个待审阅的步骤（如 AI 写好的剧本），可同时提交修改后的内容。"""
    item_id: str
    step_id: str
    content: Optional[str] = Field(None, description="修改后的产物内容；None 表示原样确认")


class ResetStepRequest(BaseModel):
    """把某条目的某一步连同下游标记为待重做，下次执行时重新生成。"""
    item_id: str
    step_id: str


class FieldOption(BaseModel):
    value: str = Field(..., description="实际拼进参数的值")
    label: str = ""


class TemplateFieldInfo(BaseModel):
    """条目需要填写的附加字段（提示词之外），如角色的音色描述。"""
    id: str
    label: str = ""
    required: bool = False
    default: str = ""
    options: List[FieldOption] = Field(default_factory=list, description="非空时界面用下拉选择")


class TemplateStepInfo(BaseModel):
    id: str
    type: str
    label: str = ""
    deliverable: bool = Field(False, description="是否为最终交付的素材；否则是中间产物")
    review: bool = Field(False, description="完成后需用户确认，下游步骤才继续")
    inputs: List[str] = Field(default_factory=list, description="上游步骤 id，界面按它算素材的执行流程")


class TemplateInfo(BaseModel):
    id: str
    type: str
    name: str = ""
    description: str = ""
    prompt_label: str = Field("", description="条目主提示词在界面上的名称，如“外观描述”“台词”")
    cover: str = Field("", description="用作封面缩略图的步骤 id，空表示没有图片产物")
    source: str = Field("", description="需要绑定的来源资源包类型，如 character；空表示不需要")
    fields: List[TemplateFieldInfo] = Field(default_factory=list)
    steps: List[str] = Field(default_factory=list, description="步骤 id，按执行顺序")
    step_details: List[TemplateStepInfo] = Field(default_factory=list)


class TemplateListResponse(BaseResponse):
    templates: List[TemplateInfo] = Field(default_factory=list)


class PackResponse(BaseResponse):
    manifest: Optional[Manifest] = None
    media_base: str = Field("", description="资源包目录的媒体 URL，拼上产物相对路径即可访问")
    # 程序中途被关掉时 manifest 里可能残留 running，界面以这个字段判断是否真的在执行
    running: bool = False
    # 绑定的来源角色立绘在做完产物后被重做过（ADR 0006）
    source_changed: bool = False


class PackListResponse(BaseResponse):
    packs: List[PackResponse] = Field(default_factory=list)


# 模型权重的磁盘占用与清理
class ModelStorageEntry(BaseModel):
    id: str = Field(..., description="删除时按它定位，如 models/image/sdxl-base")
    root: str = Field("", description="存放位置：models / ace_step / rembg")
    category: str = Field("", description="models 下的模态目录，如 image、text")
    name: str
    size: int = Field(0, description="字节")
    manual: bool = Field(False, description="删了不会自动重新下载，需要手动放回")


class ModelStorageResponse(BaseResponse):
    entries: List[ModelStorageEntry] = Field(default_factory=list, description="按占用从大到小")
    total: int = 0


class DeleteModelStorageRequest(BaseModel):
    id: str
