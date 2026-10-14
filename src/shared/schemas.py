from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, field_validator


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
    progress: Optional[Dict[str, Any]] = Field(default=None, description="当前这一项的进度")


# 资料库 / 资源包（ADR 0004）：manifest 是资源包进度的唯一来源，前后端都按这套结构解析
# skipped：模板里带条件（when）的步骤在这个条目上不需要执行（如"不分层"时的分层步骤）。
# 单独成一个状态而不是停在 pending：否则进度永远到不了 100%，界面也会一直显示"待执行"
StepStatus = Literal["pending", "running", "done", "error", "skipped"]
# 条件步骤的字段取这些值时视为"不执行"，前后端共用同一个判断
INACTIVE_FIELD_VALUES: tuple[str, ...] = ("", "0")


def condition_met(value: str) -> bool:
    """模板步骤的 when 条件：引用的字段有值、且不是 "0"（如分层字段选了"不分层"）。"""
    return value.strip() not in INACTIVE_FIELD_VALUES


# when 前加 "!" 表示取反：同一个字段在两条互斥的流程间二选一（如道具的单张 / 组图）
WHEN_NOT = "!"


def when_field(when: str) -> str:
    """when 条件引用的字段 id（去掉取反前缀）。"""
    return when.removeprefix(WHEN_NOT)


def when_met(when: str, value: str) -> bool:
    """按字段值判断带 when 条件的步骤要不要执行；value 是 when_field(when) 字段的值。"""
    return condition_met(value) != when.startswith(WHEN_NOT)# 资源包分类，界面按这个顺序分区排列：按游戏里的东西分，而不是按文件类型分
PackType = Literal["scene", "character", "item", "effect", "dialogue", "audio"]
PACK_TYPES: tuple[str, ...] = ("scene", "character", "item", "effect", "dialogue", "audio")
# 模板字段的类型：图片字段由界面选本地文件，后端建包时复制进资源包
IMAGE_FIELD = "image"
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


class SeamCandidate(BaseModel):
    id: str
    file: str = Field(..., description="补丁文件，相对资源包目录")
    seed: int
    prompt: str
    left_sig: str
    right_sig: str
    harmonize: float


class SeamState(BaseModel):
    candidates: List[SeamCandidate] = Field(default_factory=list)
    adopted: str = ""


class MergedLayer(BaseModel):
    """用户把一段里的几层合成的新图层：原层不动，重做这一段时一并作废。"""
    id: str = Field(..., description="M1、M2…；原层的编号是 L1…L4")
    path: str = Field(..., description="相对资源包目录")
    sources: List[str] = Field(default_factory=list, description="被合并的层编号，按叠放顺序")
    z: int = Field(..., description="叠放位置：取被合并层里最上面那层的位置，原层 L1…L4 是 1…4")


class SegmentLayers(BaseModel):
    """一段的分层结果。按段的身份（起点 + 宽度）存：拖动一条切线只让相邻两段失效。"""
    source_sig: str = Field(..., description="分层时的 cuts_sig；对不上说明合成图变了")
    prompt: str = ""
    embeds: str = Field("", description="编码文件，相对资源包目录；只重做去噪时复用")
    layers: List[str] = Field(default_factory=list, description="从底到顶的透明图层，相对资源包目录")
    merged: List[MergedLayer] = Field(default_factory=list, description="用户合并出来的层")


class ImageVersion(BaseModel):
    """某个目标（屏 / 段 / 层）的一个历史版本，由外扩或文字改图产生。"""
    file: str = Field(..., description="相对资源包目录")
    kind: str = Field(..., description="extend 或 edit")
    prompt: str = ""


class VersionHistory(BaseModel):
    base_sig: str = Field("", description="建立历史时“原图”的签名；对不上整条历史作废（原图被重新生成了）")
    versions: List[ImageVersion] = Field(default_factory=list)
    current: int = Field(-1, description="-1 表示用原图")


class RingState(BaseModel):
    order: List[str] = Field(default_factory=list)
    # 目标键 -> 版本历史；键格式见 docs/tasks/ring-canvas/plan.md（screen:<id> / segment:<k> / layer:<k>:<Lx>）
    versions: Dict[str, VersionHistory] = Field(default_factory=dict)
    # 自动切割的建议，确认前不影响 cuts
    suggested_cuts: List[int] = Field(default_factory=list)
    seams: Dict[str, SeamState] = Field(default_factory=dict)
    # 切段：长图（按 order 拼起来的各屏 ring.png）上的 x 坐标，升序；每段交给 Qwen-Layered 分层
    cuts: List[int] = Field(default_factory=list)
    cuts_sig: str = Field("", description="切点对应的各屏 ring.png 签名；对不上说明图变了，切点已失效")
    segments: List[str] = Field(default_factory=list, description="确认切段后导出的段图，相对资源包目录")
    # order 里同一个条目可以出现多次（用户手动排的）；键是 "起点+宽度"（长图像素），见 SegmentLayers
    segment_layers: Dict[str, SegmentLayers] = Field(default_factory=dict)


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
    # 旧 manifest 不包含此字段，保持可选以兼容既有资源包
    ring: Optional[RingState] = None


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


class AddItemsRequest(BaseModel):
    """往已有资源包里追加条目：沿用包的模板、风格锁、出场角色和来源角色，只补新条目。"""
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
    source: str = Field("", description="动作包绑定的来源角色，格式同 CreatePackRequest.source；"
                                        "起草时按它的描述写动作")
    model_name: str = Field("", description="文本模型；空表示用默认模型")


class DraftItemsResponse(BaseResponse):
    items: List[NewCollectionItem] = Field(default_factory=list)


class RunPackRequest(BaseModel):
    pack_id: str


class RingGenerateRequest(BaseModel):
    pack_id: str
    keys: Optional[List[str]] = Field(
        None, description="为空时生成尚未采用或已失效的接缝")
    prompt: Optional[str] = Field(None, description="覆盖按条目提示词、风格锁和视角拼出的默认提示词")
    extra_prompt: Optional[str] = Field(
        None, description="追加在默认提示词后面的一句话；同时给了 prompt 时以 prompt 为准")


class RingPlanRequest(BaseModel):
    item_ids: Optional[List[str]] = Field(
        None, description="参与循环的条目；为空时沿用上次规划的成员，从没规划过则用全部条目")


class RingOrderRequest(BaseModel):
    order: List[str] = Field(
        ..., description="环上从左到右的条目 id；同一个条目可以出现多次，至少两个")


class AdoptSeamRequest(BaseModel):
    key: str
    candidate_id: str


class RingSeamStatus(BaseModel):
    candidates: List[SeamCandidate] = Field(default_factory=list)
    adopted: str = ""
    stale: bool = False
    has_candidates: bool = False


class RingStatusResponse(BaseResponse):
    order: List[str] = Field(default_factory=list)
    seams: Dict[str, RingSeamStatus] = Field(default_factory=dict)
    # 与 order 一一对应的合成图（相对资源包目录）；还没合成的位置是空串
    rendered: List[str] = Field(default_factory=list)
    cuts: List[int] = Field(default_factory=list)
    cuts_stale: bool = False
    segments: List[str] = Field(default_factory=list)
    # 与 segments 一一对应：该段已完成的图层（从底到顶）；没分层或已失效为空列表
    segment_layers: List[List[str]] = Field(default_factory=list)
    # 与 segments 一一对应：该段的合并层；没分层或已失效为空列表
    segment_merged: List[List[MergedLayer]] = Field(default_factory=list)
    # 与 order 一一对应：每屏当前用的图（原图 / 外扩 / 改过的版本），相对资源包目录
    screen_files: List[str] = Field(default_factory=list)
    # 与 order 一一对应：补齐到 ring_height 时上下共补了多少像素（0 = 不需要）
    screen_padding: List[int] = Field(default_factory=list)
    ring_height: int = 0
    suggested_cuts: List[int] = Field(default_factory=list)
    # 只含有效的历史（base_sig 对得上）
    versions: Dict[str, VersionHistory] = Field(default_factory=dict)


class RingCutsRequest(BaseModel):
    pack_id: str


class RingExtendRequest(BaseModel):
    pack_id: str
    item_ids: Optional[List[str]] = Field(None, description="要外扩的屏；为空表示环上全部屏")
    # 范围在 schema 里卡住：越界直接 422，不会等到加载模型之后才发现
    pad: int = Field(128, ge=16, le=256, description="上下各补的像素")


class RingEditRequest(BaseModel):
    pack_id: str
    target: str = Field(..., description="目标键：screen:<id> / segment:<k> / layer:<k>:<Lx>")
    prompt: str
    bbox: Optional[List[float]] = Field(
        None, description="归一化框选 [x, y, w, h]，0~1；给了就只改框里的画面，框外原样保留")

    @field_validator("bbox")
    @classmethod
    def _check_bbox(cls, value: Optional[List[float]]) -> Optional[List[float]]:
        # 越界直接 422，不会等到加载十几 GB 的模型之后才发现
        if value is None:
            return None
        if len(value) != 4:
            raise ValueError("框选需要 4 个数 [x, y, w, h]")
        x, y, w, h = (float(v) for v in value)
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0
                and x + w <= 1.0 + 1e-6 and y + h <= 1.0 + 1e-6):
            raise ValueError(f"框选越界：{[x, y, w, h]}（取值 0~1）")
        return [x, y, w, h]


class RingVersionRequest(BaseModel):
    target: str
    index: int = Field(..., description="-1 切回原图，否则是 versions 的下标")


class RingLayersRequest(BaseModel):
    pack_id: str
    indices: Optional[List[int]] = Field(
        None, description="要重做的段序号（从 0 起）；为空时只做还没分层或已失效的段")


class RingLayerProgress(BaseModel):
    """分层任务经 job.progress 轮询回传的进度；序号都是段在 ring.segments 里的位置。"""
    stage: str = Field(description="encode 或 denoise")
    index: int = Field(description="正在处理的段")
    remaining: List[int] = Field(description="包含当前段在内、还没做完的段")


class ConfirmCutsRequest(BaseModel):
    cuts: List[int]


class MergeLayersRequest(BaseModel):
    segment: int = Field(..., description="段序号（从 0 起）")
    layer_ids: List[str] = Field(..., description="要合并的层编号（L1…L4、M1…），至少两个")


class DeleteMergedRequest(BaseModel):
    segment: int = Field(..., description="段序号（从 0 起）")
    layer_id: str = Field(..., description="要删除的合并层编号（M1…）")


class ApproveStepRequest(BaseModel):
    """确认一个待审阅的步骤（如 AI 写好的剧本），可同时提交修改后的内容。"""
    item_id: str
    step_id: str
    content: Optional[str] = Field(None, description="修改后的产物内容；None 表示原样确认")


class ResetStepRequest(BaseModel):
    """把某条目的某一步连同下游标记为待重做，下次执行时重新生成。"""
    item_id: str
    step_id: str


class UpdateItemFieldsRequest(BaseModel):
    """事后修改条目的字段（如给已生成的场景补做分层）：依赖这些字段的步骤连同下游会被标记为待重做。"""
    item_id: str
    fields: Dict[str, str] = Field(default_factory=dict, description="要改的字段，未列出的保持不变")


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
    segments_field: str = Field("", description="这是分段描述字段时，对应的屏数字段 id："
                                                 "屏数大于 1 时这个字段才有意义，界面据此显示或隐藏")
    kind: str = Field("", description="字段类型：空是文本，image 是用户导入的本地图片路径")


class TemplateStepInfo(BaseModel):
    id: str
    type: str
    label: str = ""
    deliverable: bool = Field(False, description="是否为最终交付的素材；否则是中间产物")
    review: bool = Field(False, description="完成后需用户确认，下游步骤才继续")
    inputs: List[str] = Field(default_factory=list, description="上游步骤 id，界面按它算素材的执行流程")
    when: str = Field("", description="条件字段 id：该字段不满足 condition_met 时这一步跳过；"
                                        "前缀 ! 表示满足时跳过；空表示总是执行")
    overlay_of: str = Field("", description="非空时这一步是该交付步骤的附加视图（如整图的分层）："
                                              "界面不单独出卡片，挂在那张卡片上切换")


class TemplatePresetInfo(BaseModel):
    """模板自带的常用条目（如角色动作里的待机、走、跳），新建表单一键填进表格。"""
    name: str
    prompt: str
    fields: Dict[str, str] = Field(default_factory=dict)


class TemplateInfo(BaseModel):
    id: str
    type: str
    name: str = ""
    description: str = ""
    summary: str = Field("", description="一句话简介，新建界面常驻显示；完整说明放在悬停提示里")
    prompt_label: str = Field("", description="条目主提示词在界面上的名称，如“外观描述”“台词”")
    cover: str = Field("", description="用作封面缩略图的步骤 id，空表示没有图片产物")
    source: str = Field("", description="需要绑定的来源资源包类型，如 character；空表示不需要")
    fields: List[TemplateFieldInfo] = Field(default_factory=list)
    presets: List[TemplatePresetInfo] = Field(default_factory=list)
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
    parts: List[str] = Field(default_factory=list,
                             description="这一条包含的目录 / 文件名，附属目录（如 -gguf）并进主目录")


class ModelStorageResponse(BaseResponse):
    entries: List[ModelStorageEntry] = Field(default_factory=list, description="按占用从大到小")
    total: int = 0


class DeleteModelStorageRequest(BaseModel):
    id: str
