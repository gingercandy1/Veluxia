"""资源包的进度与分类计算（ADR 0004），不含控件，花园卡片、素材卡片、详情栏共用。

界面上的一张素材卡片 = 一个条目的一个交付步骤（如角色的立绘、声线）。
卡片进度只数这个交付物的执行流程（它和它的全部上游步骤），这样同一条目的立绘做完了，
不会因为声线还在排队而显示成未完成。
"""
from dataclasses import dataclass

from PySide6.QtCore import QT_TRANSLATE_NOOP, QCoreApplication

from src.shared.schemas import (
    PACK_TYPES,
    CollectionItem,
    Manifest,
    StepState,
    TemplateInfo,
    TemplateStepInfo,
)

_CONTEXT = "PackStatus"
CATEGORY_TEXT = {
    "scene": QT_TRANSLATE_NOOP("PackStatus", "Scenes"),
    "character": QT_TRANSLATE_NOOP("PackStatus", "Characters"),
    "item": QT_TRANSLATE_NOOP("PackStatus", "Items"),
    "effect": QT_TRANSLATE_NOOP("PackStatus", "Effects"),
    "dialogue": QT_TRANSLATE_NOOP("PackStatus", "Dialogue"),
    "audio": QT_TRANSLATE_NOOP("PackStatus", "Sound & music"),
}
MEDIA_ORDER = ("image", "video", "audio", "text")
MEDIA_TEXT = {
    "image": QT_TRANSLATE_NOOP("PackStatus", "Images"),
    "video": QT_TRANSLATE_NOOP("PackStatus", "Videos"),
    "audio": QT_TRANSLATE_NOOP("PackStatus", "Audio"),
    "text": QT_TRANSLATE_NOOP("PackStatus", "Scripts"),
}
# 卡片角标：review 是"等你确认"，比出错更需要用户动手，排在最前
STATUS_TEXT = {
    "review": QT_TRANSLATE_NOOP("PackStatus", "Needs review"),
    "running": QT_TRANSLATE_NOOP("PackStatus", "Running"),
    "error": QT_TRANSLATE_NOOP("PackStatus", "Failed"),
    "done": QT_TRANSLATE_NOOP("PackStatus", "Done"),
    "pending": QT_TRANSLATE_NOOP("PackStatus", "Pending"),
}
_STATUS_PRIORITY = ("review", "running", "error", "pending", "done")
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")
AUDIO_SUFFIXES = (".wav", ".mp3", ".flac", ".ogg")


def category_text(pack_type: str) -> str:
    return QCoreApplication.translate(_CONTEXT, CATEGORY_TEXT.get(pack_type, pack_type))


def media_text(kind: str) -> str:
    return QCoreApplication.translate(_CONTEXT, MEDIA_TEXT[kind])


def status_text(status: str) -> str:
    return QCoreApplication.translate(_CONTEXT, STATUS_TEXT[status])


def media_kind(step_type: str) -> str:
    """步骤类型前缀决定产物是什么媒体；未知类型当文本，至少能在详情里看到文件。"""
    prefix = step_type.split(".", 1)[0]
    if prefix == "image":
        return "image"
    if prefix == "animation":
        return "video"
    if prefix == "speech":
        return "audio"
    return "text"


def ordered_categories(packs_by_type: dict[str, list]) -> list[str]:
    """固定分类在前；模板将来新增的分类排在后面，不会被丢掉。"""
    extra = sorted(t for t in packs_by_type if t not in PACK_TYPES)
    return [*PACK_TYPES, *extra]


def fallback_template(manifest: Manifest) -> TemplateInfo:
    """模板已下线时按 manifest 里出现过的步骤拼一个：只能把最后一步当交付物、按顺序串起来。"""
    step_ids: list[str] = []
    for item in manifest.items:
        step_ids.extend(s for s in item.steps if s not in step_ids)
    details = [TemplateStepInfo(id=step_id, type="", inputs=step_ids[index - 1:index])
               for index, step_id in enumerate(step_ids)]
    if details:
        details[-1].deliverable = True
    return TemplateInfo(id=manifest.template, type=manifest.type, steps=step_ids,
                        step_details=details)


def deliverables(template: TemplateInfo) -> list[TemplateStepInfo]:
    marked = [step for step in template.step_details if step.deliverable]
    # 老模板没标交付物时，最后一步就是成品
    return marked or template.step_details[-1:]


def step_chain(template: TemplateInfo, step_id: str) -> list[TemplateStepInfo]:
    """step_id 和它的全部上游，按模板顺序排列：这就是详情栏里的"执行流程"。"""
    by_id = {step.id: step for step in template.step_details}
    needed = {step_id}
    for step in reversed(template.step_details):
        if step.id in needed:
            needed.update(step.inputs)
    return [step for step in template.step_details if step.id in needed and step.id in by_id]


def _state(item: CollectionItem, step_id: str) -> StepState:
    return item.steps.get(step_id) or StepState()


def _step_status(item: CollectionItem, step: TemplateStepInfo) -> str:
    state = _state(item, step.id)
    if step.review and state.status == "done" and not state.approved:
        return "review"
    return state.status


def merge_status(statuses: list[str]) -> str:
    # pending 排在 done 前面：全部完成才算 done，有完成也有排队的仍是 pending
    for status in _STATUS_PRIORITY:
        if status in statuses:
            return status
    return "pending"


@dataclass(frozen=True)
class Progress:
    done: int
    total: int
    status: str

    @property
    def percent(self) -> int:
        return round(self.done * 100 / self.total) if self.total else 0


def asset_progress(item: CollectionItem, chain: list[TemplateStepInfo]) -> Progress:
    statuses = [_step_status(item, step) for step in chain]
    done = sum(1 for step in chain if _state(item, step.id).status == "done")
    return Progress(done, len(chain), merge_status(statuses) if statuses else "pending")


def pack_progress(manifest: Manifest, template: TemplateInfo, running: bool = False) -> Progress:
    """卡片外层进度：done/total 数的是全部交付物都做完的条目；状态取全部步骤里最需要关注的那个。"""
    targets = deliverables(template)
    finished = sum(1 for item in manifest.items
                   if all(_state(item, step.id).status == "done" for step in targets))
    statuses = [_step_status(item, step) for item in manifest.items
                for step in template.step_details]
    status = merge_status(statuses) if statuses else "pending"
    if running and status != "review":
        status = "running"
    return Progress(finished, len(manifest.items), status)


def step_fraction(manifest: Manifest, template: TemplateInfo) -> int:
    """进度条用步骤完成比例，比"完成条目数"更平滑：长任务里也能看到它在动。"""
    total = len(manifest.items) * len(template.step_details)
    done = sum(1 for item in manifest.items for step in template.step_details
               if _state(item, step.id).status == "done")
    return round(done * 100 / total) if total else 0


def latest_image(item: CollectionItem, chain: list[TemplateStepInfo]) -> str:
    """流程里最靠后的已完成图片：成品还没出来时，卡片先显示中间结果。"""
    for step in reversed(chain):
        state = _state(item, step.id)
        if state.status == "done":
            images = [p for p in state.outputs if p.lower().endswith(IMAGE_SUFFIXES)]
            if images:
                return images[-1]
    return ""


def pack_cover(manifest: Manifest, template: TemplateInfo) -> str:
    """封面：模板指定的封面步骤里第一个已完成的图片；没有图片产物的包（对话、音效）返回空。"""
    cover_steps = [step for step in template.step_details if step.id == template.cover]
    if not cover_steps:
        cover_steps = [step for step in deliverables(template) if media_kind(step.type) == "image"]
    for step in cover_steps:
        for item in manifest.items:
            path = latest_image(item, [step])
            if path:
                return path
    return ""


def item_title(item: CollectionItem) -> str:
    """角色这类带名字字段的条目用名字，其余用提示词开头，都比条目 id 好认。"""
    name = item.fields.get("name", "").strip()
    if name:
        return name
    prompt = item.prompt.strip()
    return prompt if len(prompt) <= 24 else prompt[:23] + "…"
