"""资料库服务（ADR 0004）：资源包的创建、列出、读取、删除和执行。

每个资源包是 get_media_root()/library/<id>/ 下的一个文件夹，状态全部记在 manifest.json 里。
项目级风格预设存在同目录的 styles.json（ADR 0006），拷走资料库目录时预设一起带走。
"""
import os
import re
import shutil
import threading
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from pydantic import TypeAdapter

from src.backend.core.collection.executor import (
    SOURCE_META,
    CollectionExecutor,
    file_signature,
)
from src.backend.core.collection.manifest import MANIFEST_NAME, load_manifest, save_manifest
from src.backend.core.collection.steps import CastVoice, registered_runners
from src.backend.core.collection.template import SOURCE_INPUT, Template, load_template
from src.backend.core.model_utils import get_media_root
from src.shared.schemas import (
    ApproveStepRequest,
    CastMember,
    CollectionItem,
    CreatePackRequest,
    Manifest,
    ResetStepRequest,
    StepState,
    StylePreset,
)

LIBRARY_DIR_NAME = "library"
# 资源包 id 和条目 id 都会用作目录名：只允许安全字符，同时挡住 "../" 之类的路径穿越
_ID_PATTERN = re.compile(r"[A-Za-z0-9_-]{1,64}")
# 角色包模板里产出声线样本的步骤 id：对话包按它找角色的声音
CHARACTER_VOICE_STEP = "voice"
CHARACTER_VOICE_FIELD = "voice"
CHARACTER_GENDER_FIELD = "gender"
# 角色包里产出裁边立绘的步骤 id：动作包等绑定来源角色的包从它取立绘（ADR 0006）
CHARACTER_PORTRAIT_STEP = "trim"

STYLES_NAME = "styles.json"
_STYLE_LIST = TypeAdapter(list[StylePreset])

_running: set[str] = set()
_running_lock = threading.Lock()
_styles_lock = threading.Lock()


class PackBusyError(Exception):
    """资源包正在执行：不能重复执行，也不能删除。"""


def library_root() -> Path:
    root = get_media_root() / LIBRARY_DIR_NAME
    root.mkdir(parents=True, exist_ok=True)
    return root


def pack_dir(pack_id: str) -> Path:
    _check_id(pack_id, "资源包")
    return library_root() / pack_id


def is_running(pack_id: str) -> bool:
    with _running_lock:
        return pack_id in _running


def list_packs() -> list[Manifest]:
    """按最近修改排序；单个资源包的 manifest 损坏时跳过它，不影响列出其他资源包。"""
    manifests = []
    paths = sorted(library_root().glob(f"*/{MANIFEST_NAME}"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    for path in paths:
        try:
            manifests.append(load_manifest(path.parent))
        except ValueError as exc:
            print(f"⚠️ 资源包 {path.parent.name} 的 manifest 无法解析，已跳过：{exc}")
    return manifests


def get_pack(pack_id: str) -> Manifest:
    return load_manifest(pack_dir(pack_id))


def create_pack(request: CreatePackRequest) -> Manifest:
    template = load_template(request.template)
    if not request.items:
        raise ValueError("资源包至少需要一个条目")

    items = []
    for index, new_item in enumerate(request.items, start=1):
        item_id = new_item.id or f"{index:02d}"
        _check_id(item_id, "条目")
        fields = {key: value.strip() for key, value in new_item.fields.items() if value.strip()}
        try:
            template.check_item_fields(fields)
        except ValueError as exc:
            raise ValueError(f"条目 {item_id}：{exc}") from exc
        items.append(CollectionItem(id=item_id, prompt=new_item.prompt, fields=fields))
    ids = [item.id for item in items]
    duplicated = sorted({item_id for item_id in ids if ids.count(item_id) > 1})
    if duplicated:
        raise ValueError(f"条目 id 重复：{', '.join(duplicated)}")
    _check_cast(template, request.cast)
    _check_source(template, request.source)
    if request.style_preset:
        _find_style(list_styles(), request.style_preset)

    manifest = Manifest(
        id=uuid.uuid4().hex[:12],
        name=request.name,
        type=template.type,
        template=template.id,
        template_version=template.version,
        style=request.style,
        style_preset=request.style_preset,
        cast=request.cast,
        source=request.source,
        items=items,
    )
    directory = pack_dir(manifest.id)
    directory.mkdir()
    save_manifest(directory, manifest)
    return manifest


def delete_pack(pack_id: str) -> None:
    directory = pack_dir(pack_id)
    if not (directory / MANIFEST_NAME).is_file():
        raise FileNotFoundError(f"资源包不存在：{pack_id}")
    # 持锁删除：避免删到一半时同一个资源包又被提交执行
    with _running_lock:
        if pack_id in _running:
            raise PackBusyError(f"资源包 {pack_id} 正在执行，先停止再删除")
        shutil.rmtree(directory)


def run_pack(pack_id: str, cancel_event: threading.Event) -> Manifest:
    """阻塞执行整个资源包；runner 内部会 asyncio.run，必须在没有事件循环的线程里调用。"""
    directory = pack_dir(pack_id)
    manifest = load_manifest(directory)
    template = load_template(manifest.template)
    with _claim(pack_id):
        # 每次执行时现解析：角色包的声线可能是创建对话包之后才生成或重做的
        cast = [_resolve_cast_member(member) for member in manifest.cast]
        source = _source_portrait(manifest.source) if template.source else None
        return CollectionExecutor(directory, template, cancel_event=cancel_event,
                                  cast=cast, source=source).run()


def approve_step(pack_id: str, request: ApproveStepRequest) -> Manifest:
    """确认需要审阅的步骤产物；带 content 时先把用户修改后的内容写回产物文件。"""
    directory = pack_dir(pack_id)
    with _claim(pack_id):
        manifest = load_manifest(directory)
        template = load_template(manifest.template)
        step = template.step(request.step_id)
        if not step.review:
            raise ValueError(f"步骤 {step.id} 不需要审阅")
        item = _find_item(manifest, request.item_id)
        state = item.steps.get(step.id)
        if state is None or state.status != "done" or not state.outputs:
            raise ValueError(f"条目 {item.id} 的步骤 {step.id} 还没有完成，无法确认")

        if request.content is not None:
            text = registered_runners()[step.type].validate_edit(request.content)
            (directory / state.outputs[0]).write_text(text, encoding="utf-8")
            state.meta = {**state.meta, "edited": True}
            # 内容改了，基于旧内容做出来的下游产物都作废
            _reset_downstream(template, item, step.id)
        state.approved = True
        save_manifest(directory, manifest)
        return manifest


def reset_step(pack_id: str, request: ResetStepRequest) -> Manifest:
    """把某条目的某一步标记为待重做（连同下游），下次执行时重新生成。"""
    directory = pack_dir(pack_id)
    with _claim(pack_id):
        manifest = load_manifest(directory)
        template = load_template(manifest.template)
        step = template.step(request.step_id)
        item = _find_item(manifest, request.item_id)
        item.steps[step.id] = StepState()
        _reset_downstream(template, item, step.id)
        save_manifest(directory, manifest)
        return manifest


def source_changed(manifest: Manifest) -> bool:
    """来源角色的立绘在做完这些产物之后被重做过。
    立绘暂时不在（被删、正在重做）时不提示：那时重做也拿不到新立绘，执行会直接报错说明原因。"""
    return bool(_stale_source_steps(manifest))


def refresh_source(pack_id: str) -> Manifest:
    """把用旧立绘做的步骤（连同下游）标记为待重做，下次执行时按新立绘重新生成。"""
    directory = pack_dir(pack_id)
    with _claim(pack_id):
        manifest = load_manifest(directory)
        stale = _stale_source_steps(manifest)
        if not stale:
            raise ValueError("来源角色的立绘没有变化，不需要重做")
        template = load_template(manifest.template)
        for item, step_id in stale:
            item.steps[step_id] = StepState()
            _reset_downstream(template, item, step_id)
        save_manifest(directory, manifest)
        return manifest


def _stale_source_steps(manifest: Manifest) -> list[tuple[CollectionItem, str]]:
    if not manifest.source:
        return []
    try:
        current = file_signature(_source_portrait(manifest.source))
        template = load_template(manifest.template)
    except (ValueError, FileNotFoundError):
        return []
    steps = [step.id for step in template.steps if SOURCE_INPUT in step.inputs]
    stale = []
    for item in manifest.items:
        for step_id in steps:
            state = item.steps.get(step_id)
            # 没记录版本的（ADR 0006 之前做的）无从比较，不提示
            recorded = state.meta.get(SOURCE_META) if state else None
            if state and state.status == "done" and recorded and recorded != current:
                stale.append((item, step_id))
    return stale


def sync_style(pack_id: str) -> Manifest:
    """把资源包的风格锁更新成它所选预设的当前内容。
    已生成的产物不动：要不要按新风格重做由用户逐项决定。"""
    directory = pack_dir(pack_id)
    with _claim(pack_id):
        manifest = load_manifest(directory)
        if not manifest.style_preset:
            raise ValueError("这个资源包没有使用风格预设")
        manifest.style = _find_style(list_styles(), manifest.style_preset).style()
        save_manifest(directory, manifest)
        return manifest


def list_styles() -> list[StylePreset]:
    path = library_root() / STYLES_NAME
    if not path.is_file():
        return []
    return _STYLE_LIST.validate_json(path.read_text(encoding="utf-8"))


def save_style(preset: StylePreset) -> list[StylePreset]:
    """新建（id 为空）或修改预设，返回全部预设。"""
    name = preset.name.strip()
    if not name:
        raise ValueError("风格预设需要名字")
    with _styles_lock:
        styles = list_styles()
        if any(s.name == name and s.id != preset.id for s in styles):
            raise ValueError(f"已有同名的风格预设：{name}")
        saved = preset.model_copy(update={"name": name, "id": preset.id or uuid.uuid4().hex[:8]})
        if preset.id:
            index = styles.index(_find_style(styles, preset.id))
            styles[index] = saved
        else:
            styles.append(saved)
        _write_styles(styles)
        return styles


def delete_style(style_id: str) -> list[StylePreset]:
    """删除预设不影响已用它建好的资源包：包里存的是风格内容的副本。"""
    with _styles_lock:
        styles = list_styles()
        remaining = [s for s in styles if s.id != style_id]
        if len(remaining) == len(styles):
            raise FileNotFoundError(f"风格预设不存在：{style_id}")
        _write_styles(remaining)
        return remaining


def _find_style(styles: list[StylePreset], style_id: str) -> StylePreset:
    for style in styles:
        if style.id == style_id:
            return style
    raise ValueError(f"风格预设不存在：{style_id}")


def _write_styles(styles: list[StylePreset]) -> None:
    path = library_root() / STYLES_NAME
    temp = path.with_suffix(".json.tmp")
    temp.write_bytes(_STYLE_LIST.dump_json(styles, indent=2))
    os.replace(temp, path)


def _reset_downstream(template: Template, item: CollectionItem, step_id: str) -> None:
    for downstream in template.downstream_of(step_id):
        item.steps[downstream] = StepState()


def _find_item(manifest: Manifest, item_id: str) -> CollectionItem:
    for item in manifest.items:
        if item.id == item_id:
            return item
    raise ValueError(f"资源包 {manifest.id} 没有条目：{item_id}")


def _check_cast(template: Template, cast: list[CastMember]) -> None:
    if template.type != "dialogue":
        if cast:
            raise ValueError("只有对话包可以设置出场角色")
        return
    if not cast:
        raise ValueError("对话包至少需要一个出场角色")
    names = [member.name.strip() for member in cast]
    if any(not name for name in names):
        raise ValueError("出场角色的名字不能为空")
    duplicated = sorted({name for name in names if names.count(name) > 1})
    if duplicated:
        raise ValueError(f"出场角色重名：{', '.join(duplicated)}")
    for member in cast:
        if member.character:
            _character_item(member.character)
        elif not member.description.strip():
            raise ValueError(f"出场角色 {member.name} 需要选择角色包里的角色，或填写角色描述")


def _character_item(reference: str) -> tuple[Path, CollectionItem]:
    """解析 "<角色包 id>/<条目 id>"，返回角色包目录和对应条目。"""
    pack_id, _, item_id = reference.partition("/")
    directory = pack_dir(pack_id)
    try:
        manifest = load_manifest(directory)
    except FileNotFoundError as exc:
        # 引用的角色包被删了属于请求内容有误，不是对话包本身不存在
        raise ValueError(f"找不到角色包：{pack_id}") from exc
    # 动作包也归在角色分类下，但它的条目是动作而不是角色，不能再被引用
    if manifest.type != "character" or manifest.source:
        raise ValueError(f"资源包 {manifest.name or pack_id} 不是角色包")
    return directory, _find_item(manifest, item_id)


def _check_source(template: Template, source: str) -> None:
    if not template.source:
        if source:
            raise ValueError(f"模板 {template.id} 不需要绑定来源角色")
        return
    if not source:
        raise ValueError("这类资源包需要先选择一个角色")
    _character_item(source)


def _source_portrait(reference: str) -> Path:
    """执行时现取来源角色的立绘：立绘可能是建包之后才生成或重做的。
    还没生成时直接报错提示用户，而不是让整个包静默地停在待执行。"""
    directory, item = _character_item(reference)
    state = item.steps.get(CHARACTER_PORTRAIT_STEP)
    if state and state.status == "done" and state.outputs:
        portrait = directory / state.outputs[-1]
        if portrait.is_file():
            return portrait
    raise ValueError(f"角色 {item.prompt[:20] or item.id} 的立绘还没有生成，请先执行角色包")


def _resolve_cast_member(member: CastMember) -> CastVoice:
    if not member.character:
        return CastVoice(name=member.name, description=member.description)
    directory, item = _character_item(member.character)
    description = member.description or item.prompt
    # 性别不写进描述时，现场设计出的声音性别是随机的
    voice_prompt = "，".join(filter(None, (item.fields.get(CHARACTER_GENDER_FIELD, ""),
                                          item.fields.get(CHARACTER_VOICE_FIELD, ""))))
    state = item.steps.get(CHARACTER_VOICE_STEP)
    sample_text = state.meta.get("prompt", "") if state else ""
    # 克隆需要样本音频和它念的文本；角色的声线还没生成时退回按描述现场设计
    if state and state.status == "done" and state.outputs and sample_text:
        sample = directory / state.outputs[0]
        if sample.is_file():
            return CastVoice(member.name, description, voice_prompt, sample, sample_text)
    print(f"⚠️ 角色 {member.name} 还没有声线样本，将按描述现场设计声音")
    return CastVoice(member.name, description, voice_prompt)


@contextmanager
def _claim(pack_id: str) -> Iterator[None]:
    """同一个资源包同时只允许一个执行者，否则两个线程会交替覆盖同一份 manifest。"""
    with _running_lock:
        if pack_id in _running:
            raise PackBusyError(f"资源包 {pack_id} 已在执行")
        _running.add(pack_id)
    try:
        yield
    finally:
        with _running_lock:
            _running.discard(pack_id)


def _check_id(value: str, label: str) -> None:
    if not _ID_PATTERN.fullmatch(value):
        raise ValueError(f"{label} id 只能包含字母、数字、下划线和短横线：{value!r}")
