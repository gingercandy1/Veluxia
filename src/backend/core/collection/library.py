"""资料库服务（ADR 0004）：资源包的创建、列出、读取、删除和执行。

每个资源包是 get_media_root()/library/<id>/ 下的一个文件夹，状态全部记在 manifest.json 里。
"""
import re
import shutil
import threading
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from src.backend.core.collection.executor import CollectionExecutor
from src.backend.core.collection.manifest import MANIFEST_NAME, load_manifest, save_manifest
from src.backend.core.collection.steps import CastVoice, registered_runners
from src.backend.core.collection.template import Template, load_template
from src.backend.core.model_utils import get_media_root
from src.shared.schemas import (
    ApproveStepRequest,
    CastMember,
    CollectionItem,
    CreatePackRequest,
    Manifest,
    ResetStepRequest,
    StepState,
)

LIBRARY_DIR_NAME = "library"
# 资源包 id 和条目 id 都会用作目录名：只允许安全字符，同时挡住 "../" 之类的路径穿越
_ID_PATTERN = re.compile(r"[A-Za-z0-9_-]{1,64}")
# 角色包模板里产出声线样本的步骤 id：对话包按它找角色的声音
CHARACTER_VOICE_STEP = "voice"
CHARACTER_VOICE_FIELD = "voice"

_running: set[str] = set()
_running_lock = threading.Lock()


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

    manifest = Manifest(
        id=uuid.uuid4().hex[:12],
        name=request.name,
        type=template.type,
        template=template.id,
        template_version=template.version,
        style=request.style,
        cast=request.cast,
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
        return CollectionExecutor(directory, template, cancel_event=cancel_event,
                                  cast=cast).run()


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
    if manifest.type != "character":
        raise ValueError(f"资源包 {manifest.name or pack_id} 不是角色包")
    return directory, _find_item(manifest, item_id)


def _resolve_cast_member(member: CastMember) -> CastVoice:
    if not member.character:
        return CastVoice(name=member.name, description=member.description)
    directory, item = _character_item(member.character)
    description = member.description or item.prompt
    voice_prompt = item.fields.get(CHARACTER_VOICE_FIELD, "")
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
