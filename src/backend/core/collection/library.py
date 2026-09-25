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
from src.backend.core.collection.template import load_template
from src.backend.core.model_utils import get_media_root
from src.shared.schemas import CollectionItem, CreatePackRequest, Manifest

LIBRARY_DIR_NAME = "library"
# 资源包 id 和条目 id 都会用作目录名：只允许安全字符，同时挡住 "../" 之类的路径穿越
_ID_PATTERN = re.compile(r"[A-Za-z0-9_-]{1,64}")

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
        items.append(CollectionItem(id=item_id, prompt=new_item.prompt))
    ids = [item.id for item in items]
    duplicated = sorted({item_id for item_id in ids if ids.count(item_id) > 1})
    if duplicated:
        raise ValueError(f"条目 id 重复：{', '.join(duplicated)}")

    manifest = Manifest(
        id=uuid.uuid4().hex[:12],
        name=request.name,
        type=template.type,
        template=template.id,
        template_version=template.version,
        style=request.style,
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
        return CollectionExecutor(directory, template, cancel_event=cancel_event).run()


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
