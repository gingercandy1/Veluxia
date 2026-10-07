"""模型权重的磁盘占用与清理。

权重都是首次使用时自动下载的，全装下来要一两百 GB，但以前应用里看不到哪个模型占了多少、
不用的也只能去目录里手动删。这里把各个存放位置按"一个模型一个目录（或文件）"列出来，
删掉后下次用到时会重新下载；少数需要手动准备的权重（插帧）会标出来，删之前要提醒用户。

各生成器的目录命名并不统一（有的按模型名，有的按仓库，几个模型也可能共用一个底座目录），
所以这里不去猜"目录属于哪个模型"，而是按存放位置的层级列出，删除只允许删列出来的条目。
唯一的合并规则：一个模型拆成主目录加 `主目录名-后缀` 的附属目录（如 z-image-turbo 与
z-image-turbo-gguf / -base / -controlnet、wan2 与 wan2-gguf），附属目录并进主目录算一条，
删除时整组一起删——只删附属目录模型照样用不了，分开列只会让同一个模型占好几行。
"""
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from src.backend.core.model_base import GeneratorFactory, SingletonMeta
from src.backend.core.model_utils import models_root

# models/ 下按模态分的目录：列它们的下一级；其他直接放在 models/ 下的（如记忆用的向量模型）整个算一条
_CATEGORY_DIRS = {"text", "image", "speech", "animation", "image_frame", "transcription",
                  "upscale", "lora", "prompt_refiner"}
# 插帧权重和 RIFE 仓库要手动准备（film_generator 缺了只报错不下载），删了就得重新手动放回去
_MANUAL_CATEGORIES = {"image_frame"}


def _roots() -> list[tuple[str, Path]]:
    """(条目 id 前缀, 目录)。ACE-Step 和 rembg 的权重不在 models/ 下，也一并列出来。"""
    from src.backend.core.speech.ace_step_music import ACE_STEP_ROOT
    rembg_home = Path(os.environ.get("U2NET_HOME", Path.home() / ".u2net"))
    return [("models", models_root()), ("ace_step", ACE_STEP_ROOT / "checkpoints"),
            ("rembg", rembg_home)]


@dataclass(frozen=True)
class StorageEntry:
    id: str          # 形如 models/image/sdxl-base，删除时按它定位
    root: str        # models / ace_step / rembg
    category: str    # models 下的模态目录；其他位置为空
    name: str
    paths: tuple[Path, ...]  # 主目录在前，后面是并进来的附属目录
    manual: bool     # 删了不会自动重新下载


def _size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    total = 0
    for folder, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += (Path(folder) / name).stat().st_size
            except OSError:
                pass  # 下载中被移走的临时文件
    return total


def _children(folder: Path) -> list[Path]:
    # .cache 是 huggingface 下载时的元数据，跟着模型目录走，不单独列
    return sorted(p for p in folder.iterdir() if not p.name.startswith("."))


def _grouped(items: list[Path]) -> list[tuple[Path, ...]]:
    """同一层里 `xxx-后缀` 并进 `xxx`；能并进多个时取最短的，整串同前缀的目录落到同一组。"""
    names = {item.name for item in items}
    groups: dict[str, list[Path]] = {}
    for item in items:
        bases = [n for n in names if n != item.name and item.name.startswith(n + "-")]
        groups.setdefault(min(bases, key=len) if bases else item.name, []).append(item)
    # 主目录排第一个，条目的 id / 名字取它
    return [tuple(sorted(paths, key=lambda p: p.name != base)) for base, paths in groups.items()]


def list_entries() -> list[StorageEntry]:
    entries = []
    for root_id, root in _roots():
        if not root.is_dir():
            continue
        top_level = []
        for child in _children(root):
            if root_id == "models" and child.name in _CATEGORY_DIRS and child.is_dir():
                for paths in _grouped(_children(child)):
                    entries.append(StorageEntry(
                        f"{root_id}/{child.name}/{paths[0].name}", root_id, child.name,
                        paths[0].name, paths, manual=child.name in _MANUAL_CATEGORIES))
            else:
                top_level.append(child)
        for paths in _grouped(top_level):
            entries.append(StorageEntry(f"{root_id}/{paths[0].name}", root_id, "", paths[0].name,
                                        paths, manual=False))
    return entries


def list_storage() -> list[tuple[StorageEntry, int]]:
    """按占用从大到小；大小每次现算，模型下载或删除后立刻反映出来。"""
    sized = [(entry, sum(_size(path) for path in entry.paths)) for entry in list_entries()]
    return sorted(sized, key=lambda pair: pair[1], reverse=True)


def delete_entry(entry_id: str) -> None:
    """只接受 list_entries 里列出的 id，传进来的字符串本身不拼成路径，防止删到别处。

    删之前先卸掉所有已加载的模型：Windows 下已映射的权重文件删不掉，
    也避免删完后驻留的模型还指着已经不存在的文件。有任务在跑时直接拒绝（GeneratorBusyError）。
    """
    entry = next((e for e in list_entries() if e.id == entry_id), None)
    if entry is None:
        raise FileNotFoundError(f"没有这个模型文件：{entry_id}")
    with GeneratorFactory.exclusive("清理模型文件"):
        # exclusive 只卸占显存的模型；抠图、语音识别这类 CPU 模型也可能开着这些文件
        for instance in list(SingletonMeta._instances.values()):
            unload = getattr(instance, "unload", None)
            if callable(unload):
                unload()
        for path in entry.paths:
            try:
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
            except PermissionError as exc:
                raise ValueError(
                    f"文件正被占用，删除失败（可以关掉正在使用它的程序后重试）：{exc}") from exc
            print(f"🗑️ 已删除模型文件：{path}")
