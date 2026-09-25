"""manifest.json 的读写（ADR 0004）：资源包进度的唯一来源。"""
import os
from pathlib import Path

from src.shared.schemas import Manifest

MANIFEST_NAME = "manifest.json"


def load_manifest(pack_dir: Path) -> Manifest:
    path = Path(pack_dir) / MANIFEST_NAME
    if not path.is_file():
        raise FileNotFoundError(f"资源包缺少 {MANIFEST_NAME}：{pack_dir}")
    return Manifest.model_validate_json(path.read_text(encoding="utf-8"))


def save_manifest(pack_dir: Path, manifest: Manifest) -> None:
    """先写临时文件再替换：批量任务可能整夜运行，中途崩溃也不能留下写了一半的 manifest。"""
    path = Path(pack_dir) / MANIFEST_NAME
    temp = path.with_suffix(".json.tmp")
    temp.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    os.replace(temp, path)
