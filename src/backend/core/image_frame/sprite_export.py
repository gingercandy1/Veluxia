import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

from PIL import Image


@dataclass
class SpriteSheetResult:
    sheet_path: Path
    atlas_path: Path
    columns: int
    rows: int
    frame_width: int
    frame_height: int


def _load_frames(frame_paths: Sequence[str]) -> List[Image.Image]:
    if not frame_paths:
        raise ValueError("没有可导出的帧")
    frames = []
    for path in frame_paths:
        if not Path(path).exists():
            raise FileNotFoundError(f"帧文件不存在: {path}")
        with Image.open(path) as image:
            frames.append(image.convert("RGBA"))
    return frames


def _union_alpha_bbox(frames: List[Image.Image]) -> Optional[tuple]:
    boxes = [box for box in (f.getchannel("A").getbbox() for f in frames) if box]
    if not boxes:
        return None
    return (min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes))


def _normalize(frames: List[Image.Image], trim: bool) -> List[Image.Image]:
    """所有帧统一成同一尺寸：trim 时按全部帧的并集包围盒裁（逐帧单独裁会让角色在帧间抖动），
    尺寸不一致的帧居中放进最大的单元格。"""
    if trim:
        box = _union_alpha_bbox(frames)
        if box:
            frames = [f.crop(box) for f in frames]
    width = max(f.width for f in frames)
    height = max(f.height for f in frames)
    normalized = []
    for frame in frames:
        if frame.size == (width, height):
            normalized.append(frame)
            continue
        cell = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        cell.paste(frame, ((width - frame.width) // 2, (height - frame.height) // 2))
        normalized.append(cell)
    return normalized


def export_sprite_sheet(frame_paths: Sequence[str], output_dir, name: str, columns: int = 0,
                        padding: int = 0, trim: bool = False, fps: int = 12) -> SpriteSheetResult:
    """把序列帧拼成一张精灵图并写出 atlas JSON（帧坐标、网格、fps），供引擎直接导入。

    columns=0 时取接近正方形的列数，避免一长条超出部分引擎的纹理尺寸上限。
    """
    frames = _normalize(_load_frames(frame_paths), trim)
    count = len(frames)
    columns = columns if columns > 0 else math.ceil(math.sqrt(count))
    columns = min(columns, count)
    rows = math.ceil(count / columns)
    cell_w, cell_h = frames[0].size

    sheet = Image.new("RGBA", (columns * cell_w + (columns + 1) * padding,
                               rows * cell_h + (rows + 1) * padding), (0, 0, 0, 0))
    atlas_frames = []
    for index, frame in enumerate(frames):
        x = padding + (index % columns) * (cell_w + padding)
        y = padding + (index // columns) * (cell_h + padding)
        sheet.paste(frame, (x, y))
        atlas_frames.append({"name": f"{name}_{index:04d}", "x": x, "y": y, "w": cell_w, "h": cell_h})

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sheet_path = output_dir / f"{name}_sheet.png"
    atlas_path = output_dir / f"{name}_sheet.json"
    sheet.save(sheet_path)
    atlas = {
        "image": sheet_path.name,
        "size": {"w": sheet.width, "h": sheet.height},
        "frame_size": {"w": cell_w, "h": cell_h},
        "columns": columns, "rows": rows, "padding": padding, "fps": fps,
        "frames": atlas_frames,
    }
    atlas_path.write_text(json.dumps(atlas, ensure_ascii=False, indent=2), encoding="utf-8")
    return SpriteSheetResult(sheet_path, atlas_path, columns, rows, cell_w, cell_h)


def export_frames(frame_paths: Sequence[str], output_dir, name: str, trim: bool = False) -> List[Path]:
    """导出编号的 PNG 序列（name_0000.png ...），统一尺寸、保留透明通道。"""
    frames = _normalize(_load_frames(frame_paths), trim)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for index, frame in enumerate(frames):
        path = output_dir / f"{name}_{index:04d}.png"
        frame.save(path)
        paths.append(path)
    return paths
