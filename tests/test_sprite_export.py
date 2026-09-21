import json

import pytest
from PIL import Image

from src.backend.core.image_frame.sprite_export import export_frames, export_sprite_sheet


def _frame(tmp_path, name, size=(10, 10), box=None, color=(255, 0, 0, 255)):
    """透明底图，box 内画一块不透明色块。"""
    image = Image.new("RGBA", size, (0, 0, 0, 0))
    if box:
        image.paste(Image.new("RGBA", (box[2] - box[0], box[3] - box[1]), color), box[:2])
    path = tmp_path / name
    image.save(path)
    return str(path)


def test_sheet_grid_and_atlas(tmp_path):
    frames = [_frame(tmp_path, f"f{i}.png", box=(0, 0, 10, 10)) for i in range(5)]
    result = export_sprite_sheet(frames, tmp_path / "out", name="run", columns=3, padding=2)

    sheet = Image.open(result.sheet_path)
    # 3 列 2 行，单元格 10x10，间隔 2：外圈也留 padding
    assert sheet.size == (3 * 10 + 4 * 2, 2 * 10 + 3 * 2)
    atlas = json.loads(open(result.atlas_path, encoding="utf-8").read())
    assert atlas["columns"] == 3 and atlas["rows"] == 2
    assert [f["name"] for f in atlas["frames"]][:2] == ["run_0000", "run_0001"]
    assert (atlas["frames"][3]["x"], atlas["frames"][3]["y"]) == (2, 14)
    assert atlas["size"] == {"w": sheet.width, "h": sheet.height}


def test_auto_columns_is_near_square(tmp_path):
    frames = [_frame(tmp_path, f"f{i}.png") for i in range(9)]
    result = export_sprite_sheet(frames, tmp_path / "out", name="a")
    assert result.columns == 3


def test_trim_uses_union_bbox_so_frames_stay_aligned(tmp_path):
    frames = [
        _frame(tmp_path, "a.png", box=(2, 2, 4, 4)),
        _frame(tmp_path, "b.png", box=(6, 5, 9, 8)),
    ]
    result = export_sprite_sheet(frames, tmp_path / "out", name="t", trim=True)
    # 并集包围盒 (2,2)-(9,8) → 7x6；逐帧单独裁会让角色在帧间抖动
    assert (result.frame_width, result.frame_height) == (7, 6)


def test_mixed_sizes_are_centered_in_the_largest_cell(tmp_path):
    frames = [_frame(tmp_path, "s.png", size=(4, 4), box=(0, 0, 4, 4)),
              _frame(tmp_path, "l.png", size=(8, 8), box=(0, 0, 8, 8))]
    result = export_sprite_sheet(frames, tmp_path / "out", name="m", columns=2)
    assert (result.frame_width, result.frame_height) == (8, 8)
    sheet = Image.open(result.sheet_path)
    assert sheet.getpixel((0, 0))[3] == 0      # 小图居中，角落是透明的
    assert sheet.getpixel((2, 2))[3] == 255


def test_export_frames_numbered_png(tmp_path):
    frames = [_frame(tmp_path, f"f{i}.png") for i in range(3)]
    paths = export_frames(frames, tmp_path / "out", name="walk")
    assert [p.name for p in paths] == ["walk_0000.png", "walk_0001.png", "walk_0002.png"]


def test_empty_and_missing_frames_raise(tmp_path):
    with pytest.raises(ValueError):
        export_sprite_sheet([], tmp_path / "out", name="x")
    with pytest.raises(FileNotFoundError):
        export_sprite_sheet([str(tmp_path / "nope.png")], tmp_path / "out", name="x")


def test_export_route(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from src.backend import server
    import src.backend.core.model_utils as model_utils

    monkeypatch.setattr(model_utils, "get_media_root", lambda: tmp_path)
    frames = [_frame(tmp_path, f"f{i}.png") for i in range(4)]
    with TestClient(server.app) as client:
        ok = client.post("/image_frame/export", json={
            "model_name": "SpriteSheet",
            "extra": {"frame_paths": frames, "name": "hit", "columns": 2, "output_dir": "o"}})
        bad = client.post("/image_frame/export", json={
            "model_name": "SpriteSheet", "extra": {"frame_paths": []}})
    assert ok.status_code == 200
    body = ok.json()
    assert body["columns"] == 2 and body["rows"] == 2 and len(body["frame_paths"]) == 4
    assert body["sheet_path"].startswith("/media/")
    assert bad.status_code == 400


class _FakeClient:
    def __init__(self, fail=False):
        self.fail, self.calls = fail, []

    def export_sprites(self, frame_paths, name, **options):
        self.calls.append((frame_paths, options))
        if self.fail:
            raise RuntimeError("boom")
        from types import SimpleNamespace
        return SimpleNamespace(sheet_path="/media/s.png", atlas_path="/media/s.json")

    def download_media(self, url, cache_dir):
        return "local" + url


def _worker(client, params):
    from src.app.param import GenerationRequest
    from src.app.work import ApiWorker
    from src.shared.enum_type import FactoryType

    request = GenerationRequest(FactoryType.Animation, "LTX-Video", "run", model_params=params)
    return ApiWorker(client, request, FactoryType.Animation)


def test_worker_attaches_sheet_and_atlas():
    from src.shared.schemas import AnimationResponse

    client = _FakeClient()
    result = AnimationResponse(video_path="v.mp4", frame_paths=["a.png", "b.png"])
    _worker(client, {"frame_rate": 24})._export_sprites(result)
    assert result.export_paths == ["local/media/s.png", "local/media/s.json"]
    assert client.calls[0][1]["fps"] == 24 and result.error is None


def test_worker_export_failure_keeps_animation_and_reports_error():
    from src.shared.schemas import AnimationResponse

    result = AnimationResponse(video_path="v.mp4", frame_paths=["a.png"])
    _worker(_FakeClient(fail=True), {})._export_sprites(result)
    assert result.ok and result.video_path == "v.mp4"
    assert "boom" in result.error and result.export_paths == []
